"""Real-weight fused-versus-functional equivalence, with identical paged state."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.run_decoder import pcc, weights
from models.autoports.google_gemma_4_31b_it.tt.functional_decoder import HF_MODEL, HF_REVISION, FunctionalDecoder
from models.autoports.google_gemma_4_31b_it.tt.fused_decoder import FusedDecoder


def run(layer, batch, length, result, prefill_only=False):
    torch.set_num_threads(16)
    torch.manual_seed(914)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    state = weights(layer)
    x = torch.randn(batch, length + 1, cfg.hidden_size, dtype=torch.bfloat16)
    cos, sin = Gemma4TextRotaryEmbedding(cfg)(
        x, torch.arange(length + 1).expand(batch, -1), layer_type=cfg.layer_types[layer]
    )
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=m)

    try:
        models = [
            c.from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=m)
            for c in (FunctionalDecoder, FusedDecoder)
        ]
        n = (length + 32) // 32
        pages = tt(torch.randperm(batch * n).reshape(batch, n).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        caches = [model.allocate_cache(num_blocks=batch * n) for model in models]
        px, pc, ps = tt(x[:, :length].unsqueeze(1)), tt(cos[:, :length].unsqueeze(1)), tt(sin[:, :length].unsqueeze(1))
        outputs = [
            ttnn.to_torch(model.prefill_forward(px, cos=pc, sin=ps, kv_cache=cache, page_table=pages))
            for model, cache in zip(models, caches)
        ]
        pre = pcc(*outputs)
        dec = None
        if not prefill_only:
            dx, dc, ds = (
                tt(x[:, length].reshape(1, 1, batch, -1)),
                tt(cos[:, length].reshape(1, batch, 1, -1)),
                tt(sin[:, length].reshape(1, batch, 1, -1)),
            )
            pos = tt(torch.full((batch,), length, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            outputs = [
                ttnn.to_torch(
                    model.decode_forward(dx, cos=dc, sin=ds, kv_cache=cache, page_table=pages, current_pos=pos)
                )
                for model, cache in zip(models, caches)
            ]
            dec = pcc(*outputs)
        cache_pcc = [pcc(ttnn.to_torch(caches[0][i]), ttnn.to_torch(caches[1][i])) for i in (0, 1)]
        assert min([pre, *cache_pcc] + ([] if prefill_only else [dec])) >= 0.995, (pre, dec, cache_pcc)
        Path(result).write_text(
            json.dumps(
                dict(
                    layer_type=cfg.layer_types[layer],
                    batch=batch,
                    seq=length,
                    prefill_pcc=pre,
                    decode_pcc=dec,
                    prefill_only=prefill_only,
                    cache_pcc=cache_pcc,
                    real_weights=True,
                    reference="FunctionalDecoder",
                ),
                indent=2,
            )
        )
    finally:
        ttnn.close_mesh_device(m)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=1025)
    p.add_argument("--result", required=True)
    p.add_argument("--prefill-only", action="store_true")
    a = p.parse_args()
    run(a.layer, a.batch, a.seq, a.result, a.prefill_only)
