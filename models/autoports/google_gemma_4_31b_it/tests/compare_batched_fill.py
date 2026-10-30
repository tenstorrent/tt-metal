"""Controlled ABBA prefill comparison sharing weights, inputs, cache and indices."""
import argparse
import json
import statistics
import time
import types
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.batched_fill_candidate import BatchedFillCandidate
from models.autoports.google_gemma_4_31b_it.tests.fusion_pre_batched_fill import HF_MODEL, HF_REVISION, FusedDecoder
from models.autoports.google_gemma_4_31b_it.tests.run_decoder import weights


def run(layer, batch, result):
    torch.set_num_threads(16)
    torch.manual_seed(419)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    sd = weights(layer)
    x = torch.randn(batch, 4096, cfg.hidden_size, dtype=torch.bfloat16)
    cos, sin = Gemma4TextRotaryEmbedding(cfg)(
        x, torch.arange(4096).expand(batch, -1), layer_type=cfg.layer_types[layer]
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh)

    try:
        model = FusedDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=layer, mesh_device=mesh)
        model.batch_indices = tt(torch.arange(batch, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache = model.allocate_cache(num_blocks=batch * 128)
        page = tt(torch.randperm(batch * 128).reshape(batch, 128).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        px, pc, ps = [tt(t.unsqueeze(1)) for t in (x, cos, sin)]
        methods = {"per_row": FusedDecoder.prefill_chunk_forward, "batched": BatchedFillCandidate.prefill_chunk_forward}
        values = {name: [] for name in methods}

        def once(name):
            model.prefill_chunk_forward = types.MethodType(methods[name], model)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            out = model.prefill_forward(px, cos=pc, sin=ps, kv_cache=cache, page_table=page)
            ttnn.synchronize_device(mesh)
            elapsed = (time.perf_counter() - start) * 1000
            del out
            return elapsed

        for name in methods:
            once(name)
        order = ["per_row", "batched", "batched", "per_row"] * 2
        for name in order:
            elapsed = once(name)
            values[name].append(elapsed)
            print(name, elapsed, flush=True)
        Path(result).write_text(
            json.dumps(
                dict(
                    layer_type=cfg.layer_types[layer],
                    batch=batch,
                    input_tokens=4096,
                    host_prefill_ms=values,
                    median_ms={k: statistics.median(v) for k, v in values.items()},
                    order=order,
                    scope="Host-wall candidate comparison; same allocations and warmed programs; not device telemetry",
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--result", required=True)
    a = p.parse_args()
    run(a.layer, a.batch, a.result)
