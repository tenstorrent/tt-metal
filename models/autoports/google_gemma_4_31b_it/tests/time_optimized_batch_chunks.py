"""Adapt large-batch prefill to fit L1 by reducing logical chunk size."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn

from ..tt.optimized_decoder import HF_MODEL, HF_REVISION, OptimizedDecoder
from .optimized_prefill_candidate import PrefillCandidate
from .run_decoder import weights


class SmallChunkCandidate(OptimizedDecoder):
    prefill_forward = PrefillCandidate.prefill_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    rows = []
    for layer in (0, 5):
        state = weights(layer)
        x = (
            torch.load(
                f"bringup/artifacts/optimized_decoder_layer{layer}_inputs262144.pt", weights_only=True, mmap=True
            )[:, :4096]
            .expand(32, -1, -1)
            .contiguous()
        )
        cos, sin = Gemma4TextRotaryEmbedding(cfg)(
            x, torch.arange(4096)[None].expand(32, -1), layer_type=cfg.layer_types[layer]
        )
        reference = None
        for chunk_size in (1024, 32):
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
            try:
                cls = OptimizedDecoder if chunk_size == 1024 else SmallChunkCandidate
                model = cls.from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh)
                if chunk_size != 1024:
                    model.geometry["chunk"] = chunk_size
                pages = ttnn.from_torch(
                    torch.arange(32 * 128, dtype=torch.int32).reshape(32, 128),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                )
                cache = model.allocate_cache(num_blocks=32 * 128)
                tx, tc, ts = [
                    ttnn.from_torch(t.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                    for t in (x, cos, sin)
                ]
                kw = dict(cos=tc, sin=ts, kv_cache=cache, page_table=pages)
                warm = model.prefill_forward(tx, **kw)
                ttnn.synchronize_device(mesh)
                del warm
                timings = []
                for trial in range(3):
                    start = time.perf_counter()
                    output = model.prefill_forward(tx, **kw)
                    ttnn.synchronize_device(mesh)
                    timings.append((time.perf_counter() - start) * 1000)
                    if trial == 2:
                        tail = ttnn.slice(output, (0, 0, 4064, 0), (32, 1, 4096, cfg.hidden_size))
                        actual = ttnn.to_torch(tail).float().flatten()
                        del tail
                    del output
                if reference is None:
                    reference = actual
                pcc = torch.corrcoef(torch.stack((reference, actual)))[0, 1].item()
                assert pcc >= 0.995, pcc
                rows.append(
                    dict(
                        layer_type=cfg.layer_types[layer],
                        batch=32,
                        logical_tokens=4096,
                        physical_chunk_tokens=chunk_size,
                        chunk_count=4096 // chunk_size,
                        uses_private_l1=chunk_size == 32,
                        host_ms=timings,
                        median_host_ms=statistics.median(timings),
                        last32_vs_default_pcc=pcc,
                    )
                )
                print(rows[-1], flush=True)
                del model, tx, tc, ts, cache, pages, kw
            finally:
                ttnn.close_mesh_device(mesh)
    Path(args.result).write_text(
        json.dumps(
            dict(
                scope="Warmed whole-prefill host timings, real pinned weights and recorded HF activations. Last32 output rows compared with delivered default; not device telemetry.",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
