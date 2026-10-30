"""Same-fixture warmed whole-prefill host timings at logical chunk boundaries."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.run_decoder import weights
from models.autoports.google_gemma_4_31b_it.tt.fused_decoder import FusedDecoder
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION, OptimizedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    rows = []
    for layer in (0, 5):
        state = weights(layer)
        x = torch.load(
            f"bringup/artifacts/optimized_decoder_layer{layer}_inputs262144.pt", weights_only=True, mmap=True
        )[:, :4096].contiguous()
        cos, sin = Gemma4TextRotaryEmbedding(cfg)(x, torch.arange(4096)[None], layer_type=cfg.layer_types[layer])
        for cls in (FusedDecoder, OptimizedDecoder):
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
            try:
                model = cls.from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh)
                for length in (33, 1024, 1025, 2049, 4096):
                    blocks = (length + 31) // 32
                    pages = ttnn.from_torch(
                        torch.arange(blocks, dtype=torch.int32)[None],
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh,
                    )
                    cache = model.allocate_cache(num_blocks=blocks)
                    tx, tc, ts = [
                        ttnn.from_torch(
                            t[:, :length].unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                        )
                        for t in (x, cos, sin)
                    ]
                    kw = dict(cos=tc, sin=ts, kv_cache=cache, page_table=pages)
                    warm = model.prefill_forward(tx, **kw)
                    ttnn.synchronize_device(mesh)
                    del warm
                    timings = []
                    for _ in range(3):
                        start = time.perf_counter()
                        output = model.prefill_forward(tx, **kw)
                        ttnn.synchronize_device(mesh)
                        timings.append((time.perf_counter() - start) * 1e3)
                        del output
                    chunks = [min(1024, length - start) for start in range(0, length, 1024)]
                    rows.append(
                        dict(
                            layer_type=cfg.layer_types[layer],
                            implementation=cls.__name__,
                            logical_tokens=length,
                            batch=1,
                            physical_chunks=[(n + 31) // 32 * 32 for n in chunks],
                            chunk_count=len(chunks),
                            warmed_prefill_host_ms=timings,
                            median_host_ms=statistics.median(timings),
                        )
                    )
                    print(rows[-1], flush=True)
                    del tx, tc, ts, pages, cache, kw
                del model
            finally:
                ttnn.close_mesh_device(mesh)
    root = Path(__file__).resolve().parents[1]
    Path(args.result).write_text(
        json.dumps(
            dict(
                scope="Warmed B1 whole-prefill host time including every chunk and final synchronization; excludes setup/input upload/first compilation. Not device telemetry or full-model TTFT.",
                hf_revision=HF_REVISION,
                source_hashes={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), root / "tt/optimized_decoder.py", root / "tt/fused_decoder.py")
                },
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
