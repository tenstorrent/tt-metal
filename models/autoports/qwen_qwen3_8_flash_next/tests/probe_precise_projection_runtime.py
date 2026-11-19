# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight QSA precise projection shape, trace, and warmed latency probe."""

import argparse
import json
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_projection_precision import compact_metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--source-cache", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt")
    )
    parser.add_argument("--chunk-start", type=int, default=222272)
    parser.add_argument("--counts", default="1,3,7,31,32")
    args = parser.parse_args()
    torch.set_num_threads(8)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    _, hf, _, weights, provenance = load_reference(args.snapshot)
    report = {"scope": "Component diagnostic, not full decoder performance", "provenance": provenance, "shapes": {}}
    capture = {}

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.precise_projection import PreciseIndexerProjection

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 * 1024 * 1024)
    try:
        loader = WeightLoader(weights, mesh)
        projection = PreciseIndexerProjection(loader)
        for count in map(int, args.counts.split(",")):
            hidden = source["hidden"][:, args.chunk_start : args.chunk_start + count]
            expected = hf.indexer.index_qk_proj(hidden)
            x = loader.tensor(hidden.unsqueeze(0))
            with runtime_audit():
                actual_device = projection(x)
            actual = ttnn.to_torch(actual_device).reshape_as(expected)
            eager_times = []
            for _ in range(3):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                actual_device = projection(x)
                ttnn.synchronize_device(mesh)
                eager_times.append(time.perf_counter() - start)
            item = {
                "accuracy": compact_metrics(expected, actual),
                "warmed_eager_seconds": eager_times,
                "runtime_fallback_audit": "clean",
            }
            if count in (1, 32):
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                with runtime_audit():
                    traced = projection(x)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                traced_times = []
                for _ in range(3):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    traced_times.append(time.perf_counter() - start)
                traced_host = ttnn.to_torch(traced).reshape_as(expected)
                item["trace_accuracy"] = compact_metrics(expected, traced_host)
                item["trace_eager_bitwise_equal"] = torch.equal(actual, traced_host)
                item["warmed_traced_seconds"] = traced_times
                ttnn.release_trace(mesh, trace_id)
            report["shapes"][str(count)] = item
            capture[str(count)] = {"expected": expected, "actual": actual}
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            torch.save(capture, args.capture)
            print(json.dumps({"count": count, **item}), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
