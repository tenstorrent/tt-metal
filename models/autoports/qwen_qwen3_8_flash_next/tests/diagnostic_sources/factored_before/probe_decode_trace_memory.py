"""Reduced real-layer trace retention measurement; device owner runs serially.

No accuracy, throughput, capacity-limit, or full48-fit claim. Retain model and
sampling traces for widths1,2,4,8,16,32 without eviction. CPU source preparation
alone does not run this probe. Bounded device-owner command:

 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 timeout 1200 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_decode_trace_memory \
 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/decode_trace_memory_reduced_v1.json
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    assert "TT_METAL_WATCHER" not in os.environ
    for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
        assert os.environ.get(name) in (None, "", "0")
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
    from models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_accuracy import memory

    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    result = {
        "passed": False,
        "scope": "Reduced real0/1/3 retained decode trace memory only; not a full48 capacity proof",
        "capacity": 128,
        "configured_batch_size": 32,
        "shared_kv": True,
        "widths": [1, 2, 4, 8, 16, 32],
        "measurements": [],
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("tt/generator.py", "tt/model.py", "tt/multichip_decoder.py")
        },
        "memory_semantics": "Allocator deltas include retained trace command buffers and other retained allocation effects; not isolated trace-buffer bytes",
    }
    mesh = generator = None

    def snapshot():
        ttnn.synchronize_device(mesh)
        view = memory(mesh)
        return {
            **view,
            "allocated_bytes_per_rank": view["num_banks"] * view["total_bytes_allocated_per_bank"],
            "free_bytes_per_rank": view["num_banks"] * view["total_bytes_free_per_bank"],
            "largest_free_block_bytes_per_bank": view["largest_contiguous_bytes_free_per_bank"],
        }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        generator = build_generator(
            root, mesh, layer_indices=(0, 1, 3), max_seq_len=128, max_batch_size=32, defer_kv_cache=True
        )
        pool = generator.allocate_kv_cache_per_layer([((128, 1, 32, 256), torch.bfloat16, 0)])
        assert len(pool.pairs) == 1
        result["memory_loaded"] = snapshot()
        generator.prepare()
        result["memory_prepared"] = snapshot()
        assert not generator.traces and not generator.sampling_traces and not generator.prefill_traces
        programs = mesh.num_program_cache_entries()
        for width in result["widths"]:
            slots = tuple(range(width))
            before = snapshot()
            trace = generator._capture(slots)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            model = snapshot()
            generator._sample(slots)
            sampling = snapshot()
            item = {
                "width": width,
                "before": before,
                "after_model": model,
                "after_sampling": sampling,
                "model_retained_delta_bytes_per_rank": model["allocated_bytes_per_rank"]
                - before["allocated_bytes_per_rank"],
                "sampling_retained_delta_bytes_per_rank": sampling["allocated_bytes_per_rank"]
                - model["allocated_bytes_per_rank"],
                "model_trace_count": len(generator.traces),
                "sampling_trace_count": len(generator.sampling_traces),
            }
            result["measurements"].append(item)
            assert generator._capture(tuple(reversed(slots))) == trace
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            generator._sample(tuple(reversed(slots)))
            repeat = snapshot()
            assert repeat["allocated_bytes_per_rank"] == sampling["allocated_bytes_per_rank"], (
                "Repeated identical signature retained more allocation",
                width,
            )
            assert mesh.num_program_cache_entries() == programs
            item["reverse_order_reuses_trace_and_has_zero_retained_growth"] = True
            print(
                json.dumps(
                    {
                        "width": width,
                        "model_bytes": item["model_retained_delta_bytes_per_rank"],
                        "sampling_bytes": item["sampling_retained_delta_bytes_per_rank"],
                    }
                ),
                flush=True,
            )
            save()
        result["memory_final"] = snapshot()
        result["passed"] = True
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        save()
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
