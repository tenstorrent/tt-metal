"""Precision-locked short-prefill geometry using real TP4 model activations."""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def read(tensor):
    host = tensor.cpu(blocking=True)
    return [ttnn.to_torch(shard).float().clone() for shard in ttnn.get_device_tensors(host)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert not os.environ.get("TT_METAL_DEVICE_PROFILER") and not os.environ.get("TT_METAL_WATCHER")
    torch.set_num_threads(16)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    trace = None
    result = {"pass": False, "prompt_len": 128, "mesh": [1, 4], "precision_changed": False, "rows": []}
    try:
        gen = K2Generator(mesh, override_num_layers=2)
        retained = {}
        for index, layer in enumerate(gen.model.layers):
            original = layer._linear
            names = {id(getattr(layer, "w" + role)): role for role in ("qkv", "o", "gate", "up", "down")}

            def record(x, w, *, original=original, index=index, names=names):
                output = original(x, w)
                role = names[id(w)]
                if (index == 0 and role != "up") or (index == 1 and role == "gate"):
                    retained[(index, role)] = (x, w, output)
                return output

            layer._linear = record
        prompt = gen.tokenizer.encode("The sky appears blue because sunlight scatters in the atmosphere. " * 30)[:128]
        logits = gen.prefill_logits(prompt)
        ttnn.synchronize_device(mesh)
        del logits
        for (index, role), (x, w, baseline) in retained.items():
            layer = gen.model.layers[index]
            reference = read(baseline)
            k, n = tuple(w.shape)[-2:]
            candidates = [{"kind": "auto", "input": "DRAM"}]
            for cores in (8, 16, 32, 64):
                for block in (2, 4, 8, 12, 16, 24, 32, 48, 64):
                    if (k // 32) % block:
                        continue
                    for memory in ("DRAM", "L1"):
                        candidates.append({"kind": "1d", "cores": cores, "block": block, "input": memory})
            for candidate in candidates:
                row = {"layer": index, "role": role, "shape": [128, k, n], "weight_dtype": str(w.dtype), **candidate}
                try:
                    config = None
                    if candidate["kind"] == "1d":
                        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(8, candidate["cores"] // 8),
                            in0_block_w=candidate["block"],
                            out_subblock_h=4,
                            out_subblock_w=1,
                            per_core_M=4,
                            per_core_N=math.ceil(n / 32 / candidate["cores"]),
                            fuse_batch=True,
                            mcast_in0=True,
                        )

                    def body():
                        activation = (
                            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if candidate["input"] == "L1" else x
                        )
                        return ttnn.linear(
                            activation,
                            w,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=layer._weight_compute(w),
                            program_config=config,
                        )

                    warm = body()
                    actual = read(warm)
                    row["exact_all_ranks"] = all(torch.equal(a, b) for a, b in zip(actual, reference))
                    row["min_pcc"] = min(
                        float(torch.corrcoef(torch.stack((a.flatten(), b.flatten())))[0, 1])
                        for a, b in zip(actual, reference)
                    )
                    row["max_abs_error"] = max(float((a - b).abs().max()) for a, b in zip(actual, reference))
                    del warm
                    with gen._capture() as captured:
                        output = body()
                    trace = captured
                    timings = []
                    for _ in range(3):
                        ttnn.synchronize_device(mesh)
                        started = time.perf_counter()
                        for _ in range(64):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        timings.append((time.perf_counter() - started) * 1e6 / 64)
                    replay = read(output)
                    row["eager_replay_exact"] = all(torch.equal(a, b) for a, b in zip(actual, replay))
                    assert row["eager_replay_exact"]
                    row.update(pass_geometry=True, traced_us=timings, median_us=statistics.median(timings))
                except RuntimeError as error:
                    row.update(pass_geometry=False, error=str(error))
                finally:
                    if trace is not None:
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                        trace = None
                    if "output" in locals():
                        del output
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print("SHORT_PREFILL", json.dumps(row), flush=True)
        result["pass"] = True
    finally:
        if trace is not None:
            ttnn.synchronize_device(mesh)
            ttnn.release_trace(mesh, trace)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
