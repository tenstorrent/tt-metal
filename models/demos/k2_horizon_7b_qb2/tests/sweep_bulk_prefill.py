"""Current-policy bulk-prefill block/grid controls on real retained TP4 inputs."""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import K2Generator
from .sweep_short_prefill import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=257)
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
    result = {"pass": False, "prompt_len": args.length, "mesh": [1, 4], "precision_changed": False, "rows": []}
    try:
        gen = K2Generator(mesh, override_num_layers=2)
        retained = {}
        for index, layer in enumerate(gen.model.layers):
            fused = layer._fused_prefill_projection
            linear = layer._linear

            def record_fused(x, w, *, swiglu=False, original=fused, index=index):
                output = original(x, w, swiglu=swiglu)
                retained[(index, "mlp" if swiglu else "qkv")] = (x, w, output, swiglu)
                return output

            def record_linear(x, w, *, original=linear, index=index, layer=layer):
                output = original(x, w)
                if index == 0 and (w is layer.wo or w is layer.wdown):
                    retained[(index, "o" if w is layer.wo else "down")] = (x, w, output, False)
                return output

            layer._fused_prefill_projection = record_fused
            layer._linear = record_linear
        prompt = gen.tokenizer.encode(
            "The sky appears blue because sunlight scatters in the atmosphere. " * args.length
        )[: args.length]
        logits = gen.prefill_logits(prompt)
        ttnn.synchronize_device(mesh)
        del logits
        native_agmm = ttnn.experimental.all_gather_minimal_matmul_async
        for (index, role), (x, w, baseline, swiglu) in list(retained.items()):
            layer = gen.model.layers[index]
            reference = read(baseline)
            fused = role in ("qkv", "mlp")
            configs = [(8, 8, 8, 11, 9)] if fused else [(4, 8, 16, 11, 10)]
            configs += [
                (m, k, n, gx, gy)
                for m, k, n in [(4, 8, 8), (4, 16, 8), (4, 8, 16), (4, 16, 16), (4, 4, 32), (8, 8, 8), (8, 16, 8)]
                for gx, gy in [(11, 9 if fused else 10), (8, 8)]
            ]
            configs = list(dict.fromkeys(configs))
            for config_tuple in configs:
                m, k, n, gx, gy = config_tuple
                row = {
                    "layer": index,
                    "role": role,
                    "input_shape": list(x.shape),
                    "weight_shape": list(w.shape),
                    "weight_dtype": str(w.dtype),
                    "config": list(config_tuple),
                    "fused": fused,
                }
                try:
                    config = ttnn.MinimalMatmulConfig(
                        M_block_size=m,
                        K_block_size=k,
                        N_block_size=n,
                        subblock_h=2,
                        subblock_w=2,
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    )

                    def agmm(*a, **kw):
                        kw["config"] = config
                        kw["num_workers_per_link"] = math.ceil(gx / layer.num_links)
                        return native_agmm(*a, **kw)

                    def body():
                        if fused:
                            with patch.object(ttnn.experimental, "all_gather_minimal_matmul_async", agmm):
                                # Call the production method without the recording wrapper.
                                return type(layer)._fused_prefill_projection(layer, x, w, swiglu=swiglu)
                        return ttnn.experimental.minimal_matmul(
                            x, w, config=config, compute_kernel_config=layer._weight_compute(w), dtype=ttnn.bfloat16
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
                        for _ in range(32):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        timings.append((time.perf_counter() - started) * 1e6 / 32)
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
                print("BULK_PREFILL", json.dumps(row), flush=True)
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
