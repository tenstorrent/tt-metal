# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-run the 18 configurations from PR #52074, comment 5436334701."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

import ttnn


CASES = [
    ((1, 1, 1536, 32768), (1, 1, 1536, 256), -1),
    ((1, 1, 64, 32000), (1, 1, 64, 3200), -1),
    ((1, 1, 32, 15360), (1, 1, 32, 7680), -1),
    ((1, 1, 20, 31990), (1, 1, 20, 7670), -1),
    ((128, 256), (128, 128), -1),
    ((1, 1, 256, 256), (1, 1, 256, 128), -1),
    ((1, 64, 128), (1, 64, 64), -1),
    ((64, 128), (64, 64), -1),
    ((1, 1, 128, 128), (1, 1, 128, 64), -1),
    ((1, 1, 128, 128), (1, 1, 64, 128), -2),
    ((1, 1, 32, 64), (1, 1, 32, 32), -1),
    ((32, 64), (32, 32), -1),
    ((1, 32, 64), (1, 32, 32), -1),
    ((1, 1, 64, 64), (1, 1, 64, 32), -1),
    ((1, 1, 64, 128), (1, 1, 32, 128), -2),
    ((1, 1, 32, 64), (1, 1, 16, 64), -2),
    ((1, 1, 4352, 128), (1, 1, 4352, 96), -1),
    ((1, 151936), (1, 151936), -1),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    if args.profile:
        from tracy import signpost
    else:

        def signpost(*unused):
            pass

    torch.set_num_threads(1)
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    entries = {
        "native": ttnn._ttnn.operations.data_movement.gather_force_native,
        "codegen": ttnn._ttnn.operations.data_movement.gather_force_codegen,
    }
    rows = []
    try:
        for case, (shape, index_shape, dim) in enumerate(CASES, 1):
            if args.case is not None and args.case != case:
                continue
            torch.manual_seed(55847 + case)
            x = torch.randn(shape, dtype=torch.bfloat16)
            index = torch.randint(0, shape[dim], index_shape, dtype=torch.int64)
            expected = torch.gather(x, dim, index)
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            it = ttnn.from_torch(index.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                public = ttnn.gather(xt, dim, it)
                ttnn.synchronize_device(device)
            finally:
                graph = ttnn.graph.end_graph_capture()
            factories = [
                n["params"]["program_factory_type"]
                for n in graph
                if n.get("node_type") == "function_start"
                and "GatherCodegenProgramFactory" in n.get("params", {}).get("program_factory_type", "")
            ]
            assert len(factories) == 1, (case, "public call did not select codegen", factories)
            assert torch.equal(ttnn.to_torch(public), expected), (case, "public accuracy")
            del public
            for variant, entry in entries.items():
                for _ in range(5):
                    out = entry(xt, dim, it)
                    ttnn.synchronize_device(device)
                assert torch.equal(ttnn.to_torch(out), expected), (case, variant, "exact BF16 accuracy")
                del out
            if args.profile:
                ttnn.ReadDeviceProfiler(device)
            for window in range(6):
                # Adjacent paired windows and alternating order reduce systematic
                # host drift between native and codegen measurements.
                order = ("native", "codegen") if window % 2 == 0 else ("codegen", "native")
                for variant in order:
                    times = []
                    region = f"{args.label}_case{case:02d}_{variant}_{window}"
                    signpost(region + "-start")
                    count = 10 if args.profile else 30
                    for _ in range(count):
                        start = time.perf_counter_ns()
                        out = entries[variant](xt, dim, it)
                        ttnn.synchronize_device(device)
                        times.append((time.perf_counter_ns() - start) / 1000)
                        del out
                    signpost(region + "-end")
                    if args.profile:
                        ttnn.ReadDeviceProfiler(device)
                    rows.append(
                        dict(
                            label=args.label,
                            case=f"case{case:02d}",
                            variant=variant,
                            window=window,
                            shape=shape,
                            index_shape=index_shape,
                            dim=dim,
                            arch=str(device.arch()),
                            factory=factories[0],
                            profiled=args.profile,
                            calls=count,
                            median_host_us=statistics.median(times),
                            samples_host_us=times,
                            exact_bf16=True,
                        )
                    )
                    Path(args.output).write_text(json.dumps(rows, indent=2))
            print(f"PASS {case}: {shape} / {index_shape}, dim={dim}, {factories[0]}", flush=True)
            del xt, it
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
