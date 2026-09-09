# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Valid-input base/fix benchmark. Run separately with and without Tracy profiling.
Profile example: python -m tracy -r -p -o OUTPUT bench.py --label fix --profile
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
import ttnn

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--output", required=True)
args = parser.parse_args()
if args.profile:
    from tracy import signpost
else:

    def signpost(*args):
        pass


device = ttnn.open_device(device_id=0)
device.enable_program_cache()
budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
# UINT32 index: two BF16 pages, plus four output pages for Wt_index == 1.
boundary_wt = budget // 2048 - 6
cases = [
    ("tiny", (1, 32), (1, 1), -1),
    ("row-full", (64, 128), (64, 32), -1),
    ("row-partial", (47, 95), (47, 19), -1),
    ("tiled-full", (64, 128), (64, 96), -1),
    ("tiled-partial", (47, 95), (47, 65), -1),
    ("public-valid", (64, 128), (32, 128), 0),
    ("large-row", (4096, 2048), (4096, 32), -1),
    ("large-tiled", (128, 4096), (128, 2048), -1),
    ("small-square", (32, 32), (32, 32), -1),
    ("wide-valid", (32, 131072), (32, 32), -1),
] + [(f"l1-{delta:+}", (64, 32 * (boundary_wt + delta)), (64, 32), -1) for delta in (-1, 0, 1)]
entries = {
    "codegen": ttnn._ttnn.operations.data_movement.gather_force_codegen,
    "native": ttnn._ttnn.operations.data_movement.gather_force_native,
}
rows = []
try:
    for name, shape, index_shape, dim in cases:
        torch.manual_seed(55847)
        x = torch.randn(shape, dtype=torch.bfloat16)
        index = torch.randint(0, shape[dim], index_shape, dtype=torch.int64)
        expected = torch.gather(x, dim, index)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        it = ttnn.from_torch(index.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
        for variant, entry in entries.items():
            for _ in range(5):
                out = entry(xt, dim, it)
                ttnn.synchronize_device(device)
            if not torch.equal(ttnn.to_torch(out), expected):
                if variant == "codegen":
                    raise AssertionError(f"{name}/{variant}: exact valid BF16 check failed")
                rows.append(dict(label=args.label, case=name, variant=variant, accuracy=False))
                Path(args.output).write_text(json.dumps(rows, indent=2))
                continue
            for window in range(5):
                times = []
                region = f"{args.label}_{name}_{variant}_{window}"
                signpost(region + "-start")
                for _ in range(10 if args.profile else 30):
                    start = time.perf_counter_ns()
                    out = entry(xt, dim, it)
                    ttnn.synchronize_device(device)
                    times.append((time.perf_counter_ns() - start) / 1000)
                    del out
                signpost(region + "-end")
                if args.profile:
                    ttnn.ReadDeviceProfiler(device)
                rows.append(
                    dict(
                        label=args.label,
                        case=name,
                        variant=variant,
                        window=window,
                        shape=shape,
                        index_shape=index_shape,
                        dim=dim,
                        arch=str(device.arch()),
                        profiled=args.profile,
                        median_host_us=statistics.median(times),
                        samples_host_us=times,
                    )
                )
                Path(args.output).write_text(json.dumps(rows, indent=2))
finally:
    ttnn.close_device(device)
