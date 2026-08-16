# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone Tracy bench for the large-k ttnn.topk routing (PR 2).

Underscore-prefixed so routine pytest collection skips it. Run under Tracy:

    unset TT_METAL_DPRINT_CORES && \
    TOPK_K=512 TOPK_W=65536 python -m tracy -r -v \
        tests/ttnn/nightly/unit_tests/operations/experimental/_topk_routed_bench.py

Env knobs:
    TOPK_K        top-k value (default 512)
    TOPK_W        row width (default 65536)
    TOPK_ROWS     logical rows (default 32; 1 engages the column-parallel
                  factory inside topk_large_indices)
    TOPK_LARGEST  1 (default) takes the routed path; 0 falls back to the
                  stock single-core factory on the same shape — the
                  pre-routing baseline arm for the A/B
    TOPK_ITERS    measured iterations after warmup (default 5)

The routed path is a composite (untilize + TopkLargeIndices + gather +
2x tilize + eq + where + typecast + slice): sum DEVICE KERNEL DURATION over
all ops of one iteration when comparing against the baseline's single row.
"""

import os
import time

import torch
import ttnn

k = int(os.environ.get("TOPK_K", "512"))
n = int(os.environ.get("TOPK_W", "65536"))
num_rows = int(os.environ.get("TOPK_ROWS", "32"))
largest = os.environ.get("TOPK_LARGEST", "1") != "0"
iters = int(os.environ.get("TOPK_ITERS", "5"))

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

torch.manual_seed(0)
torch_input = torch.randn(1, 1, num_rows, n, dtype=torch.bfloat16)
tt_input = ttnn.from_torch(torch_input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

# Warmup: JIT compile + program cache fill for every op in the composite.
values, indices = ttnn.topk(tt_input, k, dim=-1, largest=largest, sorted=True)
ttnn.synchronize_device(device)

t0 = time.perf_counter()
for _ in range(iters):
    values, indices = ttnn.topk(tt_input, k, dim=-1, largest=largest, sorted=True)
ttnn.synchronize_device(device)
t1 = time.perf_counter()

# Sanity: value multiset must match torch's top-k values.
actual = ttnn.to_torch(values)
ref, _ = torch.topk(torch_input, k, dim=-1, largest=largest, sorted=True)
assert torch.equal(
    actual.sort(dim=-1, descending=True).values, ref.sort(dim=-1, descending=True).values
), "top-k value set mismatch"
print(
    f"OK k={k} W={n} rows={num_rows} largest={largest} iters={iters} "
    f"e2e={(t1 - t0) / iters * 1e3:.3f} ms/iter (host wall clock incl. dispatch)"
)

ttnn.close_device(device)  # triggers profiler data dump
