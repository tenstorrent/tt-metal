# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone Tracy bench for ttnn.experimental.topk_large_indices.

Underscore-prefixed so routine pytest collection skips it. Run under Tracy:

    unset TT_METAL_DPRINT_CORES && \
    TOPK_K=2048 TOPK_W=65536 TOPK_ROWS=1 python -m tracy -r -v \
        tests/ttnn/nightly/unit_tests/operations/experimental/_topk_large_indices_bench.py

Env knobs:
    TOPK_K     top-k value (default 2048)
    TOPK_W     row width W (default 65536)
    TOPK_ROWS  number of rows (default 1). ROWS=1 exercises the
               column-parallel path; ROWS=2 is the row-parallel proxy
               baseline (each row is reduced by a single core, so the
               per-op device kernel duration approximates the old
               single-core-per-row time for the same W).
    TOPK_ITERS measured iterations after warmup (default 5)
    TOPK_VALID optional valid_length (default: unset = full row)
    TOPK_SEED  torch.manual_seed value (default 0). Telemetry sweeps vary it
               for row diversity; timing runs keep the default.

Read the per-op "DEVICE KERNEL DURATION [ns]" from the newest
generated/profiler/reports/*/ops_perf_results_*.csv afterwards.
"""

import os

import torch
import ttnn

k = int(os.environ.get("TOPK_K", "2048"))
n = int(os.environ.get("TOPK_W", "65536"))
num_rows = int(os.environ.get("TOPK_ROWS", "1"))
iters = int(os.environ.get("TOPK_ITERS", "5"))
seed = int(os.environ.get("TOPK_SEED", "0"))
valid_length = os.environ.get("TOPK_VALID")
valid_length = int(valid_length) if valid_length else None

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

torch.manual_seed(seed)
torch_input = torch.randn(num_rows, n, dtype=torch.bfloat16)
tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

# Warmup: JIT compile + program cache fill.
out = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)
ttnn.synchronize_device(device)

for _ in range(iters):
    out = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)
ttnn.synchronize_device(device)

# Sanity: selected values must match torch's top-k value multiset (over the
# valid prefix when valid_length is set).
ref_input = torch_input if valid_length is None else torch_input[:, :valid_length]
indices = ttnn.to_torch(out, dtype=torch.uint32).to(torch.int64)
actual = torch.gather(torch_input.float(), dim=-1, index=indices)
ref, _ = torch.topk(ref_input.float(), k, dim=-1, largest=True, sorted=True)
assert torch.equal(actual.sort(dim=-1).values, ref.sort(dim=-1).values), "top-k value set mismatch"
print(f"OK k={k} W={n} rows={num_rows} iters={iters} valid={valid_length} seed={seed}")

ttnn.close_device(device)  # triggers profiler data dump
