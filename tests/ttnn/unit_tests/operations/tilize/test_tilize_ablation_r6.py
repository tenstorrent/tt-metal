# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6 ablation harness — PERF ONLY, no correctness asserts.

DO NOT DELETE. Ablation profiling (see /perf-measure) removes one stage's
PAYLOAD while keeping its synchronization, so the output is wrong by design and
this file must never assert on values.

The payload switches themselves are TEMPORARY kernel edits, not a shipped knob:
  * reads  — `noc_async_read(...)` in `read_sticks_for_tilize`
             (`ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl`) and in the
             reader's padded branch (`kernels/tilize_reader.cpp`)
  * writes — `noc_async_write<out_tile_bytes>(...)` in `kernels/tilize_writer.cpp`
  * compute— the `compute_kernel_lib::tilize` call in `kernels/tilize_compute.cpp`
Comment the payload line out (keeping the loop, the CB ops and the barrier),
profile, restore. The measured numbers are recorded in changelog.md.

    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_ablation_r6.py
"""

import pytest
import torch

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

CASES = [
    ((1, 1, 16384, 32), False, "tall_narrow"),
    ((1, 1, 1, 50304), True, "rough_c"),
]


@pytest.mark.parametrize("shape,pad,label", CASES, ids=[c[2] for c in CASES])
def test_ablation(device, shape, pad, label):
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pad_value = 0.0 if pad else None
    tt_output = tilize(tt_input, pad_value=pad_value)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid, pad_value=pad_value)
    print(
        f"\n[ablate {label}] {shape}: bw={plan.block_width_tiles} chunks={plan.num_w_chunks} "
        f"blocks={plan.num_blocks_total} cores={len(plan.assignment)}/{grid.x * grid.y}"
    )
    ttnn.synchronize_device(device)
