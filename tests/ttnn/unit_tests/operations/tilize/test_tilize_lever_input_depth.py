# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lever measurement: `INPUT_DEPTH_ROWS` (the reader/compute overlap depth).

DO NOT DELETE. This is the harness behind op_design.md's "Overlap (input depth)"
lamp. Depth 2 is the catalog-measured default, but the catalog has NOTHING past
2, and on the wide-chunk regime one tile-row is `tile_h * block_row_bytes` read
behind a single barrier while compute is nearly free — so the largest legal
per-tile-row read may serialize movement against compute in a way a third slot
would hide. The lamp names the two shapes to check; both are here.

Depth 1 is included as the lower bound even though it is illegal to *ship*
(`read_sticks_for_tilize` and `compute_kernel_lib::tilize` both assert the CB
holds at least `block_width_tiles` pages, which depth 1 exactly meets, so it
runs — it just removes all overlap). It is the reference point that says how
much the double-buffering is worth at all.

Run for numbers:
    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_input_depth.py
Correctness needs a plain (non-profile) run.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

# The two shapes the lamp names: bw=16 (1 KiB reads) and bw=64 (4 KiB reads).
SHAPES = [((1, 1, 32, 32768), "bw16_1KiB_reads"), ((1, 1, 2048, 2048), "bw64_4KiB_reads")]
DEPTHS = [1, 2, 3, 4]  # 2 is the production default


@pytest.mark.parametrize("shape,shape_id", SHAPES, ids=[i for _, i in SHAPES])
@pytest.mark.parametrize("depth", DEPTHS, ids=[f"d{d}" for d in DEPTHS])
def test_input_depth_lever(device, shape, shape_id, depth, monkeypatch):
    monkeypatch.setattr(pd, "INPUT_DEPTH_ROWS", depth)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})
    grid = device.compute_with_storage_grid_size()

    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)

    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    assert plan.input_depth_rows == depth
    print(
        f"\n[lever INPUT_DEPTH_ROWS={depth}] {shape}: bw={plan.block_width_tiles} "
        f"read={plan.block_row_bytes} B/stick, in_cb={plan.input_cb_pages} pages, "
        f"cores={len(plan.assignment)}/{grid.x * grid.y}, "
        f"L1/core={plan.l1_per_core_bytes // 1024} KiB"
    )
    assert_with_pcc(torch_input.float(), ttnn.to_torch(tt_output).float(), 0.995)
