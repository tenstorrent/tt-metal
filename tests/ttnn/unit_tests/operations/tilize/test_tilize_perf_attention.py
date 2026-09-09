# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 perf harness — the `attention:` LOOSE_CASE `[1,1,32,16384]`.

DO NOT DELETE. This is the measurement harness behind
op_requirements.md's "Refinement 3 — Speed up the perf-flagged attention
profile". It runs the flagged config EXACTLY (bf16 -> bf16, interleaved
DRAM -> DRAM, rank 4, tile-aligned, 32x32 tile) plus the config-spanning guard
set the refinement's Done-when names, so a `--profile` run produces one CSV row
per distinct kernel path in a fixed order.

Run for numbers (device kernel ns per row, in the parametrize order):
    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_perf_attention.py

Correctness needs a plain (non-profile) run — the Tracy wrapper masks the exit
code. Every case here asserts BIT IDENTITY (`torch.equal`), because tilize does
no arithmetic: any deviation is a bug, not a precision budget.
"""

import pytest
import torch

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

# The flagged config, exactly. R = 1 x C = 512 — DeepSeek-V3 MLA `wo_tilize`.
ATTENTION_SHAPE = (1, 1, 32, 16384)

# One representative per distinct kernel path x placement (the Done-when list).
GUARD_SHAPES = [
    ((1, 1, 2048, 64), "grid2d_full_width"),  # num_w_chunks == 1
    ((1, 1, 32, 2048), "grid2d_width_chunked"),
    ((1, 1, 2048, 2048), "square_large"),  # widest block
    ((1, 1, 16384, 32), "tall_narrow"),  # bw == 1, write_rows_per_barrier is the knob
    ((1, 1, 1024, 1024), "square_mid"),  # 1024 B read at 1 wave — the floor's other side
    ((1, 1, 32, 32768), "short_wide_wide"),  # R == 1 at C == 1024
]


def _run_identity(device, shape, label):
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
    tt_output = tilize(tt_input)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    print(
        f"\n[perf {label}] {shape}: R={plan.tensor_row_blocks} C={plan.tensor_col_tiles} "
        f"bw={plan.block_width_tiles} rows/blk={plan.tensor_row_blocks // plan.num_row_groups} "
        f"blocks={plan.num_blocks_total} cores={len(plan.assignment)}/{grid.x * grid.y} "
        f"in_cb={plan.input_cb_pages}p out_cb={plan.output_cb_pages}p "
        f"L1/core={plan.l1_per_core_bytes // 1024}KiB"
    )
    got = ttnn.to_torch(tt_output)
    assert torch.equal(got, torch_input), f"{label}: tilize is not bit-identical"
    return plan


def test_perf_attention_flagged(device):
    """The mandatory perf target — measured on this row of the CSV."""
    plan = _run_identity(device, ATTENTION_SHAPE, "attention")
    # Pin the geometry the measurement is taken at, so a later plan change that
    # silently drops occupancy shows up here rather than in the ns.
    assert plan.tensor_row_blocks == 1 and plan.tensor_col_tiles == 512
    assert len(plan.assignment) == 64, "the flagged number is only meaningful at 64/64 cores"


@pytest.mark.parametrize("shape,label", GUARD_SHAPES, ids=[g[1] for g in GUARD_SHAPES])
def test_perf_guard_set(device, shape, label):
    _run_identity(device, shape, label)


def test_perf_guard_padded(device):
    """Refinement 2's path. `pad_active` runs the SAME solved column cut, so the
    wave rule reaches it; the segmented+filled reader branch is the difference."""
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    shape = (8, 1, 249, 2048)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input, pad_value=0.0)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid, pad_value=0.0)
    assert plan.pad_active
    print(
        f"\n[perf padded] {shape}: bw={plan.block_width_tiles} chunks={plan.num_w_chunks} "
        f"cores={len(plan.assignment)}/{grid.x * grid.y} L1/core={plan.l1_per_core_bytes // 1024}KiB"
    )
    assert torch.equal(ttnn.to_torch(tt_output), torch_input)


def test_perf_guard_sharded(device):
    """Refinement 1's path. The shard partition DRIVES the block grid, so the
    wave rule (solved branch only) must not reach it — bw stays the shard width."""
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    shape = (1, 1, 2048, 2048)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    mem = ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem,
    )
    tt_output = tilize(tt_input)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    print(
        f"\n[perf sharded] {shape}: bw={plan.block_width_tiles} chunks={plan.num_w_chunks} "
        f"cores={len(plan.assignment)}/{grid.x * grid.y} native_in={plan.input_native}"
    )
    # A shard IS the block — the solved column cut (and its wave rule) is bypassed.
    assert plan.block_width_tiles == 64 and plan.num_w_chunks == 1
    assert torch.equal(ttnn.to_torch(tt_output), torch_input)
