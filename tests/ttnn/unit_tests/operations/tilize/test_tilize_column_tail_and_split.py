# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6 correctness: the RAGGED COLUMN TAIL and the SPLIT READER.

DO NOT DELETE. Both are structural changes that no earlier test reaches:

  * the ragged tail puts a SECOND core range in the program, running the same
    three kernels at its own `block_width_tiles` / `col_tile_offset` / CB sizes.
    Only a rough `C` (one that `block_width_tiles` does not divide) produces it.
  * the split reader makes the WRITER kernel read the trailing tile-rows of
    every block into its own input CB, so the block reaches compute as two
    sub-blocks. Only a small read transaction (<= `SPLIT_READER_MAX_ROW_BYTES`)
    on a block at least two tile-rows tall turns it on.

tilize does no arithmetic, so every case asserts BIT IDENTITY. The `plan`
assertions are what stop the test passing vacuously: they pin that the path
under test was actually taken.
"""

import pytest
import torch

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize


def _tilize_and_check(device, shape, *, pad=False, dtype=ttnn.bfloat16, torch_dtype=torch.bfloat16):
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(3)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pad_value = 0.0 if pad else None
    tt_output = tilize(tt_input, pad_value=pad_value)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid, pad_value=pad_value)
    got = ttnn.to_torch(tt_output)
    if pad:
        # The pad region is checked by test_tilize_padded.py; here the point is
        # that the two core ranges cover the LOGICAL extent exactly once.
        assert torch.equal(got[tuple(slice(0, d) for d in shape)], torch_input)
        assert torch.equal(
            got[tuple(slice(d, None) for d in shape[:-2]) + (slice(shape[-2], None), slice(None))],
            torch.zeros_like(got[tuple(slice(d, None) for d in shape[:-2]) + (slice(shape[-2], None), slice(None))]),
        )
    else:
        assert torch.equal(got, torch_input)
    return plan


# --- the ragged column tail -------------------------------------------------
# C is chosen ROUGH on purpose. The old divisor rule would have collapsed each
# of these onto a much narrower block (C's coarsest small divisor); the tail
# group is what lets the extent stay at `ceil(C / target)`.
TAIL_CASES = [
    # C = 67 (prime) — the tail is the only thing that can carry the remainder.
    ((1, 1, 32, 32 * 67), False, "c67_r1"),
    # C = 67 with EIGHT row groups, so the tail family is 8 blocks, not 1 —
    # the two-family core split has to balance a whole column of them.
    ((1, 1, 256, 32 * 67), False, "c67_r8"),
    # C = 1572 = 2^2*3*131, the production `LOOSE_CASES` form, padded on H.
    ((1, 1, 1, 50304), True, "c1572_padded"),
    # A rough C whose tail is WIDER than one tile, with an H tail as well.
    ((1, 1, 100, 32 * 45 + 17), True, "c46_padded"),
]


@pytest.mark.parametrize("shape,pad,label", TAIL_CASES, ids=[c[2] for c in TAIL_CASES])
def test_ragged_column_tail(device, shape, pad, label):
    plan = _tilize_and_check(device, shape, pad=pad)
    assert plan.tail_group is not None, f"{label}: expected a ragged column tail"
    tail = plan.tail_group
    # The two families tile the column axis exactly once, with no overlap.
    assert tail.col_tile_offset == plan.num_w_chunks * plan.block_width_tiles
    assert tail.col_tile_offset + tail.block_width_tiles == plan.tensor_col_tiles
    assert 0 < tail.block_width_tiles < plan.block_width_tiles
    # Disjoint core ranges, and every block of both families is assigned.
    full_cores = {(int(c.x), int(c.y)) for c, *_ in plan.assignment}
    tail_cores = {(int(c.x), int(c.y)) for c, *_ in tail.assignment}
    assert full_cores and tail_cores and not (full_cores & tail_cores)
    assert sum(a[2] for a in plan.assignment) == plan.num_blocks_total
    assert sum(a[2] for a in tail.assignment) == tail.num_blocks


def test_ragged_tail_knob_off_is_the_divisor_rule(device, monkeypatch):
    """`RAGGED_COLUMN_TAIL = False` is the Refinement 5 plan, byte for byte."""
    monkeypatch.setattr(pd, "RAGGED_COLUMN_TAIL", False)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})
    plan = _tilize_and_check(device, (1, 1, 32, 32 * 67))
    assert plan.tail_group is None
    assert plan.tensor_col_tiles % plan.block_width_tiles == 0


# --- the split reader -------------------------------------------------------
SPLIT_CASES = [
    # R / num_row_groups == 8 tile-rows per block: writer takes 3 of 8.
    ((1, 1, 16384, 32), ttnn.bfloat16, torch.bfloat16, "bf16_extent8"),
    # R = 500 over 64 groups: RAGGED row extents (7 and 8), so the split point
    # differs per block (2 vs 3) while the split CB is sized for the larger.
    ((1, 1, 16000, 32), ttnn.bfloat16, torch.bfloat16, "bf16_ragged_extent"),
    # float32 — 128 B sticks, still inside the issue-bound threshold.
    ((1, 1, 8192, 32), ttnn.float32, torch.float32, "fp32_extent4"),
]


@pytest.mark.parametrize("shape,dtype,torch_dtype,label", SPLIT_CASES, ids=[c[3] for c in SPLIT_CASES])
def test_split_reader(device, shape, dtype, torch_dtype, label):
    plan = _tilize_and_check(device, shape, dtype=dtype, torch_dtype=torch_dtype)
    assert plan.split_reader > 0, f"{label}: expected the split reader to be on"
    # The split CB holds the writer's WHOLE half-block — the sizing that makes
    # the two-producer pipeline deadlock-free (see tilize_writer.cpp).
    max_extent = -(-plan.tensor_row_blocks // plan.num_row_groups)
    assert plan.split_reader == min(max_extent - 1, max_extent * pd.SPLIT_READER_WRITER_SHARE_PCT // 100)


def test_split_reader_knob_off_is_identical(device, monkeypatch):
    """`SPLIT_READER_MAX_ROW_BYTES = 0` turns the split off everywhere."""
    monkeypatch.setattr(pd, "SPLIT_READER_MAX_ROW_BYTES", 0)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})
    plan = _tilize_and_check(device, (1, 1, 16384, 32))
    assert plan.split_reader == 0


def test_split_reader_off_on_native_sharded_output(device):
    """The gate's HANG leg. `cb_input_rows_split`'s only producer is the WRITER
    kernel, and a natively sharded output emits no writer kernel at all — the
    packer has already written the shard. With the split left on there, compute
    would wait forever on a CB nothing pushes to. Found by the static analyzer;
    this test is the regression pin."""
    grid = device.compute_with_storage_grid_size()
    shape = (1, 1, 16384, 32)
    torch.manual_seed(3)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_mem = ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_output = tilize(tt_input, memory_config=out_mem)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    # The shape is exactly the split's target geometry — 64 B sticks, 8-tile-row
    # blocks — so only the `output_native` leg of the gate can turn it off.
    assert plan.output_native and plan.block_row_bytes <= pd.SPLIT_READER_MAX_ROW_BYTES
    assert plan.split_reader == 0
    assert torch.equal(ttnn.to_torch(tt_output), torch_input)


def test_split_reader_off_where_it_cannot_apply(device):
    """The gate's structural legs: a one-tile-row block cannot be split, and a
    padded block is a different reader block operation the writer does not run."""
    grid = device.compute_with_storage_grid_size()
    # R / num_row_groups == 1 — nothing to split.
    plan = _tilize_and_check(device, (1, 1, 2048, 64))
    assert plan.tensor_row_blocks // plan.num_row_groups == 1 and plan.split_reader == 0
    # Padded: the reader's fill branch has no writer-side twin.
    plan = _tilize_and_check(device, (1, 1, 16000, 24), pad=True)
    assert plan.pad_active and plan.split_reader == 0
    assert grid.x * grid.y >= 1
