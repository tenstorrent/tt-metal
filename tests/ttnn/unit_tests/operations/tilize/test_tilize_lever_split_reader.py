# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lever measurement: the SPLIT READER share (Refinement 6, item 1).

DO NOT DELETE. `SPLIT_READER_WRITER_SHARE_PCT` is the fraction of each block's
tile-rows whose sticks the WRITER kernel reads instead of the reader. The two
data-movement RISC-Vs are not symmetric — the writer also carries the block's
stores, and its reads share NoC1 with them — so the balance point is a
measurement, not `50`.

`SPLIT_READER_MAX_ROW_BYTES = 0` disables the split entirely, which is the
`share=off` row and the byte-identical baseline.

Run for numbers (device kernel ns per row, in the parametrize order):
    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_split_reader.py

Every variant asserts BIT IDENTITY — the split changes which RISC-V moves which
sticks and nothing else, so anything but an identical result is a bug.
"""

import pytest
import torch

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

TALL_NARROW = (1, 1, 16384, 32)  # R=512 x C=1, 64 B sticks — the issue-bound target

# `None` = the split off (baseline); an int = the writer's share, in percent.
# At the target's 8-tile-row block these map to 0 / 1 / 2 / 3 / 4 / 6 writer
# tile-rows — every distinct split an 8-row block has.
SHARES = [None, 13, 25, 38, 50, 75]


def _run(device, shape, share, monkeypatch):
    monkeypatch.setattr(pd, "SPLIT_READER_MAX_ROW_BYTES", 0 if share is None else 256)
    if share is not None:
        monkeypatch.setattr(pd, "SPLIT_READER_WRITER_SHARE_PCT", share)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})

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
        f"\n[split share={share}] {shape}: bw={plan.block_width_tiles} "
        f"rows/blk={plan.tensor_row_blocks // plan.num_row_groups} split_rows={plan.split_reader} "
        f"cores={plan.num_cores_used}/{grid.x * grid.y} L1/core={plan.l1_per_core_bytes // 1024}KiB"
    )
    assert torch.equal(ttnn.to_torch(tt_output), torch_input), f"share={share}: not bit-identical"
    return plan


@pytest.mark.parametrize("share", SHARES, ids=[("off" if s is None else f"pct{s}") for s in SHARES])
def test_split_reader_share(device, share, monkeypatch):
    plan = _run(device, TALL_NARROW, share, monkeypatch)
    rows_per_block = plan.tensor_row_blocks // plan.num_row_groups
    expected = 0 if share is None else min(rows_per_block - 1, rows_per_block * share // 100)
    assert plan.split_reader == expected
