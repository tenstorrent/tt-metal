# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Knob matrix for tilize: every parked lever stays correct at non-default values.

The default path is covered by test_tilize.py; this turns the host knobs in
tilize_program_descriptor.py (READ_AHEAD, SPLIT_READER_MAX_SEGMENT_BYTES,
CB_BUDGET_BYTES -> block_width) so the ragged column block, multi-column-block
walks, the transaction-id read-ahead and the split reader (odd/even walk
positions over two input CBs) are all exercised, as are the Refinement 3 levers
(in-flight windows / write-ahead, eager publish, NoC split, bank-stride
addressing). Output must be bit-identical.
"""
import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

SHAPES = [
    (1, 1, 32, 32),  # 1 position: split reader must fall back
    (1, 1, 64, 160),  # R=2, C=5: ragged column block under the tiny budget
    (1, 1, 96, 160),  # R=3, C=5: odd tile-row count per core
    (2, 3, 64, 96),  # leading-dim fold
    (1, 1, 1600, 96),  # R=50 uneven row split
    (1, 1, 16384, 64),  # 8 tile-rows per core
    (1, 1, 4320, 160),  # R=135, C=5: 3 or 2 tile-rows per core (partial final quantum), ragged column block
]

CONFIGS = {
    "read_ahead2": dict(READ_AHEAD=2),
    "split": dict(SPLIT_READER_MAX_SEGMENT_BYTES=1 << 20),
    "split_read_ahead2": dict(SPLIT_READER_MAX_SEGMENT_BYTES=1 << 20, READ_AHEAD=2),
    "tiny_budget": dict(CB_BUDGET_BYTES={False: 3 * 8192, True: 3 * 8192}),
    "tiny_budget_split_ra2": dict(
        CB_BUDGET_BYTES={False: 3 * 12288, True: 3 * 12288},
        SPLIT_READER_MAX_SEGMENT_BYTES=1 << 20,
        READ_AHEAD=2,
    ),
    "depth3_ra3": dict(DEPTH_IN=3, READ_AHEAD=3),
    # tile_row streaming window: several tile-rows per CB quantum, incl. a partial
    # final quantum (odd walk lengths) and a quantum spanning column blocks.
    # QUANTUM_MIN_TILES=1 -> one tile-row per quantum everywhere (the pre-knob schedule)
    "quantum1": dict(QUANTUM_MIN_TILES=1),
    "quantum_large": dict(QUANTUM_MIN_TILES=64),
    "quantum_large_ra2": dict(QUANTUM_MIN_TILES=48, READ_AHEAD=2),
    "quantum_tiny_budget": dict(QUANTUM_MIN_TILES=64, CB_BUDGET_BYTES={False: 3 * 8192, True: 3 * 8192}),
    # block_width capped at 2 with budget to spare: C=5 -> 3 column blocks, and the
    # multi-tile-row quanta straddle column-block boundaries in the walk.
    "quantum_narrow_blocks": dict(QUANTUM_MIN_TILES=8, FAST_TILIZE_MAX_BLOCK_WIDTH=2),
    "quantum_depth3_ra3": dict(QUANTUM_MIN_TILES=64, DEPTH_IN=3, READ_AHEAD=3),
    # split reader must pin the quantum to one tile-row
    "quantum_split": dict(QUANTUM_MIN_TILES=64, SPLIT_READER_MAX_SEGMENT_BYTES=1 << 20),
    # Refinement 3 levers (parked at defaults; every one must stay bit-exact when turned).
    # In-flight windows: 1-row quanta, read_ahead / write_ahead windows deepen the CBs.
    "windows8": dict(QUANTUM_MIN_TILES=2, READ_WINDOW_MIN_TILES=8, WRITE_WINDOW_MIN_TILES=8),
    "write_window_only": dict(WRITE_WINDOW_MIN_TILES=32),
    # the budget loop must shrink the windows back inside the CB budget
    "windows_tiny_budget": dict(
        QUANTUM_MIN_TILES=1,
        READ_WINDOW_MIN_TILES=64,
        WRITE_WINDOW_MIN_TILES=64,
        CB_BUDGET_BYTES={False: 3 * 8192, True: 3 * 8192},
    ),
    # eager publish with a read-ahead as deep as the CB (the lazy-completion worst case)
    "eager_full_read_ahead": dict(EAGER_PUBLISH=True, QUANTUM_MIN_TILES=2, DEPTH_IN=8, READ_AHEAD=8),
    "eager_windows": dict(EAGER_PUBLISH=True, QUANTUM_MIN_TILES=1, READ_WINDOW_MIN_TILES=8),
    # stream split across both NoCs (DM_DYNAMIC_NOC kernels); a write split pins write_ahead to 1
    "noc_split_read2": dict(READ_NOC_SPLIT=2),
    "noc_split_write2": dict(WRITE_NOC_SPLIT=2),
    "noc_split_both3_windows": dict(
        READ_NOC_SPLIT=3, WRITE_NOC_SPLIT=3, QUANTUM_MIN_TILES=2, READ_WINDOW_MIN_TILES=8, WRITE_WINDOW_MIN_TILES=8
    ),
    # bank-stride addressing / bank-major order, incl. multi-column-block walks (segment offsets)
    "bank_stride": dict(BANK_STRIDE=1),
    "bank_major": dict(BANK_STRIDE=2),
    "bank_major_tiny_budget_ra2": dict(
        BANK_STRIDE=2, READ_AHEAD=2, EAGER_PUBLISH=True, CB_BUDGET_BYTES={False: 3 * 8192, True: 3 * 8192}
    ),
    # Refinement 6 bank-coalesced reads (the default on narrow DRAM sticks): off keeps StickProducer
    # covered on those shapes; the rest turn its depth / quantum / stick gate / scatter mode, incl.
    # rotated walks that wrap inside a unit and partial final units.
    "coalesce_off": dict(BANK_COALESCE_STAGE_DEPTH=0),
    "coalesce_depth1": dict(BANK_COALESCE_STAGE_DEPTH=1),
    "coalesce_depth3_rows4": dict(BANK_COALESCE_STAGE_DEPTH=3, BANK_COALESCE_QUANTUM_ROWS=4),
    "coalesce_rows1_wide_gate": dict(BANK_COALESCE_QUANTUM_ROWS=1, BANK_COALESCE_MAX_STICK_BYTES=1 << 20),
    "coalesce_rows3_scatter_write": dict(BANK_COALESCE_QUANTUM_ROWS=3, BANK_COALESCE_SCATTER_WRITE=True),
    "coalesce_tiny_budget": dict(
        BANK_COALESCE_MAX_STICK_BYTES=1 << 20, CB_BUDGET_BYTES={False: 3 * 8192, True: 3 * 8192}
    ),
}


@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_tilize_knob(device, monkeypatch, shape, config):
    for name, value in CONFIGS[config].items():
        monkeypatch.setattr(pd, name, value)
    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(t)
    assert out.layout == ttnn.TILE_LAYOUT
    y = ttnn.to_torch(out)
    mismatches = int((y != x).sum())
    assert mismatches == 0, f"{mismatches} of {x.numel()} elements differ"
