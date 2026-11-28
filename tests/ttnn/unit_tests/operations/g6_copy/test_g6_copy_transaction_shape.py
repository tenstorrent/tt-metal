# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 (transaction shape): READ_TRIDS read pipeline over the streamed plan.

test_g6_copy_trid_forced pins correctness of the trid ring (wrap, ragged last block, more blocks than
ids, depth-limited ids) on small forced blocks. test_g6_copy_trid_sweep is the perf sweep over the
guard set: one op call per test id; run under scripts/run_safe_pytest.sh --profile --no-py-zones and
read the g6_copy rows of the CSV in collection order. Every call is graded by helpers.TOLERANCES.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_copy import describe_plan, g6_copy
from ttnn.operations.g6_copy import g6_copy_program_descriptor as plan_knobs
from eval.golden_tests.g6_copy import helpers as H

from .test_g6_copy_overlap import GUARD_ROWS, row_id

_F32, _BF16, _BF8B = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
_TILE, _RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT


def _run(device, monkeypatch, row, knobs):
    for name, value in knobs.items():
        monkeypatch.setattr(plan_knobs, name, value)
    shape, dtype, layout = row
    gen = torch.Generator().manual_seed(11)
    src = H.make_source(dtype, shape, generator=gen)
    dst = H.make_source(dtype, shape, generator=gen)
    plan = describe_plan(H.create_ttnn_input_tensor(src, device, dtype=dtype, layout=layout))
    print(
        f"TRID {row_id(row)} {knobs}: plan={plan['plan']} block_pages={plan['block_pages']} "
        f"depth={plan['cb_depth_blocks']} trids={plan['read_trids']} core_pages_max={max(plan['core_pages'])}"
    )
    H.copy_and_check(g6_copy, src, dst, dtype=dtype, layout=layout, device=device, gate=H.default_gate(dtype))
    return plan


# (MAX_BLOCK_PAGES, STREAM_DEPTH_BLOCKS, READ_TRIDS): 1-3 page blocks so every core runs many blocks.
FORCED = [(1, 2, 2), (3, 2, 2), (2, 2, 4), (3, 1, 3), (1, 2, 15)]
FORCED_ROWS = [
    ((2, 12, 128, 128), _F32, _TILE),  # 384 pages, 4 / core
    ((1, 1, 32, 62080), _BF16, _TILE),  # 1940 pages, ragged per-core split
    ((16, 3, 2560), _BF16, _RM),  # 48 rows
]


@pytest.mark.parametrize("affinity", [False, True], ids=["interleaved_walk", "bank_affine"])
@pytest.mark.parametrize("forced", FORCED, ids=lambda f: f"blk{f[0]}_d{f[1]}_t{f[2]}")
@pytest.mark.parametrize("row", FORCED_ROWS, ids=row_id)
def test_g6_copy_trid_forced(device, monkeypatch, row, forced, affinity):
    block, depth, trids = forced
    plan = _run(
        device,
        monkeypatch,
        row,
        {
            "BLOCK_BYTES": None,
            "MAX_BLOCK_PAGES": block,
            "STREAM_DEPTH_BLOCKS": depth,
            "READ_TRIDS": trids,
            "BANK_AFFINITY": affinity,
        },
    )
    assert plan["read_trids"] <= plan["cb_depth_blocks"] - 1 or plan["read_trids"] == 1
    assert plan["l1_bytes_per_core"] <= plan_knobs.L1_BUDGET_BYTES


VARIANTS = {
    "r2_default": {"READ_TRIDS": 1},
    "t2": {"READ_TRIDS": 2},
    "t3": {"READ_TRIDS": 3},
    "t4": {"READ_TRIDS": 4},
    "aff": {"BANK_AFFINITY": True},
    "aff_t2": {"BANK_AFFINITY": True, "READ_TRIDS": 2},
    "aff_b32k": {"BANK_AFFINITY": True, "BLOCK_BYTES": 32768},
    "aff_one_block": {"BANK_AFFINITY": True, "BLOCK_BYTES": None},
}


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
@pytest.mark.parametrize("row", GUARD_ROWS, ids=row_id)
def test_g6_copy_trid_sweep(device, monkeypatch, row, variant):
    _run(device, monkeypatch, row, VARIANTS[variant])


# Block x trid co-tune on the flagged shapes (trids hide the per-block round trip, so they can let a
# smaller block keep the same bytes in flight).
COTUNE = {
    f"b{b // 1024}k_t{t}_d{d}": {"BLOCK_BYTES": b, "READ_TRIDS": t, "STREAM_DEPTH_BLOCKS": d}
    for b in (4096, 8192, 16384, 32768)
    for t, d in ((1, 2), (2, 2), (4, 2), (2, 3))
}


@pytest.mark.parametrize("variant", list(COTUNE), ids=list(COTUNE))
@pytest.mark.parametrize("row", GUARD_ROWS[:2], ids=row_id)
def test_g6_copy_trid_cotune(device, monkeypatch, row, variant):
    _run(device, monkeypatch, row, COTUNE[variant])


# Bank affinity on shapes where every core holds a single block (no streaming): the plain one_block
# plan with each core's pages in one bank.
SMALL_ROWS = [((1, 1, 32, 32 * 37), _BF16, _TILE), ((3, 5, 64, 96), _F32, _TILE), ((7, 33), _BF16, _RM)]


@pytest.mark.parametrize("affinity", [False, True], ids=["interleaved_walk", "bank_affine"])
@pytest.mark.parametrize("row", SMALL_ROWS, ids=row_id)
def test_g6_copy_bank_affinity_one_block(device, monkeypatch, row, affinity):
    _run(device, monkeypatch, row, {"BANK_AFFINITY": affinity})


REPEATS = 5


@pytest.mark.parametrize("repeat", range(REPEATS))
@pytest.mark.parametrize("variant", ["r2_default", "t2", "t3"])
def test_g6_copy_trid_repeats(device, monkeypatch, variant, repeat):
    """>= 5 fresh samples per variant on the flagged shape."""
    _run(device, monkeypatch, GUARD_ROWS[0], VARIANTS[variant])
