# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Knob sweep for nomic_moe_dispatch (measurement harness: perf lamps L2 and L4, and Refinement 3's ids row
multicast and issue stagger knobs).

Overrides the host plan's knob constants (single source: nomic_moe_dispatch.py) for one test, then restores them.
Run through the eval runner; per-call device time = device_kernel_ns / CALLS.

    eval/eval_test_runner.sh tests/ttnn/unit_tests/operations/nomic_moe_dispatch/perf/ <results_dir> -k knob
"""

from __future__ import annotations

import os
import sys

import pytest
import ttnn

import ttnn.operations.nomic_moe_dispatch  # noqa: F401  (the package re-exports the function under the module name)
from eval import metrics_plugin
from eval.golden_tests.nomic_moe_dispatch.helpers import (
    check_gather,
    gather_rows,
    index_tensor,
    make_table,
    reserve_l1,
    table_tensor,
)

op_module = sys.modules["ttnn.operations.nomic_moe_dispatch.nomic_moe_dispatch"]
CALLS = 5
TOKENS = {2272: 1024, 2656: 1216, 3296: 1536, 4320: 2048, 6368: 3072, 7392: 3584, 8064: 3904, 8416: 4096}
K_FIRST_2 = (2, 3, 4, 6, 1, 8, 12, 24)  # lamp L4: prefer half rows over whole rows when both fit the budget
NO_SHED = {k: 0 for k in K_FIRST_2}  # Refinement 4's sender-column shed off at every k

CASES = (
    [(r, dict(READS_IN_FLIGHT=f)) for r in (2272, 4320, 8416) for f in (4, 8, 12)]
    + [(r, dict(READ_BYTES_IN_FLIGHT=b)) for r in (8416, 6368) for b in (4 * 768, 8 * 768)]
    + [
        # Refinement 3's issue orders on the positional split (the sender shed needs halves coprime with every step).
        (r, dict(ROW_STAGGER_BY_K={}, ROW_STAGGER_DEFAULT=(st, sd), SENDER_ROW_SHED_BY_K=NO_SHED))
        for r in (2272, 4320, 6368, 8416)
        for st, sd in ((0, 1), (3, 4), (5, 2))
    ]
    + [(r, dict(IDS_LOOKAHEAD=a)) for r in (2272, 4320, 8416) for a in (2, 4)]
    + [(r, dict(IDS_SENDER_NOC=1)) for r in (4320, 8416)]
    + [(r, dict(START_SPREAD_CYCLES=700)) for r in (4320, 8416)]
    + [(r, dict(PIECES_PER_ROW_CANDIDATES=K_FIRST_2)) for r in (3296, 6368)]
    + [(r, {}) for r in (2656, 3296, 6368, 7392)]
    # Refinement 4: the NOC0 / NOC1 row split (lamp L1: positional share, path-aware split, sender-column shed) and the
    # issue order (L6). The k = 2 / 3 congestion collapse is per call: use test_nomic_moe_dispatch_percall.py to judge.
    + [(r, dict(ROWS_NOC0=n, SENDER_ROW_SHED_DEFAULT=0)) for r in (2272, 4320, 7392, 8416) for n in (13, 19)]
    + [
        (r, dict(ROW_SPLIT_PATH=True, ROW_SPLIT_SLACK=s, SENDER_ROW_SHED_DEFAULT=0))
        for r in (2272, 4320, 7392, 8416)
        for s in (0, 2, 4, 16)
    ]
    + [(r, dict(SENDER_ROW_SHED_DEFAULT=n)) for r in (4320, 7392, 8416) for n in (0, 1, 3)]
)


def _id(case):
    rows, knobs = case

    def value(name, v):
        if name == "PIECES_PER_ROW_CANDIDATES":
            return "k2first"
        return "x".join(map(str, v)) if isinstance(v, tuple) else v

    tag = "-".join(f"{k}={value(k, v)}" for k, v in knobs.items() if not k.endswith("_BY_K")) or "baseline"
    return f"R{rows}-{tag}"


@pytest.fixture
def knobs(request):
    saved = {name: getattr(op_module, name) for name in request.param}
    for name, value in request.param.items():
        setattr(op_module, name, value)
    op_module.plan.cache_clear()
    op_module._l1_output_memory_config.cache_clear()
    yield request.param
    for name, value in saved.items():
        setattr(op_module, name, value)
    op_module.plan.cache_clear()
    op_module._l1_output_memory_config.cache_clear()


@pytest.mark.parametrize("rows,knobs", [(r, k) for r, k in CASES], ids=[_id(c) for c in CASES], indirect=["knobs"])
def test_knob_sweep(rows, knobs, device):
    tokens = TOKENS[rows]
    filler = reserve_l1(device, 96)
    table = make_table(tokens)
    gather = gather_rows(tokens, rows, "real_text")
    tt_table = table_tensor(table, device, "l1")
    tt_index = index_tensor(gather, device)
    output = op_module.nomic_moe_dispatch(tt_table, tt_index)
    if not os.environ.get("NMD_SKIP_CHECK"):  # ablation runs (stubbed kernels) skip the bit-exact check
        check_gather(output, table, gather)
    ttnn.deallocate(output)
    for _ in range(CALLS):
        with metrics_plugin.op_window():
            output = op_module.nomic_moe_dispatch(tt_table, tt_index)
        ttnn.deallocate(output)
    p = op_module.plan(rows)
    print(f"PLAN R={rows} k={p.pieces_per_row} w={p.block_width_tiles} blocks={p.max_blocks_per_core} {knobs}")
    for tensor in (tt_table, tt_index):
        ttnn.deallocate(tensor)
    if filler is not None:
        ttnn.deallocate(filler)
