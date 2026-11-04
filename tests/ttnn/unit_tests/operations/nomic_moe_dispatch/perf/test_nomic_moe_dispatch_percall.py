# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-call device-time harness (Refinement 4): the k = 2 / k = 3 congestion collapse is bimodal per call, so 5-call
means hide it. Run under the profiler; every call is one row of the ops perf CSV, read back by
perf/percall_report.py (cases in the order below, 1 warm-up + NMD_PERCALL_CALLS calls each).

    NMD_PERCALL_CASES="8416:;8416:READS_IN_FLIGHT=6;4320:" scripts/run_safe_pytest.sh --profile <this file>
    python3 tests/ttnn/unit_tests/operations/nomic_moe_dispatch/perf/percall_report.py <ops csv>

Each case is "R:NAME=value&NAME=value" (plan knobs of the op module; empty = defaults).
"""

from __future__ import annotations

import ast
import os
import sys

import pytest
import ttnn

import ttnn.operations.nomic_moe_dispatch  # noqa: F401
from eval.golden_tests.nomic_moe_dispatch.helpers import (
    check_gather,
    gather_rows,
    index_tensor,
    make_table,
    reserve_l1,
    table_tensor,
)

op_module = sys.modules["ttnn.operations.nomic_moe_dispatch.nomic_moe_dispatch"]
TOKENS = {2240: 1000, 2272: 1024, 2528: 1152, 2656: 1216, 3296: 1536, 4320: 2048, 5344: 2560, 6368: 3072}
TOKENS.update({7392: 3584, 8064: 3904, 8224: 4000, 8416: 4096})
CASES = [c for c in os.environ.get("NMD_PERCALL_CASES", "8416:").split(";") if c]
CALLS = int(os.environ.get("NMD_PERCALL_CALLS", "10"))


def _parse(case):
    rows, _, knobs = case.partition(":")
    return int(rows), {k: ast.literal_eval(v) for k, v in (i.split("=") for i in knobs.split("&") if i)}


@pytest.mark.parametrize("case", CASES, ids=[c.replace(":", "-").replace("=", "_").replace("&", "-") for c in CASES])
def test_percall(case, device):
    rows, knobs = _parse(case)
    saved = {name: getattr(op_module, name) for name in knobs}
    try:
        for name, value in knobs.items():
            setattr(op_module, name, value)
        op_module.plan.cache_clear()
        op_module._l1_output_memory_config.cache_clear()
        tokens = TOKENS[rows]
        filler = reserve_l1(device, 96)
        table = make_table(tokens)
        gather = gather_rows(tokens, rows, "real_text")
        tt_table = table_tensor(table, device, "l1")
        tt_index = index_tensor(gather, device)
        for i in range(CALLS + 1):
            output = op_module.nomic_moe_dispatch(tt_table, tt_index)
            if i == 0:
                check_gather(output, table, gather)
            ttnn.deallocate(output)
        for tensor in (tt_table, tt_index):
            ttnn.deallocate(tensor)
        if filler is not None:
            ttnn.deallocate(filler)
    finally:
        for name, value in saved.items():
            setattr(op_module, name, value)
        op_module.plan.cache_clear()
        op_module._l1_output_memory_config.cache_clear()
