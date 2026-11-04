# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-kernel perf sweep for nomic_moe_dispatch over every golden shape (verifier measurement harness).

Run through the eval runner so the real-time profiler captures the op windows:

    eval/eval_test_runner.sh tests/ttnn/unit_tests/operations/nomic_moe_dispatch/perf/ <results_dir>

Each test makes one warm-up call (outside the measured window), then CALLS calls inside metrics_plugin.op_window();
test_results.json's device_kernel_ns is the sum, so the per-call device time is device_kernel_ns / CALLS. The host
time per call (Python, no sync) is printed as `HOST_US`. Table and Xg in L1, a real text's routing, 96 KB of every
bank reserved first (the perf cases' setup).
"""

from __future__ import annotations

import time

import pytest
import ttnn

from eval import metrics_plugin
from eval.golden_tests.nomic_moe_dispatch.feature_spec import INPUTS
from eval.golden_tests.nomic_moe_dispatch.helpers import (
    check_gather,
    gather_rows,
    index_tensor,
    make_table,
    reserve_l1,
    table_tensor,
)
from ttnn.operations.nomic_moe_dispatch import nomic_moe_dispatch, plan

CALLS = 5


@pytest.mark.parametrize("inputs", INPUTS, ids=[f"T{i[0][-2]}_R{i[1][-1]}" for i in INPUTS])
def test_perf_sweep(inputs, device):
    tokens, rows = inputs[0][-2], inputs[1][-1]
    filler = reserve_l1(device, 96)
    table = make_table(tokens)
    gather = gather_rows(tokens, rows, "real_text")
    tt_table = table_tensor(table, device, "l1")
    tt_index = index_tensor(gather, device)

    output = nomic_moe_dispatch(tt_table, tt_index)  # warm-up (compile + first run), not measured
    check_gather(output, table, gather)
    ttnn.deallocate(output)

    host_start = time.perf_counter()
    for _ in range(CALLS):
        with metrics_plugin.op_window():
            output = nomic_moe_dispatch(tt_table, tt_index)
        ttnn.deallocate(output)
    host_us = (time.perf_counter() - host_start) / CALLS * 1e6
    ttnn.synchronize_device(device)
    p = plan(rows)
    print(
        f"HOST_US R={rows} k={p.pieces_per_row} w={p.block_width_tiles} blocks={p.max_blocks_per_core} "
        f"host_us_per_call={host_us:.1f}"
    )
    for tensor in (tt_table, tt_index):
        ttnn.deallocate(tensor)
    if filler is not None:
        ttnn.deallocate(filler)
