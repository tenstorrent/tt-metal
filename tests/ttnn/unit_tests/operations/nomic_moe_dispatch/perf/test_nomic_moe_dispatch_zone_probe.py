# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone-profile probe (Refinement 3 measurement harness): one perf-setup call at R = NMD_PROBE_ROWS (default 8416).

Run under the device profiler (scripts/run_safe_pytest.sh --profile ...) with temporary DeviceZoneScopedN zones in the
kernels; the zones land in generated/profiler/.logs/profile_log_device.csv. Without the profiler it is a plain
bit-exactness check of the perf setup.
"""

from __future__ import annotations

import ast
import os
import sys

import ttnn

from eval.golden_tests.nomic_moe_dispatch.helpers import (
    check_gather,
    gather_rows,
    index_tensor,
    make_table,
    reserve_l1,
    table_tensor,
)
from ttnn.operations.nomic_moe_dispatch import nomic_moe_dispatch

TOKENS = {2272: 1024, 3296: 1536, 4320: 2048, 6368: 3072, 7392: 3584, 8224: 4000, 8416: 4096}


def test_zone_probe(device):
    """NMD_PROBE_KNOBS="NAME=value,..." overrides plan knobs of the op module for this call (e.g. IDS_MCAST_CHUNKS=0)."""
    rows = int(os.environ.get("NMD_PROBE_ROWS", "8416"))
    op_module = sys.modules[nomic_moe_dispatch.__module__]
    for item in filter(None, os.environ.get("NMD_PROBE_KNOBS", "").split(",")):
        name, value = item.split("=")
        setattr(op_module, name, ast.literal_eval(value))
    op_module.plan.cache_clear()
    tokens = TOKENS[rows]
    filler = reserve_l1(device, 96)
    table = make_table(tokens)
    gather = gather_rows(tokens, rows, "real_text")
    tt_table = table_tensor(table, device, "l1")
    tt_index = index_tensor(gather, device)
    for _ in range(int(os.environ.get("NMD_PROBE_CALLS", "2"))):
        output = nomic_moe_dispatch(tt_table, tt_index)
        if not os.environ.get("NMD_SKIP_CHECK"):  # ablation runs (stubbed kernels) skip the bit-exact check
            check_gather(output, table, gather)
        ttnn.deallocate(output)
    for tensor in (tt_table, tt_index):
        ttnn.deallocate(tensor)
    if filler is not None:
        ttnn.deallocate(filler)
