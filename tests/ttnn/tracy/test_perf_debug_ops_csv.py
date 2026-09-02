# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
perf-debug op-perf CSV consumer test.

Runs a TTNN matmul loop with the streaming profiler (TT_METAL_STREAMING_PROFILER=1, which implies
TT_METAL_DEVICE_PROFILER) and TT_METAL_PERF_DEBUG_OPS_CSV set, then checks the CSV the ops-csv
consumer wrote at process exit: one row per program launch keyed by runtime host-id, with the classic
device-profiler report's device columns (see perf_debug_ops_csv.hpp). The Tracy sink is opt-in
(TT_METAL_STREAMING_PROFILER_TRACY) and not opted into, so this also exercises the register_consumer
path as the sole record sink.

Hardware-gated like test_perf_debug_profiler.py: needs a Blackhole box with DRAM programmable cores;
skips cleanly when the profiler reports it cannot run. Device work runs in a subprocess so the pytest
parent never takes the PCIe lock.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.tracy.common import PROFILER_ARTIFACTS_DIR, TT_METAL_HOME

ARTIFACTS = PROFILER_ARTIFACTS_DIR / "perf_debug_ops_csv_tests"

N_MATMULS = 20

WORKLOAD = f"""
import torch
import ttnn

device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.randn(512, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.randn(512, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
for _ in range({N_MATMULS}):
    a = ttnn.matmul(a, b)
ttnn.synchronize_device(device)
ttnn.close_device(device)
"""


def test_perf_debug_ops_csv():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACTS / "ops_perf.csv"
    csv_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "TT_METAL_HOME": str(TT_METAL_HOME),
            # One switch: implies TT_METAL_DEVICE_PROFILER; the Tracy sink is opt-in and deliberately
            # NOT opted into here, so the ops-csv consumer is the sole record sink.
            "TT_METAL_STREAMING_PROFILER": "1",
            "TT_METAL_PERF_DEBUG_ROLE_SPLIT": "1",
            "TT_METAL_PERF_DEBUG_OPS_CSV": str(csv_path),
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", WORKLOAD],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=str(TT_METAL_HOME),
    )
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"workload failed (rc={proc.returncode}):\n{log[-2000:]}"
    if "[perf-debug profiler] active" not in log:
        pytest.skip("perf-debug profiler did not activate on this part")

    assert csv_path.exists(), "ops CSV was not written"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= N_MATMULS, f"expected >= {N_MATMULS} op rows, got {len(rows)}"

    for row in rows:
        prog = int(row["GLOBAL CALL COUNT"])
        assert int(row["EXECUTION"]) == 0, f"untraced ops must have one execution: {row}"
        cores = int(row["CORE COUNT"])
        start_cyc = int(row["DEVICE KERNEL START CYCLE"])
        end_cyc = int(row["DEVICE KERNEL END CYCLE"])
        kernel = float(row["DEVICE KERNEL DURATION [ns]"])
        core_min = float(row["DEVICE KERNEL DURATION PER CORE MIN [ns]"])
        core_max = float(row["DEVICE KERNEL DURATION PER CORE MAX [ns]"])
        core_avg = float(row["DEVICE KERNEL DURATION PER CORE AVG [ns]"])
        first_to_last = float(row["DEVICE KERNEL FIRST TO LAST START [ns]"])
        ctx = f"row {row}"
        assert prog > 0, ctx
        assert cores >= 1, ctx
        assert 0 < start_cyc < end_cyc, ctx
        assert kernel > 0, ctx
        assert 0 < core_min <= core_avg <= core_max <= kernel, ctx
        assert 0 <= first_to_last <= kernel, ctx
