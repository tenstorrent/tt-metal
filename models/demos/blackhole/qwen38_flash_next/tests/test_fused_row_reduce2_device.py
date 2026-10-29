# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of row_reduce v2 (``QWEN38_ROW_REDUCE=2``, barrier-free) against the stock reduce-scatter and v1.

Per case (bf16 rows 1 and 2, DRAM and L1 width-sharded; fp32 row 1): v2 == stock on rows < rows bit for bit, 8 repeated
calls, a trace of 5 calls kept alive and replayed 7 times (parity carried across replays), traced time of stock / v1 / v2.
Plus one trace that interleaves all the cases (the one landing buffer, counter and semaphore pair serve every shape).
Run (env of run_timing.sh):  flock -w 10800 /home/ttuser/atupe/device.lock timeout 1800 python -m pytest <this file> -s ...
"""

import json
import os

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import test_fused_row_reduce_device as base
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_row_reduce_device import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_reduce as rr

CASES = [
    ("bf16", 1, False),
    ("bf16", 2, False),
    ("fp32", 1, False),
    ("bf16", 1, True),
    ("bf16", 2, True),
    ("bf16", 1, "l1i"),
]  # the last: L1 interleaved partial (MoE with its local sum in L1)
RESULTS = os.environ.get("ROW_REDUCE_RESULTS")


def _call(version, partial):
    os.environ[rr.ENV] = str(version)
    return rr.row_reduce(partial, cluster_axis=1)


def _host(mesh, t, rows):
    return base._from_mesh(mesh, t, rows)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [base.DEVICE_PARAMS], indirect=True)
def test_row_reduce_v2(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    os.environ[rr.ENV] = "2"
    rr.prepare(mesh)
    inputs, report_all = [], []
    for dtype, rows, sharded in CASES:
        x = base._partials(dtype, rows, seed=900 + rows)
        partial = base._to_mesh(mesh, x, dtype, sharded is True)
        if sharded == "l1i":
            partial = ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)
        assert rr.admits(partial)
        os.environ[rr.ENV] = "1"
        stock = _host(mesh, rr.composed(partial), rows)
        report = {"dtype": dtype, "rows": rows, "l1_sharded": sharded, "fabric_payload": rr._payload()}
        v1 = _host(mesh, _call(1, partial), rows)
        v2 = _host(mesh, _call(2, partial), rows)
        report["v1_vs_stock"] = base._mismatches(v1, stock, dtype)
        report["v2_vs_stock"] = base._mismatches(v2, stock, dtype)
        report["v2_repeat_mismatch"] = sum(
            base._mismatches(_host(mesh, _call(2, partial), rows), stock, dtype) for _ in range(8)
        )
        _call(2, partial)  # eager warm-up before the capture
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = [_call(2, partial) for _ in range(5)]
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        bad = 0
        for _ in range(7):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            bad += sum(base._mismatches(_host(mesh, o, rows), stock, dtype) for o in outs)
        ttnn.release_trace(mesh, tid)
        report["v2_trace_replay_mismatch"] = bad
        report["us_slope_and_mean12"] = {
            "stock": base._traced_us(mesh, lambda: rr.composed(partial)),
            "v1": base._traced_us(mesh, lambda: _call(1, partial)),
            "v2": base._traced_us(mesh, lambda: _call(2, partial)),
        }
        print("ROW_REDUCE2_RESULT " + json.dumps(report))
        report_all.append(report)
        inputs.append((dtype, rows, partial, stock))
        assert report["v2_vs_stock"] == 0 and report["v2_repeat_mismatch"] == 0 and bad == 0, report
    # all shapes interleaved in one trace of 7 calls (odd), replayed 5 times
    order = [0, 2, 4, 1, 3, 5, 0, 2]
    for dtype, rows, partial, stock in inputs:
        _call(2, partial)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [(i, _call(2, inputs[i][2])) for i in order]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    bad = 0
    for _ in range(5):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        for i, o in outs:
            dtype, rows, _, stock = inputs[i]
            bad += base._mismatches(_host(mesh, o, rows), stock, dtype)
    ttnn.release_trace(mesh, tid)
    print("ROW_REDUCE2_INTERLEAVED_MISMATCH", bad)
    if RESULTS:
        with open(RESULTS, "a") as f:
            for r in report_all:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"interleaved_trace_mismatch": bad}) + "\n")
    assert bad == 0
