# Scratch, not for merge. Untilize A/B job: op-level shapes (test_perf) and bits coverage (test_cov).
import os
import sys

import pytest
import torch
import ttnn

sys.path.insert(0, os.environ.get("AB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../ci_ab")))
import ut_cases as U

PERF = [(dt, ct) for dt in ("bfp8", "bf16") for ct in (3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 33, 64, 128)] + [("fp32", ct) for ct in (9, 16, 33)]


@pytest.mark.parametrize("dt,ct", PERF, ids=[f"{d}-ct{c}" for d, c in PERF])
def test_perf(device, dt, ct):
    x = U.wide((1, 1, 20 * 32, ct * 32), 5 + ct, U.DT[dt][1])
    t = U.to_tt(device, x, dt)
    u = ttnn._ttnn.operations.data_movement.untilize_force_native(t)  # the native prim runs the fast untilize helper
    o = ttnn.to_torch(u)
    U.emit(f"perf-{dt}-ct{ct}", 0, "RUN", o)
    ttnn.deallocate(t)
    ttnn.deallocate(u)


@pytest.mark.parametrize("name", list(U.CASES))
def test_cov(device, name):
    U.CASES[name](device)
