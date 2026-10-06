# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Time per model time step (100 x 100 cells, 100 levels), everything per step on the chip; see tests/common.py.
Measured on a P150 (2026-10-07): baseline ~1,186 ms, optimized ~0.52 ms. The limits below are loose."""
import pytest
from loguru import logger

from models.experimental.tensorocean.tests.common import PCC_MIN, VERSIONS, check, time_per_step


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("version, limit_ms, reps", [("optimized", 1.0, 20), ("baseline", 3000.0, 2)])
def test_tensorocean_perf(device, version, limit_ms, reps):
    n, levels = 100, 100
    m, s = check(VERSIONS[version], device, n, levels)
    assert all(r["pcc"] >= PCC_MIN for r in m)
    t = time_per_step(VERSIONS[version], device, s, reps=reps)
    logger.info(f"{version} N={n} L={levels}: {t * 1e3:.3f} ms per step")
    assert t * 1e3 < limit_ms
