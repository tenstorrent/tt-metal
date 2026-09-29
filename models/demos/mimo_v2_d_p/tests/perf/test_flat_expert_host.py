# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host time of one flat_routed_expert call (program-cache hit): the Python call's wall time with the device idle
(synchronized before), median over launches, MiMo 64-expert plan at 2048 tokens per chip (indexed, row-major y).

    scripts/run_safe_pytest.sh models/demos/mimo_v2_d_p/tests/perf/test_flat_expert_host.py -s
"""

import os
import time

import pytest
from loguru import logger

import ttnn
from models.demos.mimo_v2_d_p.tests.unit.test_flat_expert_indexed_shapes import _call, _indexed_inputs, _op
from models.demos.mimo_v2_d_p.tests.unit.test_flat_routed_expert_op import CASES, _counts


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_expert_host(device):
    case = CASES[0]
    op, gids = _op(device, case)
    d, _, _ = _indexed_inputs(device, case, gids, _counts(case[3], case[4], 1), 1)
    _call(op, d, 1, True).deallocate(True)  # compile
    host = []
    for _ in range(int(os.environ.get("MIMO_HOST_ITERS", "20"))):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        y = _call(op, d, 1, True)
        host.append((time.perf_counter() - t0) * 1e6)
        ttnn.synchronize_device(device)
        y.deallocate(True)
    s = sorted(host)
    logger.info(f"HOST flat_routed_expert call: median {s[len(s) // 2]:.0f} us, min {s[0]:.0f} us")
