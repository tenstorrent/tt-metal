# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""otherlayers: all unit tests of the decode-indexer / owner-layer ops in ONE pytest file (one device hold): bit equality against the stock chains + traced time; the device profiler is read after every test
(zones: DSV41_OL_ZONES=1, DSV41_OL_SCORE_DIAG=1)."""

import pytest

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_ol_idx_ops import *  # noqa: F401,F403
from models.demos.blackhole.deepseek_v41_flash.tests.test_ol_score import *  # noqa: F401,F403
from models.demos.blackhole.deepseek_v41_flash.tests.test_pf_fp4_fused import test_fp4_nb1_matches  # noqa: F401


@pytest.fixture(autouse=True)
def _read_profiler(device):
    yield
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:  # noqa
        pass
