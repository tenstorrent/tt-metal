# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fifty distinct RMSNorm cases for the graded-run L1 timing investigation.

This is an opt-in measurement workload, not part of the ordinary test suite. Run it with
``tests.plugins.l1_timing_benchmark`` as documented in ``GRADED_RUN_L1_OVERHEAD.md``.
"""

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import run_rms_norm


pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("width", range(32, 1601, 32))
def test_distinct_rms_norm_case(width, device):
    run_rms_norm(
        ((1, 1, 32, width),),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        gamma_mode="none",
        gamma_dtype=ttnn.float32,
        gamma_layout=ttnn.TILE_LAYOUT,
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        fp32_dest_acc_en=True,
        device=device,
    )
