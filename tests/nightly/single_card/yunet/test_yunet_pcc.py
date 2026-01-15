# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YUNet PCC nightly test for CI.

Runs the YUNet face detection model PCC test on Wormhole.
"""

import pytest

from models.experimental.yunet.common import YUNET_L1_SMALL_SIZE


@pytest.mark.parametrize("device_params", [{"l1_small_size": YUNET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("input_size", [(320, 320)])
def test_yunet_pcc(device, input_size, reset_seeds):
    """
    PCC test for YUNet face detection model.

    Tests 320x320 input size (default).
    Expected PCC > 0.99 for all outputs.
    """
    from models.experimental.yunet.tests.pcc.test_pcc import test_yunet_pcc as run_pcc_test

    run_pcc_test(device=device, input_size=input_size)
