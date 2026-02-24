# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YUNet PCC nightly test for CI.

Runs the YUNet face detection model PCC test.
"""

import pytest

from models.experimental.yunet.common import YUNET_L1_SMALL_SIZE, setup_yunet_reference


@pytest.fixture(scope="module", autouse=True)
def setup_yunet():
    """Setup YUNet reference model before running tests."""
    setup_yunet_reference()


@pytest.mark.parametrize("device_params", [{"l1_small_size": YUNET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("input_size", [(640, 640)])
def test_yunet_pcc(device, input_size, reset_seeds):
    """
    PCC test for YUNet face detection model.

    Tests 640x640 input size (default).
    Expected PCC > 0.99 for all outputs.
    """
    from models.experimental.yunet.tests.pcc.test_pcc import test_yunet_pcc as run_pcc_test

    run_pcc_test(device=device, input_size=input_size)
