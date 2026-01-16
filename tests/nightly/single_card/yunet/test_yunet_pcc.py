# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YUNet PCC nightly test for CI.

Runs the YUNet face detection model PCC test.
Automatically clones the YuNet GitHub repo if not present.
"""

import os
import subprocess

import pytest

from models.experimental.yunet.common import YUNET_L1_SMALL_SIZE


def _ensure_yunet_repo():
    """Clone YuNet repository if it doesn't exist."""
    yunet_dir = os.path.join(
        os.environ.get("TT_METAL_HOME", "."),
        "models/experimental/yunet/YUNet",
    )

    # Check if already cloned
    if os.path.exists(os.path.join(yunet_dir, "nets", "nn.py")):
        return

    print(f"YuNet repo not found, cloning to {yunet_dir}...")

    # Clone the repo
    subprocess.run(
        ["git", "clone", "https://github.com/jahongir7174/YUNet.git", yunet_dir],
        check=True,
    )

    # Create __init__.py files to make it a proper Python package
    open(os.path.join(yunet_dir, "__init__.py"), "w").close()
    open(os.path.join(yunet_dir, "nets", "__init__.py"), "w").close()

    print("YuNet repo cloned successfully.")


@pytest.mark.parametrize("device_params", [{"l1_small_size": YUNET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("input_size", [(320, 320)])
def test_yunet_pcc(device, input_size, reset_seeds):
    """
    PCC test for YUNet face detection model.

    Tests 320x320 input size (default).
    Expected PCC > 0.99 for all outputs.
    """
    # Clone YuNet repo on-the-fly if not present
    _ensure_yunet_repo()

    from models.experimental.yunet.tests.pcc.test_pcc import test_yunet_pcc as run_pcc_test

    run_pcc_test(device=device, input_size=input_size)
