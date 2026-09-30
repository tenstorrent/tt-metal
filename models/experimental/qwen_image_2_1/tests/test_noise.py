# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Real-device reproducibility and distribution checks for initial latents."""

import os

import pytest


@pytest.mark.timeout(180)
def test_initial_noise_on_tt(tmp_path):
    if os.environ.get("QWEN_IMAGE21_TEST_NOISE") != "1":
        pytest.skip("set QWEN_IMAGE21_TEST_NOISE=1 and reserve TT_VISIBLE_DEVICES")
    bdf = os.environ.get("TT_VISIBLE_DEVICES")
    assert bdf, "TT_VISIBLE_DEVICES must identify the reserved card"
    from models.experimental.qwen_image_2_1.validation.tt_noise_validate import main

    report = main(["--device-bdf", bdf, "--output-dir", str(tmp_path / "noise")])
    assert report["shape"] == [1, 384, 64]
    assert report["same_seed_bitwise"] and report["zero_seed_bitwise"]
    assert report["different_seed_changes_output"]
