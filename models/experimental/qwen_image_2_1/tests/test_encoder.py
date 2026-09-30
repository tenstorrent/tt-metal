# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Full text encoder regression against a pinned independent CUDA capture."""

import json
import os

import pytest


@pytest.mark.timeout(600)
def test_prompt_encoder_on_tt(tmp_path):
    capture = os.environ.get("QWEN_IMAGE21_ENCODER_CAPTURE")
    checkpoint = os.environ.get("QWEN_IMAGE21_ENCODER_CHECKPOINT")
    if not capture and not checkpoint:
        pytest.skip("configure a real encoder capture/checkpoint and reserve a TT card")
    bdf = os.environ.get("TT_VISIBLE_DEVICES")
    assert capture and checkpoint and bdf
    from models.experimental.qwen_image_2_1.validation.tt_encoder_validate import main

    output = tmp_path / "encoder"
    main(
        [
            "--cuda-dir",
            capture,
            "--checkpoint",
            checkpoint,
            "--device-bdf",
            bdf,
            "--output-dir",
            str(output),
        ]
    )
    progress = json.loads((output / "progress.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert progress["status"] == "complete", progress
    assert report[-1]["stage"] == "prompt_embeds"
    assert report[-1]["pcc"] >= 0.99
    assert sum(row["stage"].startswith("layer_") and "/" not in row["stage"] for row in report) == 36
    assert next(row for row in report if row["stage"] == "embedding")["max_abs_error"] == 0
