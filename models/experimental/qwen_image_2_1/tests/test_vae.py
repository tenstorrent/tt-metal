# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Blackhole VAE comparisons against pinned independent CUDA captures."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.timeout(600)
@pytest.mark.parametrize("isolated", [True, False], ids=["components", "full_decode"])
def test_vae_against_cuda(tmp_path: Path, isolated: bool):
    capture = os.environ.get("QWEN_IMAGE21_VAE_CAPTURE")
    checkpoint = os.environ.get("QWEN_IMAGE21_VAE_CHECKPOINT")
    device = os.environ.get("TT_VISIBLE_DEVICES")
    if not capture and not checkpoint:
        pytest.skip("set QWEN_IMAGE21_VAE_CAPTURE and QWEN_IMAGE21_VAE_CHECKPOINT for real TT validation")
    if not all((capture, checkpoint, device)):
        pytest.fail("capture, checkpoint, and TT_VISIBLE_DEVICES must be configured together")
    from models.experimental.qwen_image_2_1.validation.tt_vae_decode import main

    output = tmp_path / "vae"
    arguments = ["--cuda-dir", capture, "--checkpoint", checkpoint, "--output-dir", str(output), "--device-bdf", device]
    if isolated:
        arguments.append("--isolated-only")
    main(arguments)
    report = json.loads((output / "report.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "complete", progress
    assert report
    if isolated:
        stages = {row["stage"] for row in report}
        assert {
            "decoder.conv_in",
            "decoder.mid_block.attentions.0",
            "decoder.conv_out",
            "decoder.up_blocks.2.avg_shortcut",
            "decoder.up_blocks.3.avg_shortcut",
        } <= stages
        assert all(row["pcc"] >= 0.99 for row in report), report
        for row in report:
            if row["stage"].endswith(("avg_shortcut", "resample.0")):
                assert row["max_abs_error"] == 0, row
    else:
        assert report[-1]["stage"] == "output"
        assert report[-1]["pcc"] >= 0.99, report[-1]
        assert (output / "tt_vae.png").is_file()
        assert (output / "output.pt").is_file()
