# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare real Blackhole denoising with the pinned CUDA pipeline capture."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_denoiser_against_cuda_capture(tmp_path: Path) -> None:
    capture = os.environ.get("QWEN_IMAGE21_CUDA_CAPTURE")
    checkpoint = os.environ.get("QWEN_IMAGE21_CHECKPOINT")
    device_bdf = os.environ.get("QWEN_IMAGE21_DEVICE_BDF")
    if not all((capture, checkpoint, device_bdf)):
        pytest.skip("set QWEN_IMAGE21_CUDA_CAPTURE, QWEN_IMAGE21_CHECKPOINT, and QWEN_IMAGE21_DEVICE_BDF")

    command = [
        sys.executable,
        "-m",
        "models.experimental.qwen_image_2_1.validation.tt_full_denoise",
        "--cuda-dir",
        capture,
        "--checkpoint",
        checkpoint,
        "--output-dir",
        str(tmp_path / "tt_output"),
        "--device-bdf",
        device_bdf,
    ]
    steps = os.environ.get("QWEN_IMAGE21_TEST_STEPS", "1")
    if steps != "full":
        command.extend(("--max-steps", steps))
    subprocess.run(command, check=True, timeout=7200)

    report = json.loads((tmp_path / "tt_output/report.json").read_text())
    assert len(report) == (
        json.loads((Path(capture) / "manifest.json").read_text())["steps"] if steps == "full" else int(steps)
    )
    for row in report:
        assert row["latent_pcc"] >= 0.98, row
        assert row["velocity_pcc"] >= 0.98, row
    assert (tmp_path / f"tt_output/step_{len(report) - 1:03d}/updated_latents.pt").is_file()
