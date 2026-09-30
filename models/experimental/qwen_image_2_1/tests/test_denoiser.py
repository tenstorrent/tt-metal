# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare real Blackhole denoising with the pinned CUDA pipeline capture."""

import json
import os
import re
from pathlib import Path

import pytest


@pytest.mark.timeout(7200)
def test_denoiser_against_cuda_capture(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    capture = os.environ.get("QWEN_IMAGE21_CUDA_CAPTURE")
    checkpoint = os.environ.get("QWEN_IMAGE21_CHECKPOINT")
    device_bdf = os.environ.get("TT_VISIBLE_DEVICES")
    if not capture and not checkpoint:
        pytest.skip("set QWEN_IMAGE21_CUDA_CAPTURE and QWEN_IMAGE21_CHECKPOINT to run the hardware test")
    if not all((capture, checkpoint, device_bdf)):
        pytest.fail("set QWEN_IMAGE21_CUDA_CAPTURE, QWEN_IMAGE21_CHECKPOINT, and TT_VISIBLE_DEVICES together")
    parallel = os.environ.get("QWEN_IMAGE21_TEST_TENSOR_PARALLEL", "1")
    if parallel not in {"1", "2"}:
        pytest.fail("QWEN_IMAGE21_TEST_TENSOR_PARALLEL must be 1 or 2")
    bdfs = [value.strip() for value in device_bdf.split(",")]
    if (
        len(bdfs) != int(parallel)
        or len(set(bdfs)) != len(bdfs)
        or any(re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value) is None for value in bdfs)
    ):
        pytest.fail("TT_VISIBLE_DEVICES must identify one distinct reserved PCI BDF per tensor-parallel rank")
    if not request.config.pluginmanager.hasplugin("timeout"):
        pytest.fail("install pytest-timeout before running this hardware test")

    from models.experimental.qwen_image_2_1.validation.tt_full_denoise import main as run_denoise

    arguments = [
        "--cuda-dir",
        capture,
        "--checkpoint",
        checkpoint,
        "--output-dir",
        str(tmp_path / "tt_output"),
        "--device-bdf",
        device_bdf,
    ]
    arguments.extend(("--tensor-parallel", parallel))
    if os.environ.get("QWEN_IMAGE21_TEST_RESIDENT_BLOCK_WEIGHTS") == "1":
        arguments.append("--resident-block-weights")
    steps = os.environ.get("QWEN_IMAGE21_TEST_STEPS", "1")
    if steps != "full":
        arguments.extend(("--max-steps", steps))
    run_denoise(arguments)

    execution = json.loads((tmp_path / "tt_output/execution.json").read_text())
    assert execution["cards"] == int(parallel)
    assert execution["device_bdfs"] == bdfs
    assert execution["resident_block_weights"] == (os.environ.get("QWEN_IMAGE21_TEST_RESIDENT_BLOCK_WEIGHTS") == "1")
    report = json.loads((tmp_path / "tt_output/report.json").read_text())
    progress = json.loads((tmp_path / "tt_output/progress.json").read_text())
    total_steps = json.loads((Path(capture) / "manifest.json").read_text())["steps"]
    expected_steps = total_steps if steps == "full" else int(steps)
    assert len(report) == expected_steps
    assert progress["status"] == ("complete" if expected_steps == total_steps else "partial"), progress
    assert progress["completed_steps"] == len(report), progress
    for row in report:
        assert row["latent_pcc"] >= 0.98, row
        assert row["velocity_pcc"] >= 0.98, row
    assert (tmp_path / f"tt_output/step_{len(report) - 1:03d}/updated_latents.pt").is_file()
