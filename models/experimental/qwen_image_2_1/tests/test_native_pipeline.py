# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Raw-prompt TT pipeline smoke test with no activation capture inputs."""

import json
import os
import re
from pathlib import Path

import pytest


@pytest.mark.timeout(7200)
def test_native_pipeline(tmp_path: Path, request: pytest.FixtureRequest):
    if os.environ.get("QWEN_IMAGE21_TEST_NATIVE") != "1":
        pytest.skip("set QWEN_IMAGE21_TEST_NATIVE=1 and reserve a Blackhole card")
    names = (
        "QWEN_IMAGE21_CHECKPOINT",
        "QWEN_IMAGE21_ENCODER_CHECKPOINT",
        "QWEN_IMAGE21_VAE_CHECKPOINT",
        "TT_VISIBLE_DEVICES",
    )
    settings = {name: os.environ.get(name) for name in names}
    if not all(settings.values()):
        pytest.fail("configure DiT, encoder, and VAE checkpoint paths and TT_VISIBLE_DEVICES")
    bdf = settings["TT_VISIBLE_DEVICES"]
    parallel = os.environ.get("QWEN_IMAGE21_TEST_TENSOR_PARALLEL", "1")
    if parallel not in {"1", "2"}:
        pytest.fail("QWEN_IMAGE21_TEST_TENSOR_PARALLEL must be 1 or 2")
    bdfs = [value.strip() for value in bdf.split(",")]
    if (
        len(bdfs) != int(parallel)
        or len(set(bdfs)) != len(bdfs)
        or any(re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value) is None for value in bdfs)
    ):
        pytest.fail("TT_VISIBLE_DEVICES must identify one distinct reserved PCI BDF per tensor-parallel rank")
    if not request.config.pluginmanager.hasplugin("timeout"):
        pytest.fail("install pytest-timeout before running the hardware test")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        pytest.fail("set CUDA_VISIBLE_DEVICES='' before importing torch to verify CUDA independence")
    from models.experimental.qwen_image_2_1.validation.tt_full_denoise import main

    output = tmp_path / "native"
    arguments = [
        "--native",
        "--checkpoint",
        settings["QWEN_IMAGE21_CHECKPOINT"],
        "--tt-encoder-checkpoint",
        settings["QWEN_IMAGE21_ENCODER_CHECKPOINT"],
        "--vae-checkpoint",
        settings["QWEN_IMAGE21_VAE_CHECKPOINT"],
        "--device-bdf",
        bdf,
        "--output-dir",
        str(output),
        "--prompt",
        "the quick brown fox jumps over the lazy dog",
        "--height",
        "256",
        "--width",
        "384",
        "--steps",
        "20",
        "--seed",
        "42",
    ]
    arguments.extend(("--tensor-parallel", parallel))
    if os.environ.get("QWEN_IMAGE21_TEST_RESIDENT_BLOCK_WEIGHTS") == "1":
        arguments.append("--resident-block-weights")
    steps = os.environ.get("QWEN_IMAGE21_TEST_STEPS", "1")
    if steps != "full":
        arguments.extend(("--max-steps", steps))
    main(arguments)
    execution = json.loads((output / "execution.json").read_text())
    assert execution["cards"] == int(parallel)
    assert execution["device_bdfs"] == bdfs
    assert execution["resident_block_weights"] == (os.environ.get("QWEN_IMAGE21_TEST_RESIDENT_BLOCK_WEIGHTS") == "1")
    provenance = json.loads((output / "inputs.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    assert provenance["injected_activations"] is False
    assert provenance["cuda_available"] is False
    assert provenance["prompt_encoder"] == "TT"
    assert provenance["initial_noise"] == "TT"
    assert progress["completed_steps"] == (20 if steps == "full" else int(steps))
    assert progress["status"] == ("complete" if steps == "full" or int(steps) == 20 else "partial")
    assert (output / "tt_vae.png").is_file()
    assert (output / "decoded_image.pt").is_file()
