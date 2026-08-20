# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from loguru import logger


# This test will run all the nightly fast dispatch tests for LFM2.5-VL in CI [N150 / N300 only]
@pytest.mark.parametrize(
    "hf_model_name",
    ["LiquidAI/LFM2.5-VL-1.6B"],
    ids=["LFM2.5-VL-1.6B"],
)
def test_ci_dispatch(hf_model_name, is_ci_env, model_location_generator):
    if not is_ci_env:
        pytest.skip("Skipping CI dispatch tests when running locally.")

    model_weights_path = str(model_location_generator(hf_model_name, download_if_ci_v2=True, ci_v2_timeout_in_s=1800))
    os.environ["HF_MODEL"] = model_weights_path
    os.environ["TT_CACHE_PATH"] = model_weights_path

    logger.info(f"Running fast dispatch tests for {model_weights_path}")

    tests = [
        "models/demos/multimodal/lfm25_vl/tests/test_load_checkpoints.py",
        "models/demos/multimodal/lfm25_vl/tests/test_short_conv.py",
        "models/demos/multimodal/lfm25_vl/tests/test_mlp.py",
        "models/demos/multimodal/lfm25_vl/tests/test_projector.py",
    ]

    # Pass the exit code of pytest to properly keep track of failures during runtime
    exit_code = pytest.main(tests + ["-x"])
    if exit_code != pytest.ExitCode.OK:
        pytest.fail(
            f"Pytest failed with exit code {exit_code} for {hf_model_name}. Check logs above for details.",
            pytrace=False,
        )
