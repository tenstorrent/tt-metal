# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from loguru import logger


# This test will run all the nightly fast dispatch tests for Janus Pro in CI
# on Blackhole SKUs only (P150 / P300 / QuietBox 2).
@pytest.mark.parametrize(
    "hf_model_name",
    ["deepseek-community/Janus-Pro-7B"],
    ids=["Janus-Pro-7B"],
)
def test_ci_dispatch(hf_model_name, is_ci_env, model_location_generator):
    if not is_ci_env:
        pytest.skip("Skipping CI dispatch tests when running locally.")

    model_weights_path = str(model_location_generator(hf_model_name, download_if_ci_v2=True, ci_v2_timeout_in_s=1800))
    os.environ["HF_MODEL"] = model_weights_path
    os.environ["TT_CACHE_PATH"] = model_weights_path

    logger.info(f"Running fast dispatch tests for {model_weights_path}")

    tests = [
        "models/experimental/janus_pro/tests/test_patch_embedding.py",
        "models/experimental/janus_pro/tests/test_vision_embedding.py",
        "models/experimental/janus_pro/tests/test_vision_layernorm.py",
        "models/experimental/janus_pro/tests/test_vision_mlp.py",
        "models/experimental/janus_pro/tests/test_vision_attention.py",
        "models/experimental/janus_pro/tests/test_vision_transformer_block.py",
        "models/experimental/janus_pro/tests/test_vision_transformer.py",
    ]

    # Pass the exit code of pytest to proper keep track of failures during runtime
    exit_code = pytest.main(tests + ["-x"])
    if exit_code != pytest.ExitCode.OK:
        pytest.fail(
            f"Pytest failed with exit code {exit_code} for {hf_model_name}. " "Check logs above for details.",
            pytrace=False,
        )
