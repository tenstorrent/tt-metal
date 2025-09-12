# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

from models.common.utility_functions import skip_for_grayskull


# This test will run all the nightly fast dispatch tests for all supported TTT models in CI [N150 / N300 only]
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "model_weights",
    [
        "/mnt/MLPerf/tt_dnn-models/qwen/Qwen2.5-VL-3B-Instruct",
    ],
    ids=[
        "qwen25_vl-3B",
    ],
)
def test_ci_dispatch(model_weights):
    logger.info(f"Running fast dispatch tests for {model_weights}")
    if os.getenv("HF_MODEL"):
        del os.environ["HF_MODEL"]
        del os.environ["TT_CACHE_PATH"]
    os.environ["HF_MODEL"] = model_weights
    os.environ["TT_CACHE_PATH"] = model_weights

    # Pass the exit code of pytest to proper keep track of failures during runtime
    exit_code = pytest.main(
        [
            "models/demos/qwen25_vl/tests/test_rms_norm.py",
            "models/demos/qwen25_vl/tests/test_mlp.py",
            "models/demos/qwen25_vl/tests/test_patch_merger.py",
            "models/demos/qwen25_vl/tests/test_vision_attention.py",
            "models/demos/qwen25_vl/tests/test_vision_block.py",
            "models/demos/qwen25_vl/tests/test_model.py",
            "models/demos/qwen25_vl/tests/test_wrapped_model.py",
            "models/demos/qwen25_vl/tests/test_windowed_sdpa.py",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
