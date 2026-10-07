# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

from models.tt_transformers.tt.common import get_hf_tt_cache_path


# This test will run all the nightly fast dispatch tests for all supported TTT models in CI [N150 / N300 only]
@pytest.mark.parametrize(
    "model_weights",
    [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "allenai/olmOCR-2-7B-1025",
    ],
    ids=[
        "qwen25_vl-3B",
        "olmOCR-2-7B",
    ],
)
def test_ci_dispatch(model_weights):
    logger.info(f"Running fast dispatch tests for {model_weights}")
    previous_data_parallel = os.environ.get("TT_DATA_PARALLEL")
    os.environ["HF_MODEL"] = model_weights
    os.environ["TT_CACHE_PATH"] = get_hf_tt_cache_path(model_weights)
    if "olmOCR-2-7B" in model_weights:
        os.environ["TT_DATA_PARALLEL"] = "4"

    # Pass the exit code of pytest to proper keep track of failures during runtime
    try:
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
    finally:
        if previous_data_parallel is None:
            os.environ.pop("TT_DATA_PARALLEL", None)
        else:
            os.environ["TT_DATA_PARALLEL"] = previous_data_parallel
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
