# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from loguru import logger

from models.utility_functions import skip_for_grayskull


# This test will run all the nightly fast dispatch tests for all supported TTT models in CI [N150 / N300 only]
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "model_weights",
    [
        "google/gemma-3-4b-it",
    ],
    ids=[
        "gemma-3-4b-it",
    ],
)
def test_ci_dispatch(model_weights):
    pytest.skip("Skipping CI dispatch tests for now")
    os.environ["HF_MODEL"] = model_weights
    os.environ["TT_CACHE_PATH"] = model_weights
    logger.info(f"Running fast dispatch tests for {model_weights}")

    # Pass the exit code of pytest to proper keep track of failures during runtime
    exit_code = pytest.main(
        [
            "models/demos/siglip/tests/test_attention.py",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
