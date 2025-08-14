# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

@pytest.mark.parametrize(
    "model_weights",
    [
        "/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/",
    ],
    ids=[
        "llama-3.3-70b-instruct",
    ],
)
def test_ci_evals(model_weights):
    logger.info(f"Running Evals test for {model_weights}")
    os.environ["LLAMA_DIR"] = model_weights
    
    # Pass the exit code of pytest to proper keep track of failures during runtime
    exit_code = pytest.main(
        [
            "models/demos/llama3_70b_galaxy/demo/text_demo.py -k 'evals-1'",
            "models/demos/llama3_70b_galaxy/demo/text_demo.py -k 'evals-32'",
            "models/demos/llama3_70b_galaxy/demo/text_demo.py -k 'evals-long-prompts'",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more evals tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
