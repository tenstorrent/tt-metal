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
    test_patterns = ["evals-1", "evals-32", "evals-long-prompts"]
    for pattern in test_patterns:
        exit_code = pytest.main(
            [
                "models/demos/llama3_70b_galaxy/demo/text_demo.py",
                "-k",
                pattern,
                "-x"
            ]
        )
        if exit_code == pytest.ExitCode.TESTS_FAILED:
            pytest.fail(
                f"The eval test {pattern} failed. Please check the log above for more info",
                pytrace=False,
            )
