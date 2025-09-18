# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

from models.common.utility_functions import skip_for_grayskull
from models.tt_transformers.tt.common import get_hf_tt_cache_path


# This test will run all the nightly fast dispatch tests for all supported TTT models in CI [N150 / N300 only]
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "model_weights",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    ids=[
        "ttt-llama3.2-1B",
        "ttt-llama3.2-3B",
        "ttt-llama3.1-8B",
        "ttt-llama3.2-11B",
        "ttt-mistral-7B-v0.3",
    ],
)
def test_ci_dispatch(model_weights):
    logger.info(f"Running fast dispatch tests for {model_weights}")

    if os.getenv("LLAMA_DIR"):
        del os.environ["LLAMA_DIR"]
    os.environ["HF_MODEL"] = model_weights
    os.environ["TT_CACHE_PATH"] = get_hf_tt_cache_path(model_weights)

    # Pass the exit code of pytest to proper keep track of failures during runtime
    exit_code = pytest.main(
        [
            "models/tt_transformers/tests/test_embedding.py",
            "models/tt_transformers/tests/test_rms_norm.py",
            "models/tt_transformers/tests/test_mlp.py",
            "models/tt_transformers/tests/test_attention.py",
            "models/tt_transformers/tests/test_attention_prefill.py",
            "models/tt_transformers/tests/test_decoder.py",
            "models/tt_transformers/tests/test_decoder_prefill.py",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
