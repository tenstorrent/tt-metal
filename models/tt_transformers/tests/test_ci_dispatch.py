# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from loguru import logger
from models.utility_functions import skip_for_grayskull


# This test will run all the nightly fast dispatch tests for all supported Llama3 models in CI [N150 / N300 only]
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "llama3_model",
    [
        "/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/",
        "/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/",
        "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/",
        "/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/",
    ],
    ids=[
        "llama3.2-1B",
        "llama3.2-3B",
        "llama3.1-8B",
        "llama3.2-11B",
    ],
)
def test_ci_dispatch(llama3_model):
    logger.info(f"Running fast dispatch tests for {llama3_model}")
    os.environ["LLAMA_DIR"] = llama3_model
    pytest.main(
        [
            "models/tt_transformers/tests/test_embedding.py",
            "models/tt_transformers/tests/test_rms_norm.py",
            "models/tt_transformers/tests/test_mlp.py",
            "models/tt_transformers/tests/test_attention.py",
            "models/tt_transformers/tests/test_attention_prefill.py",
            "models/tt_transformers/tests/test_decoder.py",
            "models/tt_transformers/tests/test_decoder_prefill.py",
        ]
    )
