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
def test_llama_ci_dispatch(llama3_model):
    logger.info(f"Running fast dispatch tests for {llama3_model}")
    os.environ["LLAMA_DIR"] = llama3_model
    pytest.main(
        [
            "models/demos/llama3/tests/test_llama_embedding.py",
            "models/demos/llama3/tests/test_llama_rms_norm.py",
            "models/demos/llama3/tests/test_llama_mlp.py",
            "models/demos/llama3/tests/test_llama_attention.py",
            "models/demos/llama3/tests/test_llama_attention_prefill.py",
            "models/demos/llama3/tests/test_llama_decoder.py",
            "models/demos/llama3/tests/test_llama_decoder_prefill.py",
        ]
    )
