import pytest
import os
from loguru import logger
from models.utility_functions import skip_for_grayskull


# This test will run all the nightly fast dispatch tests for all supported Llama3 models in CI
# [N150 / N300 only]
@skip_for_grayskull("Requires wormhole_b0 to run")
def test_llama_ci_dispatch():
    dir_1b = "/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/"
    dir_3b = "/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/"
    dir_8b = "/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/"

    for dir_path in [dir_1b, dir_3b, dir_8b]:
        logger.info(f"Running fast dispatch tests for {dir_path}")
        os.environ["LLAMA_DIR"] = dir_path
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
