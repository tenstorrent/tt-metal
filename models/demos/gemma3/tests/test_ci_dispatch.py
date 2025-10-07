# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from loguru import logger


# This test will run all the nightly fast dispatch tests for all supported TTT models in CI [N150 / N300 only]
@pytest.mark.parametrize(
    "hf_model_name",
    ["google/gemma-3-4b-it", "google/gemma-3-27b-it"],
    ids=["gemma-3-4b-it", "gemma-3-27b-it"],
)
def test_ci_dispatch(hf_model_name, is_ci_env, is_ci_v2_env, model_location_generator):
    if not is_ci_env and not is_ci_v2_env:
        pytest.skip("Skipping CI dispatch tests when running locally.")

    os.environ["HF_MODEL"] = hf_model_name

    model_weights_path = str(model_location_generator(hf_model_name))
    os.environ["TT_CACHE_PATH"] = model_weights_path

    logger.info(f"Running fast dispatch tests for {model_weights_path}")

    functional_tests = [
        "models/demos/gemma3/tests/test_mmp.py",
        "models/demos/siglip/tests/test_attention.py",
        "models/demos/gemma3/tests/test_patch_embedding.py",
        "models/demos/gemma3/tests/test_vision_attention.py",
        "models/demos/gemma3/tests/test_vision_cross_attention_transformer.py",
        "models/demos/gemma3/tests/test_vision_embedding.py",
        "models/demos/gemma3/tests/test_vision_layernorm.py",
        "models/demos/gemma3/tests/test_vision_mlp.py",
        "models/demos/gemma3/tests/test_vision_pipeline.py",
        # "models/demos/gemma3/tests/test_vision_rmsnorm.py",
        "models/demos/gemma3/tests/test_vision_transformer_block.py",
        "models/demos/gemma3/tests/test_vision_transformer.py",
        "models/tt_transformers/tests/test_embedding.py",
        "models/tt_transformers/tests/test_rms_norm.py",
        "models/tt_transformers/tests/test_mlp.py",
        # "models/tt_transformers/tests/test_attention.py",
        # "models/tt_transformers/tests/test_attention_prefill.py",
        "models/tt_transformers/tests/test_decoder.py",
        "models/tt_transformers/tests/test_decoder_prefill.py",
    ]
    performance_tests = [
        # TODO for mstojko - add your test here once its merged
    ]

    # at the time of writing this performance tests can't/shouldn't be run on CIV2, so this is why we have this hybrid approach
    if is_ci_v2_env:
        tests = functional_tests
    else:
        tests = performance_tests

    args = tests + ["-x"]  # Fail if one of the tests fails

    # Pass the exit code of pytest to proper keep track of failures during runtime
    exit_code = pytest.main(args)
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights_path}. Please check the log above for more info",
            pytrace=False,
        )
