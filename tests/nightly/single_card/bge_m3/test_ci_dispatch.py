# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

from models.tt_transformers.tt.common import get_hf_tt_cache_path


# This test will run all the nightly fast dispatch tests for all supported TTT models in CI [N150 / N300 only]
@pytest.mark.parametrize(
    "model_weights",
    [
        "BAAI/bge-m3",
    ],
    ids=[
        "bge-m3",
    ],
)
def test_ci_dispatch(model_weights):
    logger.info(f"Running fast dispatch tests for {model_weights}")
    os.environ["HF_MODEL"] = model_weights
    os.environ["TT_CACHE_PATH"] = get_hf_tt_cache_path(model_weights)

    # Pass the exit code of pytest to proper keep track of failures during runtime
    exit_code = pytest.main(
        [
            "models/demos/wormhole/bge_m3/tests/pcc/test_layernorm.py",
            "models/demos/wormhole/bge_m3/tests/pcc/test_embeddings.py",
            "models/demos/wormhole/bge_m3/tests/pcc/test_mlp.py",
            "models/demos/wormhole/bge_m3/tests/pcc/test_attention.py",
            "models/demos/wormhole/bge_m3/tests/pcc/test_transformer_block.py",
            "models/demos/wormhole/bge_m3/tests/pcc/test_model.py",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
