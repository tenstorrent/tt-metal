# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

from models.tt_transformers.tt.common import get_hf_tt_cache_path


# This test runs the nightly SpeechT5 dispatch suite in CI [N150 / N300 only]
@pytest.mark.parametrize(
    "model_weights",
    [
        "microsoft/speecht5_tts",
    ],
    ids=[
        "speecht5_tts",
    ],
)
def test_speecht5_ci_dispatch(model_weights):
    logger.info(f"Running fast dispatch tests for {model_weights}")
    os.environ["HF_MODEL"] = model_weights
    os.environ["TT_CACHE_PATH"] = get_hf_tt_cache_path(model_weights)
    # Keep defaults close to observed CI PCCs while still allowing small run-to-run variation.
    # setdefault keeps these overridable from workflow/job-level environment.
    os.environ.setdefault("SPEECHT5_E2E_ENCODER_PCC_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_E2E_DECODER_PCC_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_E2E_PRE_POSTNET_PCC_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_E2E_POST_POSTNET_PCC_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_E2E_STOP_LOGITS_PCC_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_LAYER_PCC_MULTI_DECODER_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_LAYER_PCC_MULTI_CONV_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_LAYER_PCC_MULTI_MEL_AFTER_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_LAYER_PCC_TRUE_AR_DECODER_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_LAYER_PCC_TRUE_AR_CONV_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_LAYER_PCC_TRUE_AR_MEL_AFTER_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_AR_TRACKING_MULTI_DECODER_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_AR_TRACKING_MULTI_MEL_PRE_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_AR_TRACKING_MULTI_MEL_POST_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_AR_TRACKING_MULTI_STOP_MIN", "0.99")
    os.environ.setdefault("SPEECHT5_AR_TRACKING_TRUE_AR_MEL_SEQ_MIN", "0.99")

    # Pass the exit code of pytest to proper keep track of failures during runtime
    exit_code = pytest.main(
        [
            "models/experimental/speecht5_tts/tests/test_autoregressive_layer_by_layer_pcc.py",
            "models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py",
            "models/experimental/speecht5_tts/tests/test_end_to_end_pcc.py",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
