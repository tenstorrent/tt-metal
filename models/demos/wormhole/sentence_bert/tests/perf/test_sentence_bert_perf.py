# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.common.utility_functions import is_wormhole_b0, run_for_wormhole_b0
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf,test",
    [
        # 460.0 was the pre-row_major_output baseline. Raised to 546.5 after the
        # ff2/qkv subblock upgrade in models/demos/wormhole/sentence_bert/ttnn/common.py
        # (commit 0a421bb6a56) which dropped matmul time -21.4% and lifted total
        # device-kernel samples/sec from ~460 to ~546.5. Locks the win in as the
        # new regression floor (test allows ±3% margin, so anything below ~530
        # sps now fails).
        [8, 546.5, "sentence_bert"],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_sentence_bert(batch_size, expected_perf, test):
    subdir = "ttnn_sentence_bert_model"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 0

    command = f"pytest models/demos/wormhole/sentence_bert/tests/pcc/test_ttnn_sentencebert_model.py::test_ttnn_sentence_bert_model"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_sentence_bert{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
