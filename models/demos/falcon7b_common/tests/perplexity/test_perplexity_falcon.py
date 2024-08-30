# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.falcon7b_common.tests.perplexity.run_perplexity_falcon import run_test_perplexity
from models.utility_functions import is_wormhole_b0


@pytest.mark.parametrize(
    "llm_mode, batch_size, max_seq_len, model_config_str, num_samples, expected_ppl, expected_top1, expected_top5",
    (
        ("prefill", 1, 128, "BFLOAT16-DRAM", 64, 19.93, 0.41, 0.66),
        ("prefill", 1, 1024, "BFLOAT16-DRAM", 64, 11.41, 0.49, 0.72),
        ("prefill", 1, 2048, "BFLOAT16-DRAM", 64, 9.96, 0.50, 0.74),
        ("decode", 32, 128, "BFLOAT16-L1_SHARDED", 64, 20.25, 0.40, 0.66),
        ("decode", 32, 1024, "BFLOAT16-L1_SHARDED", 64, 11.63, 0.48, 0.72),
        ("decode", 32, 2048, "BFLOAT16-L1_SHARDED", 64, 10.18, 0.50, 0.74),
    ),
    ids=[
        "prefill_seq128_dram",
        "prefill_seq1024_dram",
        "prefill_seq2048_dram",
        "decode_128_l1_sharded",
        "decode_1024_l1_sharded",
        "decode_2048_l1_sharded",
    ],
)
@pytest.mark.parametrize("enable_async_mode", (True,), indirect=True)  # Option to run Falcon in Async mode
@pytest.mark.parametrize("mesh_device", (1,), indirect=True)
def test_perplexity(
    llm_mode,
    batch_size,
    max_seq_len,
    model_config_str,
    num_samples,  # Total number of prompts to evaluate (all if None)
    expected_ppl,
    expected_top1,
    expected_top5,
    enable_async_mode,
    model_location_generator,
    get_tt_cache_path,
    mesh_device,
    use_program_cache,
):
    assert is_wormhole_b0(), "This test is only for Wormhole B0"

    run_test_perplexity(
        llm_mode,
        batch_size,
        max_seq_len,
        model_config_str,
        model_location_generator,
        get_tt_cache_path,
        mesh_device,
        num_samples,
        {"ppl": expected_ppl, "top1_acc": expected_top1, "top5_acc": expected_top5},
    )
