# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.falcon7b_common.tests.perplexity.run_perplexity_falcon import run_test_perplexity


@pytest.mark.parametrize(
    "llm_mode, batch_size, max_seq_len, num_samples, expected_ppl, expected_top1, expected_top5",
    (
        ("prefill", 32, 128, 64, 19.67, 0.41, 0.66),
        ("prefill", 32, 1024, 64, 11.19, 0.48, 0.72),
        ("prefill", 32, 2048, 64, 9.81, 0.50, 0.74),
        ("decode", 64, 128, 64, 19.67, 0.41, 0.66),
        ("decode", 64, 1024, 64, 11.19, 0.48, 0.72),
        ("decode", 64, 2048, 64, 9.81, 0.50, 0.74),
    ),
    ids=[
        "prefill_seq128",
        "prefill_seq1024",
        "prefill_seq2048",
        "decode_128",
        "decode_1024",
        "decode_2048",
    ],
)
def test_perplexity_huggingface(
    llm_mode,
    batch_size,
    max_seq_len,
    num_samples,  # Total number of prompts to evaluate (all if None)
    expected_ppl,
    expected_top1,
    expected_top5,
    model_location_generator,
):
    run_test_perplexity(
        llm_mode,
        batch_size,
        max_seq_len,
        None,
        model_location_generator,
        None,
        None,
        num_samples,
        {"ppl": expected_ppl, "top1_acc": expected_top1, "top5_acc": expected_top5},
        use_hf_model=True,
    )
