# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.falcon7b.demo.demo import run_falcon_demo_kv


@pytest.mark.parametrize(
    "perf_mode, max_seq_len, expected_perf_metrics, greedy_sampling, expected_greedy_output_path",
    (
        (True, 128, {"prefill_t/s": 1370, "decode_t/s": 430, "decode_t/s/u": 13.4}, False, None),
        (True, 1024, {"prefill_t/s": 1770, "decode_t/s": 370, "decode_t/s/u": 11.6}, False, None),
        (True, 2048, {"prefill_t/s": 1600, "decode_t/s": 360, "decode_t/s/u": 11.2}, False, None),
        (True, 1024, None, False, None),
        (False, 1024, None, True, "models/demos/wormhole/falcon7b/expected_greedy_output.json"),
        (False, 1024, None, True, None),
        (False, 1024, None, False, None),
    ),
    ids=[
        "perf_mode_128_stochastic_verify",
        "perf_mode_1024_stochastic_verify",
        "perf_mode_2048_stochastic_verify",
        "perf_mode_1024_stochastic",
        "default_mode_1024_greedy_verify",
        "default_mode_1024_greedy",
        "default_mode_1024_stochastic",
    ],
)
def test_demo(
    perf_mode,  # Option to measure perf using max seq length (with invalid outputs) and expected perf (t/s)
    max_seq_len,
    expected_perf_metrics,  # Expected perf (t/s) for prefill and decode in perf mode
    greedy_sampling,  # Option to use greedy decoding instead of top-k/p
    expected_greedy_output_path,  # Path for expected outputs for greedy decoding
    user_input,
    model_location_generator,
    get_tt_cache_path,
    device,
    use_program_cache,
):
    return run_falcon_demo_kv(
        user_input=user_input,
        batch_size=32,
        max_seq_len=max_seq_len,
        model_config_strs_prefill_decode=["BFLOAT16-DRAM", "BFLOAT16-L1_SHARDED"],
        model_location_generator=model_location_generator,
        get_tt_cache_path=get_tt_cache_path,
        devices=[device],
        perf_mode=perf_mode,
        greedy_sampling=greedy_sampling,
        expected_perf_metrics=expected_perf_metrics,
        expected_greedy_output_path=expected_greedy_output_path,
    )
