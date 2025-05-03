# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.falcon7b_common.demo.demo import run_falcon_demo_kv


@pytest.mark.parametrize(
    "perf_mode, expected_perf_metrics, greedy_sampling, expected_greedy_output_path",
    (
        (False, None, True, "models/demos/grayskull/falcon7b/expected_greedy_output.json"),
        (False, None, True, None),
        (False, None, False, None),
    ),
    ids=[
        "default_mode_greedy_verify",
        "default_mode_greedy",
        "default_mode_stochastic",
    ],
)
def test_demo(
    perf_mode,  # Option to measure perf using max seq length (with invalid outputs) and expected perf (t/s)
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
        max_seq_len=1024,
        model_config_strs_prefill_decode=["BFLOAT16-DRAM", "BFLOAT16-DRAM"],
        model_location_generator=model_location_generator,
        get_tt_cache_path=get_tt_cache_path,
        mesh_device=device,
        perf_mode=perf_mode,
        greedy_sampling=greedy_sampling,
        expected_perf_metrics=expected_perf_metrics,
        expected_greedy_output_path=expected_greedy_output_path,
    )
