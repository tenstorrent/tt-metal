# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.falcon7b.demo.demo import run_falcon_demo_kv
from models.utility_functions import is_wormhole_b0, get_devices_for_t3000


@pytest.mark.parametrize(
    "perf_mode, expected_perf_prefill_decode, greedy_sampling, expected_greedy_output_path",
    (
        (True, [6600, 1050], False, None),
        (True, None, False, None),
        (False, None, True, "models/demos/t3000/falcon7b/expected_greedy_output.json"),
        (False, None, True, None),
        (False, None, False, None),
    ),
    ids=[
        "perf_mode_stochastic_verify",
        "perf_mode_stochastic",
        "default_mode_greedy_verify",
        "default_mode_greedy",
        "default_mode_stochastic",
    ],
)
@pytest.mark.parametrize("async_mode", (True,))  # Option to run Falcon in Async mode
@pytest.mark.parametrize("num_devices", (1, 2, 3, 4, 5, 6, 7, 8))
def test_demo_multichip(
    perf_mode,  # Option to measure perf using max seq length (with invalid outputs) and expected perf (t/s)
    expected_perf_prefill_decode,  # Expected perf (t/s) for prefill and decode in perf mode
    greedy_sampling,  # Option to use greedy decoding instead of top-k/p
    expected_greedy_output_path,  # Path for expected outputs for greedy decoding
    num_devices,
    user_input,
    model_location_generator,
    get_tt_cache_path,
    all_devices,
    use_program_cache,
    async_mode,
):
    assert is_wormhole_b0(), "Multi-chip is only supported for Wormhole B0"
    devices = get_devices_for_t3000(all_devices, num_devices)

    for device in devices:
        device.enable_async(async_mode)
    return run_falcon_demo_kv(
        user_input=user_input,
        batch_size=32,
        max_seq_len=1024,
        model_config_strs_prefill_decode=["BFLOAT16-DRAM", "BFLOAT16-L1_SHARDED"],
        model_location_generator=model_location_generator,
        get_tt_cache_path=get_tt_cache_path,
        devices=devices,
        perf_mode=perf_mode,
        greedy_sampling=greedy_sampling,
        expected_perf_prefill_decode=expected_perf_prefill_decode,
        expected_greedy_output_path=expected_greedy_output_path,
    )
