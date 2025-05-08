# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.falcon7b_common.demo.demo import run_falcon_demo_kv
from models.utility_functions import is_wormhole_b0


@pytest.mark.parametrize(
    "perf_mode, max_seq_len, expected_perf_metrics, greedy_sampling, expected_greedy_output_path",
    (
        (True, 128, {"prefill_t/s": 21200, "decode_t/s": 7475, "decode_t/s/u": 7.30}, False, None),
        (True, 1024, {"prefill_t/s": 19180, "decode_t/s": 7127, "decode_t/s/u": 6.96}, False, None),
        (True, 2048, {"prefill_t/s": 14500, "decode_t/s": 7219, "decode_t/s/u": 7.05}, False, None),
        (True, 128, None, False, None),
        (True, 1024, None, False, None),
        (True, 2048, None, False, None),
        (False, 1024, None, True, "models/demos/tg/falcon7b/expected_greedy_output.json"),
        (False, 1024, None, True, None),
        (False, 1024, None, False, None),
    ),
    ids=[
        "perf_mode_128_stochastic_verify",
        "perf_mode_1024_stochastic_verify",
        "perf_mode_2048_stochastic_verify",
        "perf_mode_128_stochastic",
        "perf_mode_1024_stochastic",
        "perf_mode_2048_stochastic",
        "default_mode_1024_greedy_verify",
        "default_mode_1024_greedy",
        "default_mode_1024_stochastic",
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    ((8, 4),),
    indirect=True,
    ids=["32chipTG"],
)
def test_demo_multichip(
    perf_mode,  # Option to measure perf using max seq length (with invalid outputs) and expected perf (t/s)
    max_seq_len,
    expected_perf_metrics,  # Expected perf (t/s) for prefill and decode in perf mode
    greedy_sampling,  # Option to use greedy decoding instead of top-k/p
    expected_greedy_output_path,  # Path for expected outputs for greedy decoding
    user_input,
    model_location_generator,
    get_tt_cache_path,
    mesh_device,
    use_program_cache,
    is_ci_env,
    ensure_devices_tg,
):
    assert is_wormhole_b0(), "Multi-chip is only supported for Wormhole B0"
    num_devices = mesh_device.get_num_devices()

    if is_ci_env:
        if not expected_greedy_output_path and not expected_perf_metrics:
            pytest.skip("Skipping test in CI since it provides redundant testing")
        if expected_greedy_output_path:
            pytest.skip("Skipping test in CI due to Issue #11254")
    elif expected_greedy_output_path or expected_perf_metrics:
        assert num_devices == 32, "32 devices are expected for perf and greedy output verification"

    batch_size = 32
    if perf_mode:
        json_perf_targets = {
            "prefill_t/s": {128: None, 1024: None, 2048: None}[max_seq_len],
            "decode_t/s": 26 * batch_size * num_devices,
            "decode_t/s/u": 26,
        }  # performance targets that we aim for (galaxy-tg)
    else:
        json_perf_targets = {}

    return run_falcon_demo_kv(
        user_input=user_input,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        model_config_strs_prefill_decode=["BFLOAT16-DRAM", "BFLOAT16-L1_SHARDED"],
        model_location_generator=model_location_generator,
        get_tt_cache_path=get_tt_cache_path,
        mesh_device=mesh_device,
        perf_mode=perf_mode,
        greedy_sampling=greedy_sampling,
        expected_perf_metrics=expected_perf_metrics,
        expected_greedy_output_path=expected_greedy_output_path,
        json_perf_targets=json_perf_targets,
        is_ci_env=is_ci_env,
    )
