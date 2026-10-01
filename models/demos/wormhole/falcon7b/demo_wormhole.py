# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.common.utility_functions import is_wormhole_b0
from models.demos.falcon7b_common.demo.demo import run_falcon_demo_kv
from models.demos.utils.device_sku import get_current_device_sku_name


@pytest.mark.parametrize(
    "perf_mode, max_seq_len, has_expected_perf_metrics, greedy_sampling, expected_greedy_output_path",
    (
        (True, 128, True, False, None),
        (True, 1024, True, False, None),
        (True, 2048, True, False, None),
        (True, 128, False, False, None),
        (True, 1024, False, False, None),
        (True, 2048, False, False, None),
        (False, 1024, False, True, "models/demos/wormhole/falcon7b/expected_greedy_output.json"),
        (False, 1024, False, True, None),
        (False, 1024, False, False, None),
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
@pytest.mark.parametrize("mesh_device", (1,), indirect=True)
def test_demo(
    perf_mode,  # Option to measure perf using max seq length (with invalid outputs) and expected perf (t/s)
    max_seq_len,
    has_expected_perf_metrics,
    greedy_sampling,  # Option to use greedy decoding instead of top-k/p
    expected_greedy_output_path,  # Path for expected outputs for greedy decoding
    user_input,
    mesh_device,
    is_ci_env,
):
    if is_ci_env:
        if not expected_greedy_output_path and not has_expected_perf_metrics and not len(user_input) == 1:
            pytest.skip("Skipping test in CI since it provides redundant testing")

    assert is_wormhole_b0()

    batch_size = 32
    if perf_mode:
        json_perf_targets = {
            "prefill_t/s": {128: 2034, 1024: 9880, 2048: 9881}[max_seq_len],
            "decode_t/s": 26 * batch_size,
            "decode_t/s/u": 26,
        }  # performance targets that we aim for (wormhole)
        verify_sku = get_current_device_sku_name()
        verify_batch_size = batch_size
        verify_seq_len = max_seq_len
    else:
        json_perf_targets = {}
        verify_sku = None
        verify_batch_size = None
        verify_seq_len = None

    return run_falcon_demo_kv(
        user_input=user_input,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        model_config_strs_prefill_decode=["BFLOAT16-DRAM", "BFLOAT16-L1_SHARDED"],
        mesh_device=mesh_device,
        perf_mode=perf_mode,
        greedy_sampling=greedy_sampling,
        expected_perf_metrics=None,
        expected_greedy_output_path=expected_greedy_output_path,
        json_perf_targets=json_perf_targets,
        is_ci_env=is_ci_env,
        model_name="falcon-7b" if has_expected_perf_metrics else None,
        sku=verify_sku if has_expected_perf_metrics else None,
        target_batch_size=verify_batch_size if has_expected_perf_metrics else None,
        target_seq_len=verify_seq_len if has_expected_perf_metrics else None,
    )
