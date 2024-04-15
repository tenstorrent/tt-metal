# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.falcon7b.demo.demo import run_falcon_demo_kv
from models.utility_functions import is_wormhole_b0, get_devices_for_t3000


@pytest.mark.parametrize("perf_mode", (False,))  # Option to measure perf using max seq length (with invalid outputs)
@pytest.mark.parametrize("num_devices", (1, 2, 3, 4, 5, 6, 7, 8))
def test_demo_multichip(
    perf_mode,
    num_devices,
    user_input,
    model_location_generator,
    get_tt_cache_path,
    all_devices,
    use_program_cache,
):
    assert is_wormhole_b0(), "Multi-chip is only supported for Wormhole B0"
    devices = get_devices_for_t3000(all_devices, num_devices)

    return run_falcon_demo_kv(
        user_input=user_input,
        batch_size=32,
        max_seq_len=1024,
        model_config_strs_prefill_decode=["BFLOAT16-DRAM", "BFLOAT16-L1_SHARDED"],
        model_location_generator=model_location_generator,
        get_tt_cache_path=get_tt_cache_path,
        devices=devices,
        perf_mode=perf_mode,
    )
