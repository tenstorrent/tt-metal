# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.demos.falcon7b.demo.demo import run_falcon_demo_kv


@pytest.mark.parametrize("perf_mode", (False,))  # Option to measure perf using max seq length (with invalid outputs)
def test_demo(
    perf_mode,
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
        devices=[device],
        perf_mode=perf_mode,
    )
