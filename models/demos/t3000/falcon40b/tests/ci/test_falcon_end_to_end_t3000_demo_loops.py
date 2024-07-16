# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.t3000.falcon40b.demo.demo import run_falcon_demo_kv
from models.demos.t3000.falcon40b.tt.model_config import get_model_config, model_config_entries
from models.utility_functions import (
    disable_compilation_reports,
    skip_for_grayskull,
    skip_for_wormhole_b0,
)


@skip_for_grayskull("Requires eth connected devices to run")
@skip_for_wormhole_b0("See GH Issue #10320")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "num_loops",
    (1,),
    ids=[
        "loops_1",
    ],
)
@pytest.mark.parametrize("perf_mode", (False,))  # Option to measure perf using max seq length (with invalid outputs)
@pytest.mark.parametrize("greedy_sampling", (False,))
@pytest.mark.parametrize("max_seq_len", (128,))
def test_demo(
    num_loops,
    perf_mode,
    greedy_sampling,
    max_seq_len,
    model_location_generator,
    get_tt_cache_path,
    t3k_device_mesh,
    use_program_cache,
):
    # disable_persistent_kernel_cache()
    disable_compilation_reports()

    user_inputs = [
        ["Tell me a joke."],
        ["What is the capital of Serbia?"],
        ["Why to cows muh?"],
        ["Count from 1 to 100."],
        ["Who is Jim Keller?"],
        ["Is Tenstorrent the coolest company to work for or what?"],
        ["Why do all my tests hang?"],
        ["How many days does a year have?"],
        ["Can you tell me a story please?"],
        ["How do I get better at piano?"],
    ]

    for i in range(num_loops):
        input_idx = i % len(user_inputs)
        print(f"Running demo prompt: {user_inputs[input_idx]}")
        run_falcon_demo_kv(
            user_input=user_inputs[input_idx],
            model_version=model_config_entries["_name_or_path"],
            model_config_str_for_decode="BFLOAT8_B-SHARDED",  # Decode model config
            model_config_str_for_prefill="BFLOAT16-DRAM",  # Prefill model config
            batch_size=32,
            num_layers=model_config_entries["num_hidden_layers"],
            max_seq_len=max_seq_len,
            model_location_generator=model_location_generator,
            get_tt_cache_path=get_tt_cache_path,
            device_mesh=t3k_device_mesh,
            prefill_on_host=False,
            perf_mode=perf_mode,
            greedy_sampling=greedy_sampling,
        )
