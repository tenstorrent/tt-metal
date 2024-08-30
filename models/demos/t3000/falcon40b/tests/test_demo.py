# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest

from models.demos.t3000.falcon40b.tt.model_config import model_config_entries
from models.demos.t3000.falcon40b.demo.demo import run_falcon_demo_kv


@pytest.mark.parametrize("max_seq_len", (128,))
def test_demo_generate_reference_output(
    max_seq_len, model_location_generator, get_tt_cache_path, t3k_mesh_device, use_program_cache, is_ci_env
):
    if is_ci_env:
        pytest.skip("Skip generating reference output in CI")

    input_file = "models/demos/t3000/falcon40b/demo/input_data.json"

    generated_text, measurements = run_falcon_demo_kv(
        user_input=input_file,
        model_version=model_config_entries["_name_or_path"],
        model_config_str_for_decode="BFLOAT8_B-SHARDED",  # Decode model config
        model_config_str_for_prefill="BFLOAT8_B-DRAM",  # Prefill model config
        batch_size=32,
        num_layers=model_config_entries["num_hidden_layers"],
        max_seq_len=max_seq_len,
        model_location_generator=model_location_generator,
        get_tt_cache_path=get_tt_cache_path,
        mesh_device=t3k_mesh_device,
        prefill_on_host=False,
        perf_mode=False,
        greedy_sampling=True,
    )

    # Save generated_text to file as new expected output
    with open("models/demos/t3000/falcon40b/demo/expected_output_data.json", "w") as f:
        json.dump(generated_text, f)


@pytest.mark.parametrize("max_seq_len", (128,))
def test_demo(
    max_seq_len,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
):
    input_file = "models/demos/t3000/falcon40b/demo/input_data.json"
    # Enable async mode
    for device in t3k_mesh_device.get_devices():
        device.enable_async(True)

    generated_text, measurements = run_falcon_demo_kv(
        user_input=input_file,
        model_version=model_config_entries["_name_or_path"],
        model_config_str_for_decode="BFLOAT8_B-SHARDED",  # Decode model config
        model_config_str_for_prefill="BFLOAT8_B-DRAM",  # Prefill model config
        batch_size=32,
        num_layers=model_config_entries["num_hidden_layers"],
        max_seq_len=max_seq_len,
        model_location_generator=model_location_generator,
        get_tt_cache_path=get_tt_cache_path,
        mesh_device=t3k_mesh_device,
        prefill_on_host=False,
        perf_mode=False,
        greedy_sampling=True,
    )

    # Validate generated_text against expected output
    with open("models/demos/t3000/falcon40b/demo/expected_output_data.json", "r") as f:
        expected_output_data = json.load(f)
        assert expected_output_data == generated_text
