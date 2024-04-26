# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.t3000.falcon40b.tests.test_falcon_end_to_end import run_test_FalconCausalLM_end_to_end
from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)
from models.utility_functions import (
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    skip_for_grayskull,
    get_devices_for_t3000,
)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("prefill", 1, 2048, 0),
    ),
    ids=[
        "prefill_seq128",
        "prefill_seq2048",
    ],
)
@pytest.mark.parametrize(
    "num_layers, out_pcc, cache_pcc, token_pcc",
    ((1, 0.99, 0.99, 0.99),),
    ids=["layers_1"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT8_B-DRAM", "BFLOAT16-DRAM"))
def test_FalconCausalLM_prefill_end_to_end_t3000_ci(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    out_pcc,
    cache_pcc,
    token_pcc,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    all_devices,
    use_program_cache,
):
    num_devices = 8
    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = get_devices_for_t3000(all_devices, num_devices)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        devices,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        out_pcc,
        cache_pcc,
        token_pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
