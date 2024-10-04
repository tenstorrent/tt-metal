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
)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_loops",
    (10,),
    ids=[
        "loops_10",
    ],
)
@pytest.mark.parametrize(
    "seq_len",
    (
        32,
        128,
        2048,
    ),
    ids=[
        "seq32",
        "seq128",
        "seq2048",
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    (60,),
    ids=["layers_60"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize(
    "data_type, memcfg",
    (
        (
            "BFLOAT8_B",
            "DRAM",
        ),
        (
            "BFLOAT16",
            "DRAM",
        ),
    ),
)
@pytest.mark.parametrize(
    "async_mode",
    (True,),
)
def test_FalconCausalLM_prefill_end_to_end_t3000_ci_loops_10(
    model_version,
    seq_len,
    num_layers,
    data_type,
    memcfg,
    num_loops,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
    async_mode,
):
    num_devices = 8
    llm_mode = "prefill"
    batch = 1
    kv_cache_len = 0
    out_pcc = 0.99
    k_cache_pcc = 0.99
    v_cache_pcc = 0.97
    token_pcc = 0.99

    input_shape = [batch, seq_len]
    model_config_str = f"{data_type}-{memcfg}"
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    t3k_mesh_device.enable_async(async_mode)
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    assert llm_mode == "prefill" and num_layers == 60, "PCC tresholds only valid for prefill and 60 layers."

    if data_type == "BFLOAT8_B":
        if seq_len == 32:
            out_pcc = 0.986
            k_cache_pcc = 0.978
            v_cache_pcc = 0.934
            token_pcc = 0.99
        elif seq_len == 128:
            out_pcc = 0.990
            k_cache_pcc = 0.988
            v_cache_pcc = 0.940
            token_pcc = 0.99
        elif seq_len == 2048:
            out_pcc = 0.992
            k_cache_pcc = 0.990
            v_cache_pcc = 0.967
            token_pcc = 0.99
    elif data_type == "BFLOAT16":
        if seq_len == 32:
            out_pcc = 0.981
            k_cache_pcc = 0.978
            v_cache_pcc = 0.929
            token_pcc = 0.99
        elif seq_len == 128:
            out_pcc = 0.991
            k_cache_pcc = 0.993
            v_cache_pcc = 0.976
            token_pcc = 0.99
        elif seq_len == 2048:
            out_pcc = 0.992
            k_cache_pcc = 0.989
            v_cache_pcc = 0.972
            token_pcc = 0.99

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        t3k_mesh_device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        out_pcc,
        k_cache_pcc,
        v_cache_pcc,
        token_pcc,
        model_config,
        num_loops,
        tt_cache_path,
        model_location_generator,
    )
