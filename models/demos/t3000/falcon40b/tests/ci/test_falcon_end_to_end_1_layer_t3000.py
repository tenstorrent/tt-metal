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
@pytest.mark.parametrize("num_devices", (8,), ids=["8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 32, 0),
        ("prefill", 2, 32, 0),
        ("prefill", 1, 128, 0),
        ("prefill", 1, 2048, 0),
        ("decode", 32, 1, 128),
    ),
    ids=[
        "prefill_seq32",
        "prefill_seq32_batch2",
        "prefill_seq128",
        "prefill_seq2048",
        "decode_batch32",
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    (1,),
    ids=[
        "layers_1",
    ],
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
            "SHARDED",
        ),
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
def test_FalconCausalLM_end_to_end_with_program_cache(
    num_devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    request,
    data_type,
    memcfg,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
    async_mode,
):
    model_config_str = f"{data_type}-{memcfg}"
    if llm_mode == "prefill" and memcfg != "DRAM" or num_devices != 8:
        pytest.skip("Prefill is only supported for DRAM memory config and 8 chips!")
    if llm_mode == "decode" and memcfg != "SHARDED":
        pytest.skip("Decode is only supported for SHARDED memory config!")

    out_pcc = 0.99
    k_cache_pcc = 0.99
    v_cache_pcc = 0.99
    token_pcc = 0.99

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = t3k_mesh_device.get_devices()
    for device in devices:
        device.enable_async(async_mode)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

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
        1,
        tt_cache_path,
        model_location_generator,
    )
