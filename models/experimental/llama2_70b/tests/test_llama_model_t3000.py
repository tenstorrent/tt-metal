# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.llama2_70b.tt.model_config import get_model_config
from models.utility_functions import skip_for_grayskull
from models.experimental.llama2_70b.tests.test_llama_model import run_test_LlamaModel_inference

import os


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "pcc, n_layers",
    (
        (0.996, 1),
        (0.996, 2),
    ),
    ids=("1L", "2L"),
)
@pytest.mark.parametrize(
    "batch, seq_len",
    ((1, 128), (32, 1), (1, 2048)),
    ids=("prefill_128", "decode", "prefill_2k"),
)
def test_LlamaModel_inference(
    batch,
    seq_len,
    pcc,
    n_layers,
    t3k_device_mesh,
    n_devices=8,
):
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)

    if t3k_device_mesh.get_num_devices() < n_devices:
        pytest.skip(f"Requires at {n_devices} devices to run")

    compute_grid_size = t3k_device_mesh.get_device(0).compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    for i in t3k_device_mesh.get_device_ids():
        device = t3k_device_mesh.get_device(i)
        device.enable_program_cache()

    run_test_LlamaModel_inference(
        t3k_device_mesh,
        batch,
        seq_len,
        pcc,
        model_config,
        n_layers,
        n_devices,
    )
