# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.llama2_70b.tt.model_config import get_model_config
from models.utility_functions import get_devices_for_t3000, skip_for_grayskull
from models.experimental.llama2_70b.tests.test_llama_attention import run_test_LlamaAttention_inference


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    ((32, 1, 0.9997), (1, 128, 0.9997), (1, 2048, 0.9997)),
    ids=("decode", "prefill_128", "prefill_2k"),
)
def test_LlamaAttention_inference_t3000(
    batch,
    seq_len,
    pcc,
    all_devices,
    use_program_cache,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices)
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if len(all_devices) < n_devices:
        pytest.skip(f"Requires at least {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_test_LlamaAttention_inference(
        devices,
        batch,
        seq_len,
        pcc,
        model_config,
        n_devices,
    )
