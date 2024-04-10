# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import gc
import pytest
from loguru import logger
from pathlib import Path
import torch
from torch import nn
import tt_lib
import ttnn

from models.demos.llama2_70b.reference.llama.llama import Llama
from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, get_devices_for_t3000
from models.demos.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized


def run_cache_model(
    devices,
    batch,
    seq_len,
    model_config,
    n_layers,
    n_devices,
    emulated=False,
    # tt_cache_path,
    # model_location_generator,
):
    # model_name = model_location_generator(model_version, model_subdir="Llama2")
    if emulated:
        ckpt_dir = "/proj_sw/user_dev/llama-data-repacked-2/llama-2-70b/"
        tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"
        cache_path = Path("/proj_sw/user_dev/llama-data-cache/weights-cache")
        device = devices[0]
        devices = [device for _ in range(n_devices)]  # Emulate fracturing on N chips
    else:
        ckpt_dir = model_config["DEFAULT_CKPT_DIR"]
        tokenizer_path = model_config["DEFAULT_TOKENIZER_PATH"]
        cache_path = model_config["DEFAULT_CACHE_PATH"]

    print(f"Running emulated: {emulated}")

    max_seq_len = 4096
    layer_group_size = 4
    n_layers = 80

    for start_layer_idx in range(0, n_layers, layer_group_size):
        hugging_face_reference_model = Llama.build(
            ckpt_dir,
            tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=batch,
            n_layers=layer_group_size,
            skip_model_load=False,
            start_layer_idx=start_layer_idx,
        ).model
        hugging_face_reference_model.eval()
        state_dict = hugging_face_reference_model.state_dict()
        print(state_dict.keys())

        new_state_dict = {
            ".".join([k.split(".")[0], str(start_layer_idx + int(k.split(".")[1]))] + k.split(".")[2:])
            if "layers" in k
            else k: v
            for k, v in state_dict.items()
        }

        print(new_state_dict.keys())

        torch.manual_seed(0)
        base_url = "layers"
        configuration = hugging_face_reference_model.params
        n_heads = configuration.n_heads
        n_kv_heads = configuration.n_kv_heads
        hidden_dim = configuration.dim
        head_dim = hidden_dim // n_heads

        # TT model -------------------------------------------------------------
        # Create TT model which caches weights as it inits
        tt_model = TtLlamaModel_optimized(
            devices,
            state_dict,
            base_url,
            layer_group_size,
            model_config,
            configuration,
            batch,
            cache_path=cache_path,
            emulated=emulated,
            start_layer_idx=start_layer_idx,
        )

        for device in devices:
            tt_lib.device.Synchronize(device)

        # Free up all space on device
        tt_model.free_layers(start_layer_idx, layer_group_size)
        del tt_model

        gc.collect()


@pytest.mark.parametrize(
    "n_layers",
    ((4, 10, 20)),
)
@pytest.mark.parametrize(
    "batch, seq_len, n_devices, emulated",
    (
        (32, 1, 4, False),
        (32, 1, 8, False),
        (32, 1, 4, True),
        (32, 1, 8, True),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_cache_model(
    batch,
    seq_len,
    model_config_str,
    n_layers,
    n_devices,
    all_devices,
    emulated,
):
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices if not emulated else 1)
    model_config = get_model_config(model_config_str, num_devices=n_devices)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if len(devices) < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_cache_model(
        devices,
        batch,
        seq_len,
        model_config,
        n_layers,
        n_devices,
        emulated,
    )
