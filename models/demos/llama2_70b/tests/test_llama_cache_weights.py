# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

from pathlib import Path
import scipy
from sklearn.metrics import top_k_accuracy_score
import numpy as np

import tt_lib

from models.demos.llama2_70b.reference.llama import Llama

from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.demos.llama2_70b.tt.llama_model import TtLlamaModel
from models.demos.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized


def run_cache_model(
    devices,
    batch,
    seq_len,
    pcc,
    model_config,
    optimized,
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
        device = devices[0]
        devices = [device for _ in range(n_devices)]  # Emulate fracturing on N chips
    else:
        ckpt_dir = "/home/llama-data-repacked-2/llama-2-70b/"
        tokenizer_path = "/home/llama-data/tokenizer.model"

    max_seq_len = 4096
    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=batch,
        n_layers=n_layers,
        skip_model_load=False,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    print(state_dict.keys())

    torch.manual_seed(0)
    base_url = "layers"
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    CACHE_PATH = Path("/home/llama-data-cache/weights-cache")
    # TT model -------------------------------------------------------------
    tt_model = TtLlamaModel_optimized(
        devices, state_dict, base_url, n_layers, model_config, configuration, batch, cache_path=CACHE_PATH
    )


@pytest.mark.parametrize(
    "batch, seq_len, n_layers, n_devices",
    ((32, 1, 4, 4),),
)
@pytest.mark.parametrize(
    "model_version, pcc, optimized",
    (("llama-2-70B", 0.98, True),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_cache_model(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    optimized,
    n_layers,
    n_devices,
    # model_location_generator,
    pcie_devices,
):
    model_config = get_model_config(model_config_str, num_devices=n_devices)
    # tt_cache_path = get_tt_cache_path(model_version)
    compute_grid_size = pcie_devices[0].compute_with_storage_grid_size()
    if len(pcie_devices) < n_devices:
        pytest.skip(f"Requires at {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_cache_model(
        pcie_devices[:n_devices],
        batch,
        seq_len,
        pcc,
        model_config,
        optimized,
        n_layers,
        n_devices
        # tt_cache_path,
        # model_location_generator,
    )
