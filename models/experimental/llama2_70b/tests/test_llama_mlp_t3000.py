# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.llama2_70b.tt.model_config import get_model_config
from models.utility_functions import get_devices_for_t3000, skip_for_grayskull
from models.experimental.llama2_70b.tests.test_llama_mlp import run_test_LlamaMLP_inference

import os

# Set Llama flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "llama_version",
    (
        ("llama2"),
        # ("llama3"),
    ),
)
@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    ((32, 1, 0.9999), (1, 128, 0.9997), (1, 2048, 0.9997)),
    ids=("decode", "prefill_128", "prefill_2k"),
)
def test_LlamaMLP_inference_t3000(
    batch,
    seq_len,
    pcc,
    t3k_device_mesh,
    llama_version,
    n_devices=8,
):
    # Set Llama flags for CI, if CI environment is setup
    if os.getenv("CI") == "true":
        if llama_version == "llama3":
            os.environ["LLAMA_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/llama-3/llama-3-70b-repacked/"
            os.environ["LLAMA_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/llama-3/tokenizer.model"
            os.environ["LLAMA_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/llama-3/llama-data-cache/weights-cache-3"
        else:
            os.environ["LLAMA_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/llama-2/llama-2-70b-repacked/"
            os.environ["LLAMA_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/llama-2/tokenizer.model"
            os.environ["LLAMA_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/llama-2/llama-data-cache/weights-cache-2"
    # For local testing
    else:
        if llama_version == "llama3":
            os.environ["LLAMA_CKPT_DIR"] = "/home/llama3-data-repacked/llama-3-70b/"
            os.environ["LLAMA_TOKENIZER_PATH"] = "/home/llama3-data/Meta-Llama-3-70B/tokenizer.model"
            os.environ["LLAMA_CACHE_PATH"] = "/home/llama3-data-cache/weights-cache"
        else:
            os.environ["LLAMA_CKPT_DIR"] = "/home/llama-data-repacked-2/llama-2-70b/"
            os.environ["LLAMA_TOKENIZER_PATH"] = "/home/llama-data/tokenizer.model"
            os.environ["LLAMA_CACHE_PATH"] = "/home/llama-data-cache/weights-cache-2"

    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)

    if t3k_device_mesh.get_num_devices() < n_devices:
        pytest.skip(f"Requires at {n_devices} devices to run")

    compute_grid_size = t3k_device_mesh.get_device(0).compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    for i in t3k_device_mesh.get_device_ids():
        device = t3k_device_mesh.get_device(i)
        device.enable_program_cache()

    run_test_LlamaMLP_inference(
        t3k_device_mesh,
        batch,
        seq_len,
        pcc,
        model_config,
        n_devices,
    )
