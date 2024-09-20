# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.utility_functions import tt2torch_tensor
import torch
import ttnn
from transformers import GPT2LMHeadModel
from tt_lib.utils import pad_weight
from pathlib import Path
import os


def unpad_from_zero(x, desired_shape):
    if x.get_legacy_shape()[-1] == desired_shape[-1] and x.get_legacy_shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = x.to(ttnn.ROW_MAJOR_LAYOUT)
        x = x.unpad((0, 0, 0, 0), (desired_shape[0], desired_shape[1], desired_shape[2], desired_shape[3]))
        x = x.to_torch().to(torch.float)
    return x


def cache_weights_in_weka(device, dtype, reset_seeds):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    state_dict = model_hf.state_dict()
    weights_dtype = dtype

    # initial weights are stored in "models/experimental/nanogpt/weights/" and moved to weka path
    file_name = "models/experimental/nanogpt/weights/"
    for key, value in state_dict.items():
        if key.startswith("transformer.wte.") or key.startswith("transformer.wpe."):
            torch.save(value, file_name + str(key) + ".pt")
            continue
        elif len(value.shape) == 0:
            continue
        while len(value.shape) < 4:
            value = value.unsqueeze(0)
        if value.shape[-2] % 32 == 0 and value.shape[-1] % 32 == 0:
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                weights_dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
        else:
            value = pad_weight(value)
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                weights_dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(file_name + str(key) + str(weights_dtype) + ".bin", value)


"""This function will load weights from the state_dict and check if the needed weights are available in given path.
If they are not available, it will convert torch tensor weights to TT tensor weights and store them in the given path."""


def store_weights(model_version, file_name, base_address, dtype):
    model_hf = GPT2LMHeadModel.from_pretrained(model_version)
    state_dict = model_hf.state_dict()
    weights_dtype = dtype

    for key, value in state_dict.items():
        if base_address == "" and (
            (key.startswith("transformer.wte.") and os.path.exists(file_name + str(key) + ".pt") == False)
            or (key.startswith("transformer.wpe.") and os.path.exists(file_name + str(key) + ".pt") == False)
        ):
            torch.save(value, file_name + str(key) + ".pt")
            continue
        if key.startswith("transformer.wte.") or key.startswith("transformer.wpe.") or (len(value.shape) == 0):
            continue
        if (os.path.exists(file_name + str(key) + str(weights_dtype) + ".bin")) or (
            key.startswith(base_address) == False and base_address != ""
        ):
            continue
        while len(value.shape) < 4:
            value = value.unsqueeze(0)
        if value.shape[-2] % 32 == 0 and value.shape[-1] % 32 == 0:
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                weights_dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
        else:
            value = pad_weight(value)
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                weights_dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(file_name + str(key) + str(weights_dtype) + ".bin", value)


def get_tt_cache_path(model_version):
    tt_cache_path = Path("/mnt/MLPerf/tt_dnn-models/tt/NanoGPT") / model_version
    if tt_cache_path.exists():
        return str(tt_cache_path) + "/"
    else:
        Path(f"models/experimental/nanogpt/datasets/{model_version}").mkdir(parents=True, exist_ok=True)
        return str(Path(f"models/experimental/nanogpt/datasets/{model_version}")) + "/"
