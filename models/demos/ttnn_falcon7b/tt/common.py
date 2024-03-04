# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
import ttnn
import torch
import models
from pathlib import Path
from loguru import logger


def from_torch_cached(
    filename,
    torch_tensor,
    device=None,
    dtype=None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    layout=ttnn.TILE_LAYOUT,
    unsqueeze_to_4d=False,
):
    filename = f"{filename}_{dtype.name}.bin"
    try:
        tensor = ttnn.load_tensor(filename)
        if tuple(tensor.shape) != tuple(torch_tensor.shape):
            logger.warning(
                f"Cached file {filename} has shape {tensor.shape}, expected {torch_tensor.shape}, regenerating cache"
            )
            raise RuntimeError
        logger.info(f"Loaded cache for {filename} of shape {tensor.shape}")
    except (FileNotFoundError, RuntimeError):
        tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)
        logger.info(f"Generating cache for {filename} of shape {tensor.shape}")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        ttnn.dump_tensor(filename, tensor)
    if unsqueeze_to_4d:
        tensor = ttnn.unsqueeze_to_4D(tensor)
    tensor = ttnn.to_device(tensor, device, memory_config=memory_config)
    return tensor


def create_custom_preprocessor(model_config, tt_cache_path, device, base_file_name=None):
    def rotary_embedding_custom_processor(torch_model, name):
        parameters = {}
        if base_file_name:
            base_file_path = f"{tt_cache_path}/{base_file_name}.{name}"
        else:
            base_file_path = f"{tt_cache_path}/{name}"

        if isinstance(torch_model, transformers.models.falcon.modeling_falcon.FalconRotaryEmbedding):
            parameters["cos_cached"] = from_torch_cached(
                f"{base_file_path}.cos_cached",
                torch_model.cos_cached,
                device=device,
                dtype=model_config["COS_CACHED_WEIGHTS_DTYPE"],
                unsqueeze_to_4d=True,
            )
            parameters["sin_cached"] = from_torch_cached(
                f"{base_file_path}.sin_cached",
                torch_model.sin_cached,
                device=device,
                dtype=model_config["SIN_CACHED_WEIGHTS_DTYPE"],
                unsqueeze_to_4d=True,
            )
        elif isinstance(torch_model, torch.nn.Linear):
            linear_weight_file_name = f"{base_file_path}.weight"
            parameters["weight"] = from_torch_cached(
                linear_weight_file_name, torch_model.weight.T.contiguous(), device=device, dtype=ttnn.bfloat8_b
            )
            if torch_model.bias is not None:
                linear_bias_file_name = f"{base_file_path}.bias"
                parameters["bias"] = from_torch_cached(
                    linear_bias_file_name, torch_model.bias.reshape((1, -1)).contiguous(), device=device
                )
        elif isinstance(torch_model, torch.nn.LayerNorm):
            parameters["weight"] = from_torch_cached(
                f"{base_file_path}.weight", torch_model.weight.reshape((1, -1)), device=device, dtype=ttnn.bfloat16
            )
            parameters["bias"] = from_torch_cached(
                f"{base_file_path}.bias", torch_model.bias.reshape((1, -1)), device=device, dtype=ttnn.bfloat16
            )

        return parameters

    return rotary_embedding_custom_processor
