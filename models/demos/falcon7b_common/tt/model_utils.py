# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ReplicateTensorToMesh

from models.utility_functions import is_wormhole_b0


def get_weights_cached(
    mesh_device,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    tt_layout=ttnn.TILE_LAYOUT,
    weights_dict=None,
    custom_output_shape=None,
):
    """Load weights from weights_dict or cache and duplicate per device. Store if not cached."""
    custom_output_shape_str = ""
    if custom_output_shape is not None:
        custom_output_shape_str = f"_{custom_output_shape[-2]}_{custom_output_shape[-1]}"
    path = tt_cache_path / f"{weight_cache_str}{custom_output_shape_str}"

    def preprocess_weights(weights_to_cache):
        if weights_to_cache is None:
            raise ValueError(f"weights_to_cache is None for {weight_cache_str}")

        if custom_output_shape is not None:
            padding = (
                0,
                custom_output_shape[-1] - weights_to_cache.shape[-1],
                0,
                custom_output_shape[-2] - weights_to_cache.shape[-2],
            )
            weights_to_cache = torch.nn.functional.pad(weights_to_cache, padding, "constant", 0.0)

        return weights_to_cache

    if weights_dict and str(path) in weights_dict.keys():
        weights = weights_dict[str(path)]
    else:
        weights = ttnn.as_tensor(
            weights_to_cache,
            dtype=model_config[f"{weight_config_str}_DTYPE"],
            layout=tt_layout,
            device=mesh_device,
            memory_config=model_config[f"{weight_config_str}_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(mesh_device) if type(mesh_device) == ttnn.MeshDevice else None,
            cache_file_name=str(path),
            preprocess=preprocess_weights,
        )

        # Save weights for reuse between prefill/decode
        if weights_dict is not None:
            weights_dict[str(path)] = weights

    return weights


# TODO: Remove this once there are no more hangs on 8x8 (Issue #6795)
def get_falcon_default_core_grid(device):
    grid_size = device.compute_with_storage_grid_size()
    if is_wormhole_b0() and grid_size.y >= 8:
        return ttnn.CoreGrid(y=7, x=grid_size.x)
    return ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)


def layernorm(ln_input, ln_eps, ln_gamma, ln_betta, model_config):
    h_dim = ln_input.shape.with_tile_padding()[-2]  # corresponds to batch size (decode) or seq_len (prefill)
    if h_dim in [32, 128, 256, 1024, 2048]:
        ln_output = ttnn.interleaved_to_sharded(ln_input, model_config["LAYERNORM_BLOCK_SHARDED_MEM_CFG"][h_dim])
        ln_output = ttnn.layer_norm(
            ln_output,
            epsilon=ln_eps,
            weight=ln_gamma,
            bias=ln_betta,
            memory_config=model_config["LAYERNORM_BLOCK_SHARDED_MEM_CFG"][h_dim],
            program_config=model_config["LAYERNORM_BLOCK_SHARDED_PROG_CFG"][h_dim],
            compute_kernel_config=model_config["LAYERNORM_BLOCK_SHARDED_COMPUTE_KERNEL_CONFIG"][h_dim],
        )
        ln_output = ttnn.sharded_to_interleaved(ln_output)
    else:
        ln_output = ttnn.layer_norm(
            ln_input,
            epsilon=ln_eps,
            memory_config=model_config["LN_F_OUTPUT_MEMCFG"],
        )
        ln_output = ttnn.multiply(ln_output, ln_gamma, memory_config=model_config["LN_F_OUTPUT_MEMCFG"])
        ln_output = ttnn.add(
            ln_output,
            ln_betta,
            memory_config=model_config["LN_F_OUTPUT_MEMCFG"],
        )
    return ln_output
