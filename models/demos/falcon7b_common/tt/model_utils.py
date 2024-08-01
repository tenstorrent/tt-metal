# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ReplicateTensorToMesh

from models.utility_functions import is_wormhole_b0, torch2tt_tensor, pad_by_zero


def get_weights_cached(
    device_mesh,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    padzero=False,
    tt_layout=ttnn.TILE_LAYOUT,
    weights_dict=None,
    custom_output_shape=None,
):
    if padzero:
        assert tt_layout == ttnn.TILE_LAYOUT, "padding by zero currently only uses TILE layout"

    """Load weights from weights_dict or cache and duplicate per device. Store if not cached."""
    custom_output_shape_str = ""
    if custom_output_shape is not None:
        custom_output_shape_str = f"_{custom_output_shape[-2]}_{custom_output_shape[-1]}"
    path = tt_cache_path / f"{weight_cache_str}{custom_output_shape_str}"

    def preprocess_weights(weights_to_cache):
        if weights_to_cache is None:
            raise ValueError(f"weights_to_cache is None for {weight_cache_str}")

        if padzero:
            weights_host = pad_by_zero(
                weights_to_cache,
                device=None,
                tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
            )[0]
        else:
            if custom_output_shape is not None:
                padding = (
                    0,
                    custom_output_shape[-1] - weights_to_cache.shape[-1],
                    0,
                    custom_output_shape[-2] - weights_to_cache.shape[-2],
                )
                weights_to_cache = torch.nn.functional.pad(weights_to_cache, padding, "constant", 0.0)

            weights_host = weights_to_cache
        return weights_host

    if weights_dict and str(path) in weights_dict.keys():
        weights = weights_dict[str(path)]
    else:
        weights = ttnn.as_tensor(
            weights_to_cache,
            dtype=model_config[f"{weight_config_str}_DTYPE"],
            layout=tt_layout,
            device=device_mesh,
            memory_config=model_config[f"{weight_config_str}_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
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


def layernorm(ln_input, ln_eps, ln_gamma, ln_betta, num_devices, model_config):
    h_dim = ln_input[0].get_legacy_shape()[-2]  # corresponds to batch size (decode) or seq_len (prefill)
    ln_output = []
    if h_dim in [32, 128, 256, 1024, 2048]:
        for i in range(num_devices):
            ln_output.append(
                ttnn.experimental.tensor.interleaved_to_sharded(
                    ln_input[i], sharded_mem_config=model_config["LAYERNORM_BLOCK_SHARDED_MEM_CFG"][h_dim]
                )
            )
        for i in range(num_devices):
            ln_output[i] = ttnn.layer_norm(
                ln_output[i],
                epsilon=ln_eps,
                weight=ln_gamma[i],
                bias=ln_betta[i],
                memory_config=model_config["LAYERNORM_BLOCK_SHARDED_MEM_CFG"][h_dim],
                program_config=model_config["LAYERNORM_BLOCK_SHARDED_PROG_CFG"][h_dim],
                compute_kernel_config=model_config["LAYERNORM_BLOCK_SHARDED_COMPUTE_KERNEL_CONFIG"][h_dim],
            )
        for i in range(num_devices):
            ln_output[i] = ttnn.experimental.tensor.sharded_to_interleaved(ln_output[i])
    else:
        for i in range(num_devices):
            ln_output.append(
                ttnn.layer_norm(
                    ln_input[i],
                    epsilon=ln_eps,
                    memory_config=model_config["LN_F_OUTPUT_MEMCFG"],
                )
            )
        for i in range(num_devices):
            ln_output[i] = ttnn.experimental.tensor.bcast(
                ln_output[i],
                ln_gamma[i],
                ttnn.experimental.tensor.BcastOpMath.MUL,
                ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config=model_config["LN_F_OUTPUT_MEMCFG"],
            )
        for i in range(num_devices):
            ln_output[i] = ttnn.experimental.tensor.bcast(
                ln_output[i],
                ln_betta[i],
                ttnn.experimental.tensor.BcastOpMath.ADD,
                ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config=model_config["LN_F_OUTPUT_MEMCFG"],
            )
    return ln_output
