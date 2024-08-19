# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.utility_functions import is_wormhole_b0, torch2tt_tensor, pad_by_zero


def get_weights_cached(
    devices,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    overwrite=False,
    padzero=False,
    tt_layout=ttnn.experimental.tensor.Layout.TILE,
    weights_dict=None,
    custom_output_shape=None,
):
    if padzero:
        assert tt_layout == ttnn.experimental.tensor.Layout.TILE, "padding by zero currently only uses TILE layout"

    """Load weights from weights_dict or cache and duplicate per device. Store if not cached."""
    custom_output_shape_str = ""
    if custom_output_shape is not None:
        custom_output_shape_str = f"_{custom_output_shape[-2]}_{custom_output_shape[-1]}"
    path = (
        tt_cache_path
        / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}{custom_output_shape_str}.bin"
    )

    if weights_dict and str(path) in weights_dict.keys():
        weights = weights_dict[str(path)]
    elif not overwrite and path.exists():
        # Load cached weights
        weights_host = ttnn.load_tensor(str(path))
        # Duplicate weights on all devices
        weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]
        # Add to weights_dict
        if weights_dict is not None:
            weights_dict[str(path)] = weights
    else:
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

            weights_host = torch2tt_tensor(
                weights_to_cache,
                tt_device=None,
                tt_layout=tt_layout,
                tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
            )

        weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]
        # Save weights for reuse between prefill/decode
        if weights_dict is not None:
            weights_dict[str(path)] = weights
        # Store weights
        ttnn.experimental.tensor.dump_tensor(str(path), weights_host)

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
            ln_output[i] = ttnn.multiply(ln_output[i], ln_gamma[i], memory_config=model_config["LN_F_OUTPUT_MEMCFG"])
        for i in range(num_devices):
            ln_output[i] = ttnn.add(
                ln_output[i],
                ln_betta[i],
                memory_config=model_config["LN_F_OUTPUT_MEMCFG"],
            )
    return ln_output
