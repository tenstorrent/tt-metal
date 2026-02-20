# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import time


def to_channel_last_ttnn(torch_tensor, dtype, device, memory_config, layout):
    torch_tensor = torch.permute(torch_tensor, (0, 2, 3, 1))
    ttnn_tensor = ttnn.from_torch(torch_tensor, dtype, device=device, memory_config=memory_config, layout=layout)
    return ttnn_tensor


def from_channel_last_ttnn(ttnn_tensor, output_shape):
    torch_tensor = ttnn.to_torch(ttnn_tensor)
    torch_tensor = torch_tensor.reshape(output_shape)
    torch_tensor = torch.permute(torch_tensor, (0, 3, 1, 2))
    return torch_tensor


def prepare_gn_mask(device, C, G, num_cores):
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, G, num_cores, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)
    return input_mask_tensor


def prepare_gn_mask_negative_mask(device, C, G, num_cores):
    input_mask_tensor = ttnn.create_group_norm_input_negative_mask(C, G, num_cores, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)
    return input_mask_tensor


def prepare_gn_beta_gamma(device, weights, bias, num_cores):
    gamma = ttnn.create_group_norm_weight_bias_rm(weights, weights.shape[0], num_cores)
    beta = ttnn.create_group_norm_weight_bias_rm(bias, bias.shape[0], num_cores)
    tt_gamma = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_gamma, tt_bias


def prepare_linear_params(device, weights, bias, dtype, is_lora_impacted=False):
    host_creation_time_ms = None
    host_to_device_time_ms = None

    if is_lora_impacted:
        # TIMING: Host tensor creation for weights (LoRA-impacted only)
        weights_bf16 = weights.to(torch.bfloat16)
        weights_bf16 = weights_bf16.movedim(-1, -2)
        ttnn.synchronize_device(device)
        start_time = time.perf_counter()
        tt_weights_host = ttnn.from_torch(weights_bf16, dtype)
        # tt_weights_host = ttnn.from_torch(weights.movedim(-1, -2), dtype, layout=ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(device)
        host_creation_time_ms = (time.perf_counter() - start_time) * 1000

        tt_weights_host = ttnn.to_layout(tt_weights_host, ttnn.TILE_LAYOUT)

        tt_weights_device = ttnn.allocate_tensor_on_device(tt_weights_host.spec, device)

        # TIMING: Host-to-device copy for weights (LoRA-impacted only)
        ttnn.synchronize_device(device)
        start_time = time.perf_counter()
        ttnn.copy_host_to_device_tensor(tt_weights_host, tt_weights_device)
        ttnn.synchronize_device(device)
        host_to_device_time_ms = (time.perf_counter() - start_time) * 1000
    else:
        # No timing for non-LoRA operations
        tt_weights_host = ttnn.from_torch(weights.movedim(-1, -2), dtype, layout=ttnn.TILE_LAYOUT)
        tt_weights_device = ttnn.allocate_tensor_on_device(tt_weights_host.spec, device)
        ttnn.copy_host_to_device_tensor(tt_weights_host, tt_weights_device)

    # Handle bias (not LoRA-impacted - no timing needed)
    tt_bias = ttnn.from_torch(bias, dtype, device=device, layout=ttnn.TILE_LAYOUT) if bias is not None else None

    return tt_weights_device, tt_bias, host_creation_time_ms, host_to_device_time_ms


def prepare_conv_params(
    weights,
    bias,
    dtype,
):
    dtype = ttnn.float32 if dtype == ttnn.bfloat8_b else dtype
    tt_weights = ttnn.from_torch(weights, dtype)
    tt_bias = ttnn.from_torch(bias, dtype) if bias is not None else None

    conv_params = {
        "input_channels": tt_weights.shape[1],
        "output_channels": tt_weights.shape[0],
        "kernel_size": (tt_weights.shape[2], tt_weights.shape[3]),
    }

    return tt_weights, tt_bias, conv_params
