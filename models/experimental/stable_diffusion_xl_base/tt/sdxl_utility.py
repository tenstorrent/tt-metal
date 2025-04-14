# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


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
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, G, num_cores)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
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


def prepare_linear_params(device, weights, bias, dtype):
    tt_weights = ttnn.from_torch(torch.permute(weights, (0, 1, 3, 2)), dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(bias, dtype, device=device, layout=ttnn.TILE_LAYOUT) if bias is not None else None
    return tt_weights, tt_bias


def prepare_conv_params(device, weights, bias, dtype, act_dtype=ttnn.bfloat16, act_block_h_override=0):
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=act_dtype,
        weights_dtype=dtype,
        shard_layout=None,
        input_channels_alignment=32,
        deallocate_activation=True,
        reallocate_halo_output=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        reshard_if_not_optimal=True,
        act_block_w_div=1,
        act_block_h_override=act_block_h_override,
        preprocess_weights_on_device=True,
        always_preprocess_weights=True,
        transpose_shards=True,
    )

    dtype = ttnn.float32 if dtype == ttnn.bfloat8_b else dtype
    tt_weights = ttnn.from_torch(weights, dtype)
    tt_bias = ttnn.from_torch(bias, dtype) if bias is not None else None
    return compute_config, conv_config, tt_weights, tt_bias
