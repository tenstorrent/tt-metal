# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
    tt_weights = ttnn.from_torch(weights.movedim(-1, -2), dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(bias, dtype, device=device, layout=ttnn.TILE_LAYOUT) if bias is not None else None
    return tt_weights, tt_bias


def prepare_conv_params(
    device,
    weights,
    bias,
    dtype,
    fp32_dest_acc_en=False,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    packer_l1_acc=False,
):
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    dtype = ttnn.float32 if dtype == ttnn.bfloat8_b else dtype
    tt_weights = ttnn.from_torch(weights, dtype)
    tt_bias = ttnn.from_torch(bias, dtype) if bias is not None else None

    conv_params = {
        "input_channels": tt_weights.shape[1],
        "output_channels": tt_weights.shape[0],
        "kernel_size": (tt_weights.shape[2], tt_weights.shape[3]),
    }

    return compute_config, tt_weights, tt_bias, conv_params


def prepare_split_conv_params(
    device,
    weights,
    bias,
    dtype,
    split_in,
    split_out,
    fp32_dest_acc_en=False,
    math_fidelity=ttnn.MathFidelity.HiFi2,
):
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )

    dtype = ttnn.float32 if dtype == ttnn.bfloat8_b else dtype  # TODO: figure out why PCC drops when dtype is used

    Cout, Cin, _, _ = weights.shape
    Cout_split = Cout // split_out
    Cin_split = Cin // split_in

    if split_out > 1:
        weight_chunks = list(torch.split(weights, Cout_split, 0))
    else:
        weight_chunks = [weights]

    for i in range(len(weight_chunks)):
        weight_chunks[i] = torch.split(weight_chunks[i], Cin_split, 1)

    tt_weights = [
        [
            ttnn.from_torch(
                weight,
                dtype=ttnn.float32,
            )
            for weight in weights_Cout_split
        ]
        for weights_Cout_split in weight_chunks
    ]

    if bias is not None:
        if split_out > 1:
            bias_chunks = list(torch.split(bias, Cout_split, 3))
        else:
            bias_chunks = [bias]

        tt_bias = [
            ttnn.from_torch(
                bias,
                dtype=ttnn.float32,
            )
            for bias in bias_chunks
        ]
    else:
        tt_bias = [None]

    conv_params = [
        [
            {
                "input_channels": tt_w.shape[1],
                "output_channels": tt_w.shape[0],
                "kernel_size": (tt_w.shape[2], tt_w.shape[3]),
            }
            for tt_w in tt_w_out
        ]
        for tt_w_out in tt_weights
    ]
    return compute_config, tt_weights, tt_bias, conv_params


def split_conv2d(
    device,
    hidden_states,
    input_shape,
    conv_weights,
    conv_bias,
    split_in,
    split_out,
    compute_config,
    conv_config,
    conv_params,
    conv_dtype,
    stride,
    padding,
    dilation,
    groups,
):
    assert hidden_states.layout == ttnn.ROW_MAJOR_LAYOUT, "Input tensor must be in ROW_MAJOR layout"

    B, C, H, W = input_shape

    if split_in > 1:
        hidden_states_split = ttnn.split(hidden_states, C // split_in, 3)
        hidden_states.deallocate(True)
    else:
        hidden_states_split = [hidden_states]

    outputs = []
    device_weights = []
    device_bias = []

    for idx_out in range(split_out):
        device_weights.append([])

        for idx_in in range(split_in):
            [intermediate, [Hout, Wout], [d_w, d_b]] = ttnn.conv2d(
                input_tensor=hidden_states_split[idx_in],
                weight_tensor=conv_weights[idx_out][idx_in],
                in_channels=conv_params[idx_out][idx_in]["input_channels"],
                out_channels=conv_params[idx_out][idx_in]["output_channels"],
                device=device,
                bias_tensor=conv_bias[idx_out],
                kernel_size=conv_params[idx_out][idx_in]["kernel_size"],
                stride=stride,
                padding=padding,
                dilation=dilation,
                batch_size=B,
                input_height=H,
                input_width=W,
                conv_config=conv_config,
                compute_config=compute_config,
                groups=groups,
                memory_config=None,
                return_output_dim=True,
                return_weights_and_bias=True,
                dtype=conv_dtype,
            )

            device_weights[idx_out].append(d_w)
            if idx_in == 0:
                device_bias.append(d_b)

            if idx_in == 0:
                dram_intermediate = ttnn.to_memory_config(intermediate, ttnn.DRAM_MEMORY_CONFIG)
                intermediate.deallocate(True)
            else:
                dram_intermediate = ttnn.add(
                    dram_intermediate, intermediate, output_tensor=dram_intermediate, use_legacy=False
                )
                intermediate.deallocate(True)

        if dram_intermediate.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            dram_intermediate = ttnn.to_memory_config(dram_intermediate, ttnn.DRAM_MEMORY_CONFIG)
        outputs.append(dram_intermediate)
    H, W = Hout, Wout
    C = conv_params[0][0]["output_channels"] * split_out

    if len(outputs) > 1:
        output = ttnn.concat(outputs, dim=-1)
        for output_slice in outputs:
            output_slice.deallocate(True)
    else:
        output = outputs[0]

    return output, [C, H, W], [device_weights, device_bias]
