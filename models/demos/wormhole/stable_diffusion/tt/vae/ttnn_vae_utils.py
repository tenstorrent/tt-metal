# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def prepare_split_conv_weights_bias(
    in_channels,
    out_channels,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    torch_weight_tensor,
    torch_bias_tensor,
):
    split_output_channels = out_channels // conv_out_channel_split_factor
    split_input_channels = in_channels // conv_in_channel_split_factor

    # Split weights
    if conv_out_channel_split_factor > 1:
        split_weight_tensors = list(torch.split(torch_weight_tensor, split_output_channels, 0))
    else:
        split_weight_tensors = [torch_weight_tensor]

    for i in range(len(split_weight_tensors)):
        split_weight_tensors[i] = torch.split(split_weight_tensors[i], split_input_channels, 1)

    ttnn_split_weights = [
        [
            ttnn.from_torch(
                weight,
                dtype=ttnn.float32,
            )
            for weight in output_channel_spit_weights
        ]
        for output_channel_spit_weights in split_weight_tensors
    ]

    # Split bias
    if conv_out_channel_split_factor > 1:
        split_bias_tensors = list(torch.split(torch_bias_tensor, split_output_channels, 3))
    else:
        split_bias_tensors = [torch_bias_tensor]

    ttnn_split_bias = [
        ttnn.from_torch(
            bias,
            dtype=ttnn.float32,
        )
        for bias in split_bias_tensors
    ]

    return ttnn_split_weights, ttnn_split_bias


# --- Input channel split
# When splitting by input channels, we split input tensor and weights tensors on the input channel dimension.
# We call conv op on each input and weight slice and accumulate the result in a DRAM tensor.
# --- Ouptut channel split
# When splitting by output channels, we split conv weights and bias tensors.
# We call conv op on the whole input tensor and each output slice of weights and bias.
# The result is concatenated along the output channel dimension in DRAM.
def split_conv_and_run(
    hidden_states,
    conv_weight,
    conv_bias,
    device,
    in_channels,
    input_height,
    input_width,
    out_channels,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    compute_config,
    conv_config,
    kernel_size=3,
    padding=1,
    return_weights_and_bias=False,
):
    split_input_channels = in_channels // conv_in_channel_split_factor
    split_output_channels = out_channels // conv_out_channel_split_factor

    conv_kwargs = {
        "in_channels": split_input_channels,
        "out_channels": split_output_channels,
        "batch_size": 1,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": (kernel_size, kernel_size),
        "stride": (1, 1),
        "padding": (padding, padding),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    outputs = []
    device_weights = []
    device_bias = []
    # First loop goes over output channel slices and saves outputs in a list
    for out_channel_slice_id in range(conv_out_channel_split_factor):
        out_channel_slice_output = None
        device_weights.append([])
        # Second loop goes over input channel slices and accumulates the outputs
        for in_channel_slice_id in range(conv_in_channel_split_factor):
            hidden_states_slice = hidden_states[
                :, :, :, in_channel_slice_id * split_input_channels : (in_channel_slice_id + 1) * split_input_channels
            ]
            results = ttnn.conv2d(
                input_tensor=hidden_states_slice,
                weight_tensor=conv_weight[out_channel_slice_id][in_channel_slice_id],
                bias_tensor=conv_bias[out_channel_slice_id],
                **conv_kwargs,
                compute_config=compute_config,
                return_weights_and_bias=return_weights_and_bias,
            )
            hidden_states_slice.deallocate(True)

            if return_weights_and_bias:
                # First time we call this function, weights and biases are passed in on host;
                # Save them so that we can reuse them on the next calls
                in_channel_slice_output, [weights, bias] = results
                device_weights[out_channel_slice_id].append(weights)
                if in_channel_slice_id == 0:
                    device_bias.append(bias)
            else:
                in_channel_slice_output = results

            if in_channel_slice_id == 0:
                if in_channel_slice_output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                    out_channel_slice_output = ttnn.to_memory_config(in_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG)
                    in_channel_slice_output.deallocate(True)
                else:
                    out_channel_slice_output = in_channel_slice_output
            else:
                out_channel_slice_output = ttnn.add(
                    out_channel_slice_output, in_channel_slice_output, output_tensor=out_channel_slice_output
                )
                in_channel_slice_output.deallocate(True)

        if out_channel_slice_output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            out_channel_slice_output = ttnn.to_memory_config(out_channel_slice_output, ttnn.DRAM_MEMORY_CONFIG)
        outputs.append(out_channel_slice_output)

    hidden_states.deallocate(True)

    # Concatenate the outputs, if we split by output channels
    if len(outputs) > 1:
        output = ttnn.concat(outputs, dim=-1)
        for output_slice in outputs:
            output_slice.deallocate(True)
    else:
        output = outputs[0]

    if return_weights_and_bias:
        return output, device_weights, device_bias

    return output


def prepare_group_norm(device, in_channels, core_grid, torch_weights, torch_bias, num_groups=32):
    num_cores_across_channel = core_grid.y
    torch_input_mask = ttnn.create_group_norm_input_mask(in_channels, num_groups, num_cores_across_channel)
    input_mask = ttnn.from_torch(
        torch_input_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weights = ttnn.from_torch(
        ttnn.create_group_norm_weight_bias_rm(torch_weights, in_channels, num_cores_across_channel),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias = ttnn.from_torch(
        ttnn.create_group_norm_weight_bias_rm(torch_bias, in_channels, num_cores_across_channel),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return input_mask, weights, bias
