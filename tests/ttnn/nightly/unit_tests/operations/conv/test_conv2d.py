# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
from loguru import logger

import torch
import pytest
from models.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
)
from models.utility_functions import skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
import ttnn
from ttnn.operations.conv2d import get_torch_act_func_from_string

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def torch_fast_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden).flatten()
    calculated = torch.Tensor(calculated).flatten()
    if torch.any(torch.isinf(calculated)) or torch.any(torch.isnan(calculated)):
        logger.error("Output tensor contains inf or nan values")
        return False, 0.0
    cov_input = torch.concat([calculated, golden])
    calc_pcc = torch.corrcoef(cov_input)
    return calc_pcc >= pcc, calc_pcc


def check_with_fast_pcc_without_tensor_printout(expected_pytorch_result, actual_pytorch_result, pcc=0.9999):
    if expected_pytorch_result.shape != actual_pytorch_result.shape:
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
        )
    pcc_passed, pcc_message = torch_fast_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, pcc_message


# Cache map used for torch tensor reuse - the tensor will not be generated if a tensor of the same dimensions has already been generated
@pytest.fixture(scope="module")
def torch_tensor_map(request):
    torch_tensor_map = {}

    return torch_tensor_map


def init_tensor_for_tiled_constants(Co, Ci, Kh, Kw, tile_size=32):
    """
    Initialize tensor so that after tiling and reshaping, each tile has a different constant value.

    Args:
        Co, Ci, Kh, Kw: Original tensor dimensions
        tile_size: Size of square tiles (default 32 for 32×32)

    Returns:
        Initialized tensor of shape [Co, Ci, Kh, Kw]
    """

    # Calculate final dimensions after reshape
    final_height = Kh * Kw * Ci  # KhKwCi
    final_width = Co

    # Calculate number of tiles in each dimension
    tiles_h = (final_height + tile_size - 1) // tile_size  # Ceiling division
    tiles_w = (final_width + tile_size - 1) // tile_size

    print(f"Original shape: [{Co}, {Ci}, {Kh}, {Kw}]")
    print(f"After reshape: [1, 1, {final_height}, {final_width}]")
    print(f"Number of tiles: {tiles_h} × {tiles_w} = {tiles_h * tiles_w}")

    # Create a tensor with different constant values for each tile
    # We'll work in the final reshaped space first
    reshaped_tensor = torch.zeros(1, 1, final_height, final_width)

    tile_counter = 0
    for tile_row in range(tiles_h):
        for tile_col in range(tiles_w):
            # Calculate tile boundaries
            h_start = tile_row * tile_size
            h_end = min((tile_row + 1) * tile_size, final_height)
            w_start = tile_col * tile_size
            w_end = min((tile_col + 1) * tile_size, final_width)

            # tile_val = tile_row * tiles_w + tile_col
            # print(tile_val)
            # Fill this tile with a constant value
            reshaped_tensor[0, 0, h_start:h_end, w_start:w_end] = tile_counter
            tile_counter += 1

    # Now we need to reverse the reshape operation
    # The reshape operation is: [Co, Ci, Kh, Kw] -> [1, 1, KhKwCi, Co]
    # So we go: [1, 1, KhKwCi, Co] -> [Co, Ci, Kh, Kw]

    # Remove the first two dimensions
    temp = reshaped_tensor.squeeze(0).squeeze(0)  # Shape: [KhKwCi, Co]
    # torch.set_printoptions(profile="full")
    # print(temp)

    # The forward transformation likely flattens [Ci, Kh, Kw] into KhKwCi
    # So we reshape [KhKwCi, Co] -> [Ci*Kh*Kw, Co] -> [Ci, Kh*Kw, Co] -> [Ci, Kh, Kw, Co]
    # But [KhKwCi, Co] = [Kh*Kw*Ci, Co], so we can reshape to [Kh, Kw, Ci, Co]
    temp = temp.reshape(Kh, Kw, Ci, Co)

    # Now permute to get [Co, Ci, Kh, Kw]
    # From [Kh, Kw, Ci, Co] to [Co, Ci, Kh, Kw] is permute(3, 2, 0, 1)
    original_tensor = temp.permute(3, 2, 0, 1)

    return original_tensor


def randomize_torch_tensor(
    torch_tensor_map,
    tensor_shape,
    generate_positive_numbers=False,
    ones_tensor=False,
    per_channel=False,
    weights_tile=False,
):
    if per_channel:
        print(tensor_shape)
        # Final tensor will be in shape [1, 1, NHW, C]
        tensor_h = tensor_shape[0] * tensor_shape[2] * tensor_shape[3]  # NHW
        tensor_w = tensor_shape[1]
        temp = torch.zeros(tensor_h, tensor_w)
        for i in range(tensor_h):
            temp[i, :] = -i
        # torch.set_printoptions(profile="full")
        # print(temp.shape)
        # print(temp)
        temp = temp.reshape(tensor_shape[0], tensor_shape[2], tensor_shape[3], tensor_w)
        torch_tensor = temp.permute(0, 3, 1, 2)  # [N, C, H, W]

        # torch.set_printoptions(profile="full")
        # print(torch_tensor)
        return torch_tensor

    if ones_tensor:
        torch_tensor = torch.ones(tensor_shape, dtype=torch.bfloat16).float()
        return torch_tensor

    if weights_tile:
        torch_tensor = init_tensor_for_tiled_constants(
            tensor_shape[0], tensor_shape[1], tensor_shape[2], tensor_shape[3]
        )
        return torch_tensor

    if generate_positive_numbers:
        torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16).float()
        torch_tensor = torch.abs(torch_tensor)
        return torch_tensor
    else:
        if tensor_shape in torch_tensor_map.keys():
            torch_tensor = torch_tensor_map[tensor_shape]
        else:
            torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16).float()
            torch_tensor_map[tensor_shape] = torch_tensor

    return torch_tensor


def run_conv(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    padding,
    config_override,
    math_approx_mode=True,
    dilation_h=1,
    dilation_w=1,
    transpose_shards=False,
    fp32_accum=False,
    packer_l1_acc=False,
    input_layout=ttnn.ROW_MAJOR_LAYOUT,
    input_dtype=None,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    groups=1,
    has_bias=False,
    shard_layout=None,
    auto_shard=False,
    memory_config=None,
    input_mesh_mapper=None,
    weight_mesh_mapper=None,
    output_mesh_composer=None,
    enable_split_reader=False,
    activation="",
    in_place=False,
    run_twice=False,
    fast_compare=False,
    slice_config=None,
    enable_kernel_stride_folding=False,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
):
    if isinstance(device, ttnn.MeshDevice) and len(device.get_device_ids()) > 1:
        assert input_mesh_mapper is not None, "Expected mesh mapper for input tensor when running on multiple devices"
        assert (
            weight_mesh_mapper is not None
        ), "Expected mesh mapper for weight tensors when running on multiple devices"
        assert (
            output_mesh_composer is not None
        ), "Expected mesh composer for output tensor when running on multiple devices"
        num_devices = len(device.get_device_ids())
        total_batch_size = num_devices * batch_size  # Batch size across all devices
        logger.info(f"Using {num_devices} devices for this test")
    else:
        total_batch_size = batch_size

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if hasattr(padding, "__len__"):
        if len(padding) == 2:
            pad_top = padding[0]
            pad_bottom = padding[0]
            pad_left = padding[1]
            pad_right = padding[1]
        elif len(padding) == 4:
            pad_top = padding[0]
            pad_bottom = padding[1]
            pad_left = padding[2]
            pad_right = padding[3]
        else:
            raise ValueError("Padding should be a scalar or a list of 2 or 4 elements")
    else:
        pad_top = padding
        pad_bottom = padding
        pad_left = padding
        pad_right = padding

    if input_dtype is None:
        input_dtype = output_dtype
    torch.manual_seed(0)
    conv_input_shape = (total_batch_size, input_channels, input_height, input_width)
    conv_weight_shape = (output_channels, input_channels // groups, filter_height, filter_width)
    conv_bias_shape = (1, 1, 1, output_channels)

    # for Sqrt activation functions we need positive numbers as output of conv2d
    # in order to get valid sqrt values
    sqrt_act_function = activation == "sqrt"
    torch_input_tensor_nchw = randomize_torch_tensor(
        torch_tensor_map, conv_input_shape, generate_positive_numbers=False, ones_tensor=False, per_channel=False
    )
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor = randomize_torch_tensor(
        torch_tensor_map, conv_weight_shape, ones_tensor=False, per_channel=False, weights_tile=False
    )
    torch_bias_tensor = (
        randomize_torch_tensor(torch_tensor_map, conv_bias_shape, generate_positive_numbers=sqrt_act_function) * 10
        if has_bias
        else None
    )

    torch_padded_input = torch.nn.functional.pad(
        torch_input_tensor_nchw,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )
    ref = torch.nn.functional.conv2d(
        torch_padded_input,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(0, 0),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )
    act_func = get_torch_act_func_from_string(activation)
    if act_func:
        ref = act_func(ref)

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        ttnn.bfloat16 if weights_dtype == ttnn.bfloat16 else ttnn.float32,
        mesh_mapper=weight_mesh_mapper,
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor,
            ttnn.bfloat16 if weights_dtype == ttnn.bfloat16 else ttnn.float32,
            mesh_mapper=weight_mesh_mapper,
        )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        input_dtype,
        mesh_mapper=input_mesh_mapper,
        layout=input_layout,
        device=device if input_dtype == ttnn.bfloat8_b else None,
    )

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=shard_layout if not auto_shard else None,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        enable_split_reader=enable_split_reader,
        enable_subblock_padding=False,
        output_layout=output_layout,
        activation=activation,
        transpose_shards=transpose_shards,
        in_place=in_place,
        enable_kernel_stride_folding=enable_kernel_stride_folding,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_approx_mode=math_approx_mode,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override and not auto_shard:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "enable_act_double_buffer" in config_override:
        conv_config.enable_act_double_buffer = config_override["enable_act_double_buffer"]
    if config_override and "enable_weights_double_buffer" in config_override:
        conv_config.enable_weights_double_buffer = config_override["enable_weights_double_buffer"]

    [tt_output_tensor_on_device, [out_height, out_width], [d_w, d_b]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_top, pad_bottom, pad_left, pad_right),
        dilation=(dilation_h, dilation_w),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        memory_config=memory_config,
        slice_config=slice_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=output_dtype,
    )
    # torch_dw = ttnn.to_torch(d_w, mesh_composer=weight_mesh_mapper)
    # torch.set_printoptions(profile="full")
    # print(torch_dw)

    if run_twice:
        [tt_output_tensor_on_device, [out_height, out_width], [d_w, d_b]] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=d_w,
            in_channels=input_channels,
            out_channels=output_channels,
            device=device,
            bias_tensor=d_b,
            kernel_size=(filter_height, filter_width),
            stride=(stride_h, stride_w),
            padding=(pad_top, pad_bottom, pad_left, pad_right),
            dilation=(dilation_h, dilation_w),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=groups,
            memory_config=memory_config,
            slice_config=slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=output_dtype,
        )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    out = ttnn.to_torch(tt_output_tensor, mesh_composer=output_mesh_composer)
    # out is in row major layout and NHWC shape
    # NHWC to NCHW
    out = out.reshape(total_batch_size, out_height, out_width, out.shape[-1])
    out = out[:, :, :, :output_channels]

    ref = torch.permute(ref, (0, 2, 3, 1))

    if not fp32_accum:
        pcc = 0.985
        if input_channels * filter_height * filter_width > 10000:
            pcc = 0.97
    elif math_fidelity == ttnn.MathFidelity.LoFi and output_dtype == ttnn.bfloat8_b:
        pcc = 0.996
    else:
        pcc = 0.997

    if activation == "tanh":
        # Scale down PCC for tanh.
        # tanh has a range of -1 to 1. So discrepancies in output values which are close to 0 tend to disproportionately affect the PCC.
        pcc = pcc * 0.99

    # torch.set_printoptions(precision=3, sci_mode=False)
    if fast_compare:
        if fp32_accum and output_dtype != ttnn.bfloat8_b and input_dtype != ttnn.bfloat8_b:
            threshold = 3e-1 + 5e-3 * math.log(input_channels * filter_height * filter_width, 2)
        else:
            threshold = 3e-1 + 1e-1 * math.log(input_channels * filter_height * filter_width, 2)
        logger.info(f"Threshold: {threshold}")
        diff = torch.abs(ref - out) / ref.abs().mean()
        assert torch.all(diff < threshold), f"Max diff: {diff.max()}, Threshold: {threshold} "
    else:
        # print(out)
        # print(ref)
        passing, pcc_msg = check_with_pcc_without_tensor_printout(out, ref, pcc=pcc)
        logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
        assert passing, pcc_msg
        assert pcc_msg != 1, "Conv2d with ranndomized input and wegihts can't ligitimately return PCC of 1"

    if memory_config:
        output_memory_config = ttnn.get_memory_config(tt_output_tensor_on_device)
        logger.info(f"Output Memory Config : {output_memory_config}")
        assert output_memory_config == memory_config


def run_conv_with_split(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    padding,
    config_override,
    shard_layout=None,
    split_input_channels_factor=2,
    split_output_channels_factor=1,
    fp32_accum=False,
    packer_l1_acc=False,
    auto_shard=False,
    pcc=0.98,
    input_layout=ttnn.ROW_MAJOR_LAYOUT,
):
    if hasattr(padding, "__len__"):
        if len(padding) == 2:
            pad_top = padding[0]
            pad_bottom = padding[0]
            pad_left = padding[1]
            pad_right = padding[1]
        elif len(padding) == 4:
            pad_top = padding[0]
            pad_bottom = padding[1]
            pad_left = padding[2]
            pad_right = padding[3]
        else:
            raise ValueError("Padding should be a scalar or a list of 2 or 4 elements")
    else:
        pad_top = padding
        pad_bottom = padding
        pad_left = padding
        pad_right = padding

    torch.manual_seed(0)
    assert input_channels % split_input_channels_factor == 0
    assert output_channels % split_output_channels_factor == 0
    split_input_channels = input_channels // split_input_channels_factor
    split_output_channels = output_channels // split_output_channels_factor
    full_conv_input_shape = (batch_size, input_channels, input_height, input_width)
    full_conv_weight_shape = (output_channels, input_channels, filter_height, filter_width)
    torch_input_tensor_nchw = randomize_torch_tensor(torch_tensor_map, full_conv_input_shape)
    torch_weight_tensor = randomize_torch_tensor(torch_tensor_map, full_conv_weight_shape)
    conv_bias_shape = (1, 1, 1, output_channels)
    torch_bias_tensor = randomize_torch_tensor(torch_tensor_map, conv_bias_shape)

    torch_padded_input = torch.nn.functional.pad(
        torch_input_tensor_nchw,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_padded_input,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1),
        stride=(stride_h, stride_w),
        padding=(0, 0),
    )

    split_input_tensors = torch.split(torch_input_tensor_nchw, split_input_channels, 1)

    # weights
    if split_output_channels_factor > 1:
        split_weight_tensors = list(torch.split(torch_weight_tensor, split_output_channels, 0))
    else:
        split_weight_tensors = [torch_weight_tensor]

    # bias
    if split_output_channels_factor > 1:
        split_bias_tensors = list(torch.split(torch_bias_tensor, split_output_channels, 3))
    else:
        split_bias_tensors = [torch_bias_tensor]

    for i in range(len(split_weight_tensors)):
        split_weight_tensors[i] = torch.split(split_weight_tensors[i], split_input_channels, 1)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=shard_layout if not auto_shard else None,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]
    torch_output_tensor = None
    for output_channel_slice in range(split_output_channels_factor):
        torch_output_tensor_per_output_slice = None
        for i in range(split_input_channels_factor):
            tt_weight_tensor = ttnn.from_torch(
                split_weight_tensors[output_channel_slice][i],
                weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
            )
            tt_bias_tensor = ttnn.from_torch(
                split_bias_tensors[output_channel_slice],
                weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
            )
            torch_input_tensor = torch.permute(split_input_tensors[i], (0, 2, 3, 1))
            tt_input_tensor = ttnn.from_torch(
                torch_input_tensor,
                output_dtype,
                layout=input_layout,
                device=device if output_dtype == ttnn.bfloat8_b else None,
            )
            [tt_output_tensor_on_device, [out_height, out_width]] = ttnn.conv2d(
                input_tensor=tt_input_tensor,
                weight_tensor=tt_weight_tensor,
                in_channels=split_input_channels,
                out_channels=split_output_channels,
                device=device,
                bias_tensor=tt_bias_tensor,
                kernel_size=(filter_height, filter_width),
                stride=(stride_h, stride_w),
                padding=(pad_top, pad_bottom, pad_left, pad_right),
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                conv_config=conv_config,
                compute_config=compute_config,
                return_output_dim=True,
                dtype=output_dtype,
            )
            tt_conv_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
            ttnn.deallocate(tt_output_tensor_on_device, True)
            torch_conv_output_tensor = ttnn.to_torch(tt_conv_output_tensor)
            torch_conv_output_tensor = torch_conv_output_tensor.reshape(
                batch_size, out_height, out_width, split_output_channels
            )

            # torch_output_tensor is in row major layout and NHWC shape
            # NHWC to NCHW
            torch_conv_output_tensor = torch.permute(torch_conv_output_tensor, (0, 3, 1, 2))
            if i == 0:
                torch_output_tensor_per_output_slice = torch_conv_output_tensor
            else:
                torch_output_tensor_per_output_slice = torch.add(
                    torch_output_tensor_per_output_slice, torch_conv_output_tensor
                )
        if output_channel_slice == 0:
            torch_output_tensor = torch_output_tensor_per_output_slice
        else:
            torch_output_tensor = torch.concat([torch_output_tensor, torch_output_tensor_per_output_slice], dim=1)

    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 2 * 16384}], indirect=True)
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (256, 256, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (16, 16, 256, 256, HS, {"act_block_h": 32}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "filter, pad",
    [
        [3, 1],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_features_multi_device(
    mesh_device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    pad,
    output_layout,
    groups,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    run_conv(
        mesh_device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        pad,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        input_mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        weight_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        output_mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        groups=groups,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (256, 256, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (32, 32, 256, 256, HS, {"act_block_h": 32}),
        (32, 32, 256, 256, HS, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype, output_layout",
    [
        [ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT],
        [ttnn.bfloat8_b, ttnn.TILE_LAYOUT],
    ],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [True],
)
@pytest.mark.parametrize(
    "has_bias",
    [True],
)
@pytest.mark.parametrize(
    "filter, pad",
    [
        [3, 1],
    ],
)
@pytest.mark.parametrize("enable_act_double_buffer", [True, False])
@pytest.mark.parametrize("enable_weights_double_buffer", [True, False])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("activation", ["", "relu", "silu", "sigmoid", "tanh", "gelu"])
def test_conv_activation(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    pad,
    output_layout,
    fp32_accum,
    has_bias,
    activation,
    enable_act_double_buffer,
    enable_weights_double_buffer,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if output_dtype == ttnn.bfloat8_b and shard_layout == HS and activation == "sqrt":
        pytest.skip(
            "Skipping sqrt activation for bfloat8_b and height sharded due to PCC error that occurs due to how sqrt over negative numbers is handled in bfloat8_b vs bfloat16"
        )

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        pad,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=has_bias,
        fp32_accum=fp32_accum,
        packer_l1_acc=False,
        activation=activation,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
    )


SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


@skip_for_blackhole("Not fully tested on Blackhole")
@pytest.mark.parametrize(
    "input_dtype, input_layout",
    [[ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT], [ttnn.bfloat8_b, ttnn.TILE_LAYOUT]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, output_channels, input_height, input_width, slice_type, num_slices, weights_dtype, output_dtype, kernel, stride, padding, dilation, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        (10,    64,  4096,   512,  SliceHeight,   4,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1),  32 * 8, ttnn.MathFidelity.LoFi  ),
        (64,    64,  2048,   256,  SliceHeight,   4,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1),  32 * 16, ttnn.MathFidelity.LoFi  ),
        (64,    64,  1024,   128,  SliceHeight,   2,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1),  0,       ttnn.MathFidelity.LoFi  ),
        (64,    64,   512,    64,  SliceHeight,   2,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1),  0,       ttnn.MathFidelity.LoFi  ),
        ( 4,    32,  1024,  1024,   SliceWidth,   4,  ttnn.bfloat8_b, ttnn.bfloat16, (5, 5), (1, 1), (0, 0), (1, 1),  32,      ttnn.MathFidelity.LoFi  ),
        (32,    48,  1020,  1020,   SliceWidth,   7,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (0, 0), (2, 2),  32 * 2,  ttnn.MathFidelity.LoFi  ),
        (48,    56,  1016,  1016,   SliceWidth,   7,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (0, 0), (4, 4),  32 * 3,  ttnn.MathFidelity.LoFi  ),
        (56,    64,  1008,   256,   SliceWidth,   2,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (0, 0), (8, 8),  0,       ttnn.MathFidelity.LoFi  ),
        (64,   128,   992,   992,   SliceWidth,   50,  ttnn.bfloat8_b, ttnn.bfloat16, (2, 2), (1, 1), (0, 0), (1, 1), 32 * 4,  ttnn.MathFidelity.LoFi  ),
        (128,  128,  1024,  1024,  SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1),  0,       ttnn.MathFidelity.LoFi  ),
        (128,  3,   1024,  1024,  SliceWidth,     8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 0,       ttnn.MathFidelity.LoFi  ),
        (16,   512,  128,    128,   SliceWidth,    2,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 0,       ttnn.MathFidelity.LoFi  ),
        (256,  128,  1024,  1024,  SliceWidth,   32,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1),  32 * 4,  ttnn.MathFidelity.LoFi  ),
        (256,  256,  1024,  1024,  SliceWidth,   37,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1),  32 * 8,  ttnn.MathFidelity.LoFi  ),
        (256,  256,  512,   512,   SliceWidth,    4,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1),  0,       ttnn.MathFidelity.LoFi  ),
        (512,  512,  256,   256,   SliceWidth,    2,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1),  0,       ttnn.MathFidelity.LoFi  ),
        (512,  256,  512,   512,   SliceWidth,    8,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1),  0,       ttnn.MathFidelity.LoFi  ),
        (512,  512,  512,   512,   SliceWidth,    10,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1),  0,       ttnn.MathFidelity.LoFi  ),
        (56,    64,  1008,  1008,  SliceWidth,    6,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (0, 0), (8, 8),  0,       ttnn.MathFidelity.LoFi  ),
     )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, False, False]],
)
def test_conv_dram(
    device,
    torch_tensor_map,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    output_dtype,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    input_dtype,
    input_layout,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")

    batch_size = 1
    config = {
        "act_block_h": act_block_h_override,
    }
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config,
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        has_bias=True,
        fp32_accum=fp32_accum,
        input_dtype=input_dtype,
        input_layout=input_layout,
        output_layout=input_layout,
        packer_l1_acc=packer_l1_acc,
        run_twice=False,
        fast_compare=True,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, pad_h, pad_w, act_block_w_div",
    (
        (2, 128, 256, 9, 9, 3, 3, 1, 1, 1),
        (2, 576, 576, 9, 9, 3, 3, 0, 0, 1),
        (2, 960, 960, 5, 5, 3, 3, 0, 0, 1),
        (2, 256, 2048, 9, 9, 3, 3, 1, 1, 1),
        (2, 512, 2048, 17, 17, 3, 3, 1, 1, 1),
        (2, 768, 768, 17, 17, 3, 3, 0, 0, 1),
        (2, 1280, 2560, 15, 15, 3, 3, 1, 1, 1),
        (2, 1280, 1280, 17, 17, 3, 3, 1, 1, 1),
        [1, 3024, 1232, 14, 14, 1, 1, 0, 0, 1],
        (2, 768, 32, 9, 9, 3, 3, 1, 1, 1),
        (2, 64, 128, 9, 9, 3, 3, 1, 1, 1),
        (2, 32, 128, 9, 9, 3, 3, 1, 1, 1),
    ),
)
@pytest.mark.parametrize(
    "has_bias",
    [True],
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "enable_act_double_buffer",
    [True, False],
)
@pytest.mark.parametrize(
    "enable_weights_double_buffer",
    [True, False],
)
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
@pytest.mark.parametrize("tilized_input", [True, False], ids=["tilized", "row_major"])
def test_conv_ws(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    pad_h,
    pad_w,
    act_block_w_div,
    stride,
    has_bias,
    weights_dtype,
    output_dtype,
    auto_shard,
    tilized_input,
    enable_act_double_buffer,
    enable_weights_double_buffer,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    if input_channels == 2560 and auto_shard:
        pytest.skip("Skipping 2560 input channels with auto_shard due to #23712")

    stride_h = stride
    stride_w = stride
    fp32_accum = True
    packer_l1_acc = True
    deallocate_activation = False
    groups = 1

    conv_input_shape = (batch_size, input_channels, input_height, input_width)
    conv_weight_shape = (output_channels, input_channels // groups, filter_height, filter_width)
    conv_bias_shape = (1, 1, 1, output_channels)

    torch_input_tensor_nchw = randomize_torch_tensor(torch_tensor_map, conv_input_shape)
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor = randomize_torch_tensor(torch_tensor_map, conv_weight_shape)

    tt_bias_tensor = None
    torch_bias_tensor = None
    if has_bias:
        torch_bias_tensor = randomize_torch_tensor(torch_tensor_map, conv_bias_shape) * 50
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
        torch_bias_tensor = torch_bias_tensor.reshape(-1)
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)

    tt_input_tensor = ttnn.reshape(tt_input_tensor, [1, 1, input_height * input_width * batch_size, input_channels])
    if tilized_input:
        tt_input_tensor = ttnn.to_device(tt_input_tensor, device)
        tt_input_tensor = ttnn.to_layout(tt_input_tensor, ttnn.TILE_LAYOUT)

    if auto_shard and (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        if input_channels == 2048:
            pytest.skip("Test is not supported on n300 (8,7) grid due to #13541")

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED if not auto_shard else None,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        enable_split_reader=False,
        enable_subblock_padding=False,
        reshard_if_not_optimal=True,
        act_block_w_div=act_block_w_div if not auto_shard else 1,
        act_block_h_override=32,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    [tt_output_tensor_on_device, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        return_output_dim=True,
        dtype=output_dtype,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    # torch_output_tensor = torch_output_tensor[:, :, : batch_size * out_height * out_width, :]
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, output_channels)
    logger.info(f"Output Shape : {torch_output_tensor.shape}")
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    pcc = 0.99
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"{pcc_msg} Threshold : {pcc}")
    if not passing:
        logger.error("Fails with PCC ", pcc_msg)
    assert passing
    assert pcc_msg != 1.0, "Conv2d with ranndomized input and wegihts can't ligitimately return PCC of 1"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, shard_layout, config_override",
    (
        # mlp sub_module
        (1, 3, 32, 512, 512, 7, 7, 4, 4, 3, 3, 1, HS, {"act_block_h": 64}),  # ncrisc build failed
        # efficient selfattention sub_module
        (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, HS, None),  # ncrisc build failed, Two times called in model
        (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, HS, None),  # ncrisc build failed, Two times called in model
        (1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, HS, None),  # pass , Two times called in model
        # dwconv sub_module
        (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 128, HS, {"act_block_h": 64}),
        (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 256, HS, None),  # pass , Two times called in model
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 640, ttnn.TensorMemoryLayout.BLOCK_SHARDED, {"act_block_h": 32}),
        # (1,1024, 1024, 16, 16, 3, 3, 1, 1, 1, 1, 1024, BS, None), #Switch to Width Sharding
        # decode_head sub_module
        # (1,1024, 256, 128, 128, 1, 1, 1, 1, 0, 0, 1, BS, {"act_block_h": 32}), #pass for activation_dtype=bf8 but fails for bf16
        (1, 256, 150, 128, 128, 1, 1, 1, 1, 0, 0, 1, HS, None),
        (1, 32, 16, 64, 64, 1, 1, 1, 1, 0, 0, 1, HS, None),
        (1, 96, 24, 32, 32, 1, 1, 1, 1, 0, 0, 1, HS, None),
        (1, 576, 576, 8, 8, 3, 3, 1, 1, 0, 0, 576, WS, None),
        (1, 576, 576, 8, 8, 3, 3, 2, 2, 0, 0, 576, WS, None),
        (1, 960, 960, 4, 4, 3, 3, 1, 1, 0, 0, 960, WS, None),
        (1, 144, 24, 32, 32, 1, 1, 1, 1, 0, 0, 1, HS, None),
        (1, 144, 32, 16, 16, 1, 1, 1, 1, 0, 0, 1, HS, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_conv_for_segformer_512x512(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    groups,
    output_layout,
    auto_shard,
):
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        groups=groups,
        output_layout=output_layout,
        has_bias=False,
        auto_shard=auto_shard,
        shard_layout=shard_layout,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, None), HANGS!!
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 256}),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),  Out of Memory!!
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, {"act_block_h": 32}),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # rn50 layer3
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # rn50 layer4
        (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        ## small test
        (1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, BS, {"num_cores_nhw": 2, "grid_size": (2, 2)}),
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, BS, {"num_cores_nhw": 4, "grid_size": (2, 4)}),
        # (1, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, BS, None), sliding_window_op_infra/sliding_window.cpp:341: indices_length_last_core <= indices_length_per_core
        (8, 256, 256, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # r50 1x1s2 shapes
        # Fails with packer_l1_acc = True (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, BS, None),  # r50 first bottleneck downsample shape
        (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, HS, None),  # r50 first bottleneck downsample shape
        # Fails with packer_l1_acc = True (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, BS, None),  # r50 second bottleneck downsample shape
        # (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, BS, None),  # r50 third bottleneck downsample shape
        # (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, BS, None),  # r50 fourth bottleneck downsample shape
        # (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, HS, None), - doesnt fit
        # (20, 128, 256, 56, 56, 1, 1, 2, 2, 0, 0, HS, None),  ## L2M1 DS: doesn't fit
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("packer_l1_acc", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_resnet50_conv_wh(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    packer_l1_acc,
    has_bias,
    auto_shard,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override=config_override,
        packer_l1_acc=packer_l1_acc,
        fp32_accum=False,
        has_bias=has_bias,
        auto_shard=auto_shard,
        shard_layout=shard_layout,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 256}),
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
    ),
)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_conv_mem_config_wh(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    memory_config,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        shard_layout=shard_layout,
        config_override=config_override,
        packer_l1_acc=True,
        fp32_accum=False,
        has_bias=True,
        auto_shard=False,
        memory_config=memory_config,
        input_layout=ttnn.TILE_LAYOUT,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, None),
        # (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32}),
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS, None),
        # # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        # (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, None),
        # (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, HS, {"act_block_h": 32}),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, HS, None),
        # # rn50 layer3
        # (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, BS, None),
        # (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS, None),
        # # rn50 layer4
        # (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, BS, None),
        # (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
        # (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, BS, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("packer_l1_acc", [True])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_resnet50_conv_wh_fp32(
    device,
    torch_tensor_map,
    math_fidelity,
    fp32_accum,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    packer_l1_acc,
    auto_shard,
):
    if batch_size > 8 and (output_dtype != ttnn.bfloat8_b or weights_dtype != ttnn.bfloat8_b):
        pytest.skip("Batch > 8 must be run fully bfp8")

    if (
        output_dtype == ttnn.bfloat16
        and batch_size == 20
        and (
            output_channels == 64
            or (
                stride_h == 2
                and (output_channels == 256 or (output_channels == 128 and weights_dtype == ttnn.bfloat16))
            )
        )
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        shard_layout=shard_layout,
        config_override=config_override,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        auto_shard=auto_shard,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # sd convs with HxW=32x32
        # (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 320, 320, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None), # bfloat16 activations doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None), # slighlty low pcc with 0.99689. bfloat16 weights doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 2, 2, 1, 1, BS, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 4, 4, 3, 3, 1, 1, 1, 1, BS, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None), # slightly low pcc with 0.99698. bfloat16 weights doesnt fit
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, None), # doesnt fit at all.. for all data types
        # sd convs with HxW=64x64 with batch size = 1
        (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  #
        (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit.
        (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (1, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # sd convs with HxW=64x64 with batch size=2
        # (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None), Hangs on WH
        (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),  # fits with bfloat8_b
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        # (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # 1x1 conv
        (2, 320, 960, 64, 64, 1, 1, 1, 1, 0, 0, HS, None),
        # Small conv
        # (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, HS, None),  ## batch = 1 is currently not supported
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("enable_auto_formatting", [True, False])
# Some tests fail with auto_shard on grayskull
@pytest.mark.parametrize("auto_shard", [False], ids=["no_auto_shard"])
def test_sd_conv(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    enable_auto_formatting,
    auto_shard,
):
    if filter_height > 1 and (input_channels > 1280 or (input_channels > 640 and input_height > 16)):
        if enable_auto_formatting:
            pytest.skip("Not running split SD conv with auto formatting")
        run_conv_with_split(
            device,
            torch_tensor_map,
            math_fidelity,
            output_dtype,
            weights_dtype,
            batch_size,
            output_channels,
            input_channels,
            input_height,
            input_width,
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            (pad_h, pad_w),
            config_override,
            shard_layout=shard_layout,
            split_input_channels_factor=3 if input_channels == 1920 else 2,
            input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
        )
    else:
        run_conv(
            device,
            torch_tensor_map,
            math_fidelity,
            output_dtype,
            weights_dtype,
            batch_size,
            output_channels,
            input_channels,
            input_height,
            input_width,
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            (pad_h, pad_w),
            config_override,
            shard_layout=shard_layout,
            auto_shard=auto_shard,
            input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # sd convs with HxW=32x32
        # (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 320, 320, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None), # bfloat16 activations doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None), # slighlty low pcc with 0.99689. bfloat16 weights doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 2, 2, 1, 1, BS, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 4, 4, 3, 3, 1, 1, 1, 1, BS, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None), # slightly low pcc with 0.99698. bfloat16 weights doesnt fit
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, None), # doesnt fit at all.. for all data types
        # sd convs with HxW=64x64 with batch size = 1
        # (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None),
        # (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        # (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  #
        # (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit.
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),
        # (1, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        # # sd convs with HxW=64x64 with batch size=2
        (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),  # fits with bfloat8_b
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        # (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), L1 Allocation Error
        (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),
        (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # 1x1 conv
        (2, 320, 960, 64, 64, 1, 1, 1, 1, 0, 0, HS, None),
        # Small conv
        # (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, HS, None), fails
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "output_dtype, output_layout",
    [(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [
        False,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
def test_sd_conv_wh(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    fp32_accum,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    output_layout,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")

    # Skip test cases raising OOM, but do not affect the SD e2e test
    if (
        (input_channels == 320 and config_override == None and output_dtype == ttnn.bfloat16)
        or (input_channels == 960 and config_override == None and fp32_accum == True)
        or (
            output_channels == 1280
            and input_height == 32
            and output_dtype == ttnn.bfloat16
            and weights_dtype == ttnn.bfloat16
        )
    ):
        pytest.skip("Skip the test cases raising OOM but not affecting e2e test")

    if filter_height > 1 and (input_channels > 1280 or (input_channels > 640 and input_height > 16)):
        run_conv_with_split(
            device,
            torch_tensor_map,
            math_fidelity,
            output_dtype,
            weights_dtype,
            batch_size,
            output_channels,
            input_channels,
            input_height,
            input_width,
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            (pad_h, pad_w),
            config_override,
            shard_layout=shard_layout,
            split_input_channels_factor=3 if input_channels == 1920 else 2,
            fp32_accum=fp32_accum,
            packer_l1_acc=True,
            input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
        )
    else:
        run_conv(
            device,
            torch_tensor_map,
            math_fidelity,
            output_dtype,
            weights_dtype,
            batch_size,
            output_channels,
            input_channels,
            input_height,
            input_width,
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            (pad_h, pad_w),
            config_override,
            shard_layout=shard_layout,
            fp32_accum=fp32_accum,
            packer_l1_acc=True,
            output_layout=output_layout,
            input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
        )


# VAE 1.4
@pytest.mark.parametrize(
    "input_channels, output_channels, input_height, input_width, split_factor_input_channels, split_factor_output_channels",
    (
        (512, 512, 64, 64, 1, 1),
        (512, 256, 256, 256, 2, 1),
        (256, 256, 256, 256, 1, 1),
        (256, 128, 512, 512, 8 if is_wormhole_b0() else 4, 1),
        (128, 128, 512, 512, 4 if is_wormhole_b0() else 2, 1),
        (512, 512, 256, 256, 8 if is_wormhole_b0() else 2, 1 if is_wormhole_b0() else 2),
        (256, 256, 512, 512, 8 if is_wormhole_b0() else 4, 2),
        (128, 3, 512, 512, 2, 1),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2 * 16384}], indirect=True)
def test_sd14_vae_conv(
    device,
    torch_tensor_map,
    input_channels,
    output_channels,
    input_height,
    input_width,
    split_factor_input_channels,
    split_factor_output_channels,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    batch = 1
    dtype = ttnn.bfloat8_b
    kernel = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    if split_factor_input_channels > 1 or split_factor_output_channels > 1:
        run_conv_with_split(
            device,
            torch_tensor_map,
            ttnn.MathFidelity.LoFi,
            dtype,
            dtype,
            batch,
            output_channels,
            input_channels,
            input_height,
            input_width,
            kernel[0],
            kernel[1],
            stride[0],
            stride[1],
            padding,
            None,
            shard_layout=None,
            split_input_channels_factor=split_factor_input_channels,
            split_output_channels_factor=split_factor_output_channels,
            auto_shard=True,
            input_layout=ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else None,
        )
    else:
        run_conv(
            device=device,
            torch_tensor_map=torch_tensor_map,
            math_fidelity=ttnn.MathFidelity.LoFi,
            output_dtype=dtype,
            weights_dtype=dtype,
            batch_size=batch,
            output_channels=output_channels,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            filter_height=kernel[0],
            filter_width=kernel[1],
            stride_h=stride[0],
            stride_w=stride[1],
            padding=padding,
            config_override=None,
            output_layout=ttnn.TILE_LAYOUT,
            shard_layout=None,
            auto_shard=True,
            input_layout=ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else None,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # unet convs with batch size 2
        # unique convs in unet (complete list)
        (2, 16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (2, 16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        # (2, 1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 5 * 32}, False) # Enable when issue #11490 resolved
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype, output_layout",
    [(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_unet_conv_wh(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    output_layout,
    auto_shard,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        output_layout=output_layout,
        auto_shard=auto_shard,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [2],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}),
        (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}),
        (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 5 * 32}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_unet_conv_groups_2_wh(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    output_layout,
    auto_shard,
    groups,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        output_layout=output_layout,
        auto_shard=auto_shard,
        groups=groups,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [4, 6],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, in_place",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, False),
        (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, False),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, False),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, False),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, False),
        (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 2 * 32}, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_unet_conv_groups_4_6_wh(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    output_layout,
    groups,
    in_place,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    if input_channels == 32 and input_height == 1056 and groups == 6:
        pytest.skip("OOM - enable when support for full in-place conv2d")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT,
        output_layout=output_layout,
        groups=groups,
        in_place=in_place,
    )


@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [8],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}),
        # (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}), # OOM - need inplace convolution
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None),
        # (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None), # OOM - need inplace convolution
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None),
        # (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}), # OOM - need inplace convolution
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}),
        # (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}), # OOM - need inplace convolution
        # (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, True, {"act_block_h": 2 * 32}), # OOM - need inplace convolution
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [False], ids=["no_auto_shard"])
def test_unet_conv_groups_8_wh(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    output_layout,
    auto_shard,
    groups,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        output_layout=output_layout,
        auto_shard=auto_shard,
        groups=groups,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override",
    (
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 1}),
        (1, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, {"act_reshard_num_cores_nhw": 1}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 4}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8, "num_cores_nhw": 4}),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8, "num_cores_nhw": 4}),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 4, "num_cores_nhw": 8}),
    ),
)
@pytest.mark.parametrize("shard_layout", [BS, HS])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_halo_reshard_conv(
    device,
    torch_tensor_map,
    shard_layout,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    config_override,
    auto_shard,
):
    math_fidelity = ttnn.MathFidelity.HiFi4
    output_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        auto_shard=auto_shard,
    )


@pytest.mark.skip("New API needs to be tested")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override, xfail",
    (
        (1, 128, 128, 17, 17, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 4}, False),
        (1, 128, 128, 17, 17, 3, 3, 2, 2, 1, 1, {"num_cores_nhw": 2}, False),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 3}, False),
        (2, 64, 64, 23, 23, 3, 3, 2, 2, 1, 1, {"num_cores_nhw": 3}, False),
        (1, 64, 64, 23, 23, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 10}, True),
    ),
)
@pytest.mark.parametrize("shard_layout", [BS, HS])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_conv_core_nondivis(
    device,
    torch_tensor_map,
    shard_layout,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    config_override,
    xfail,
    auto_shard,
):
    if xfail:
        pytest.xfail()

    math_fidelity = ttnn.MathFidelity.HiFi4
    output_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        auto_shard=auto_shard,
    )


# The following test takes various shape sizes from resnet50, unet and stable diffusion and tests for different number of groups - all the way to num_groups = num_in_channels (depthwise conv)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width,  act_block_w_div, shard_layout",
    (
        (768, 768, 16, 16, 1, WS),
        (1280, 1280, 16, 16, 1, WS),
        (1280, 1280, 8, 8, 1, WS),
        (1280, 2560, 8, 8, 1, WS),
        (128, 128, 8, 8, 1, BS),
        (128, 128, 16, 16, 1, BS),
        (128, 128, 32, 32, 1, BS),
        (32, 32, 64, 64, 1, HS),
        (32, 32, 128, 64, 1, HS),
        (16, 16, 528, 80, 1, HS),
        (32, 16, 264, 40, 1, HS),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "filter_hw, dilation_hw, pad_hw",
    [
        [(3, 3), (2, 2), (2, 2)],
        [(3, 3), (3, 3), (3, 3)],
        [(3, 3), (1, 2), (3, 3)],
        [(3, 3), (2, 1), (3, 3)],
    ],
)
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
@pytest.mark.parametrize("split_reader", [True, False], ids=["split_reader_on", "split_reader_on_off"])
def test_conv_dilation(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    act_block_w_div,
    shard_layout,
    filter_hw,
    stride,
    pad_hw,
    output_layout,
    dilation_hw,
    auto_shard,
    split_reader,
):
    config_override = {"act_block_w_div": act_block_w_div}
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_hw[0],
        filter_hw[1],
        stride,
        stride,
        pad_hw,
        config_override,
        shard_layout=shard_layout,
        output_layout=output_layout,
        dilation_h=dilation_hw[0],
        dilation_w=dilation_hw[1],
        has_bias=False,
        auto_shard=auto_shard,
        enable_split_reader=split_reader,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


# The following test takes various shape sizes from resnet50, unet and stable diffusion and tests for different number of groups - all the way to num_groups = num_in_channels (depthwise conv)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, shard_layout, config_override",
    (
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 2, HS, None),
        (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 64, HS, None),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 1, HS, None),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 2, HS, None),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 8, HS, None),
        (1, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, HS, None),
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 64, HS, None),
        (4, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 128, HS, None),
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 128, HS, None),
        # (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 256, BS, None), circular buffer error
        # (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 256, BS, None), # doesn't fit with bfloat16 weights
        # (32, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 512, BS, None), # doesn't fit with bfloat16 weights
        (32, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, 40, BS, None),
        (32, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, 10, BS, None),
        (1, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 8, HS, None),
        (1, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 16, HS, None),
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 32, HS, None),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 2, BS, None),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 4, BS, None),
        (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, 2, BS, None),
        (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, 320, BS, None),
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, BS, None), # doesn't fit with bfloat16 weights
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, 32, HS, None),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, 2, HS, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_conv_groups(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    groups,
    output_layout,
    auto_shard,
):
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        groups=groups,
        output_layout=output_layout,
        auto_shard=auto_shard,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, groups",
    (
        # yolov4 convs with batch size 1
        # unique convs in yolov4 (complete list) # groups: number
        # (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, HS, None,32),  # groups: 32
        # (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, HS, None,32),  # groups: 32
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, HS, None,64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, HS, None,64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, HS, None,64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, HS, None,64),  # groups: 64
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, 128),  # groups: 128
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, 256),  # groups: 256
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, 512),  # groups: 512
        (1, 128, 128, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, 2),  # groups: 512
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_yolov4_conv_groups_larger_than_one(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    groups,
    output_layout,
    auto_shard,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        groups=groups,
        output_layout=output_layout,
        auto_shard=auto_shard,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    " output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, groups",
    ((96, 3, 512, 512, 4, 4, 4, 4, 0, 0, HS, None, 1),),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 8],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_swin_s_conv(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    groups,
    output_layout,
    auto_shard,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")
    if batch_size == 8:
        pytest.skip("OOM issue for batch_size 8")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        groups=groups,
        output_layout=output_layout,
        auto_shard=auto_shard,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, dilation, shard_layout",
    (
        (1, 48, 32, 252, 252, 3, 3, 1, 1, 0, 0, 2, HS),
        (1, 56, 48, 248, 248, 3, 3, 1, 1, 0, 0, 4, HS),
        (1, 64, 56, 240, 240, 3, 3, 1, 1, 0, 0, 8, HS),
        (1, 48, 32, 124, 124, 3, 3, 1, 1, 0, 0, 2, HS),
        (1, 56, 48, 120, 120, 3, 3, 1, 1, 0, 0, 4, HS),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_model_k_256x256(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation,
    shard_layout,
    auto_shard,
):
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        None,
        shard_layout=shard_layout,
        dilation_h=dilation,
        dilation_w=dilation,
        auto_shard=auto_shard,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels,input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        (1, 32, 3, 480, 640, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 64}),
        (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}),
        (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 64, 64, 240, 320, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 64}),
        (1, 128, 64, 120, 160, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 128, 128, 120, 160, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 256, 128, 60, 80, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 256, 256, 60, 80, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 288, 288, 60, 80, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 512, 256, 30, 40, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 512, 512, 30, 40, 3, 3, 1, 1, 1, 1, BS, None),
        (1, 256, 512, 60, 80, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (1, 128, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}),
        (1, 64, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}),
        (1, 32, 64, 256, 256, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}),
        (1, 1, 32, 480, 640, 1, 1, 1, 1, 0, 0, HS, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_for_vanilla_unet(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    output_layout,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override,
        shard_layout=shard_layout,
        groups=1,
        output_layout=output_layout,
        has_bias=False,
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_shallow_conv_with_tiled_input(device):
    out_channels, in_channels, kernel_h, kernel_w = 7, 3, 3, 3
    kernel_shape = (out_channels, in_channels, kernel_h, kernel_w)
    batch_size = 1
    img_h, img_w = 100, 100
    input_shape = (batch_size, in_channels, img_h, img_w)

    stride = (1, 1)
    dilation = (1, 1)
    pad = (1, 1)

    torch_kernel = torch.randn(kernel_shape, dtype=torch.bfloat16)
    tt_kernel = ttnn.from_torch(torch_kernel)

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)
    tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))
    tt_input = ttnn.reshape(tt_input, (1, 1, batch_size * img_h * img_w, in_channels))
    tt_input = ttnn.to_layout(tt_input, ttnn.TILE_LAYOUT)

    [tt_out, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_kernel,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=(kernel_h, kernel_w),
        stride=stride,
        padding=pad,
        dilation=dilation,
        batch_size=batch_size,
        input_height=img_h,
        input_width=img_w,
        groups=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        return_output_dim=True,
    )

    tt_output_tensor = ttnn.from_device(tt_out)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :out_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input, torch_kernel, bias=None, stride=stride, padding=pad, dilation=dilation, groups=1
    )

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    logger.info(f"PCC = {pcc_msg}. Threshold = 0.99")
    assert passing


# Tests running conv2d which maps to matmul w/o sharding the input tensor.
# Output tensor is in DRAM.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("tiled_input", [True, False])
@pytest.mark.parametrize("input_on_device", [True, False])
def test_dram_input_mm_conv(device, torch_tensor_map, tiled_input, input_on_device):
    batch_size = 1
    out_channels, in_channels = 256, 1024
    img_h, img_w = 128, 128
    input_shape = (batch_size, in_channels, img_h, img_w)

    # Params which map conv2d to matmul op.
    kernel_h, kernel_w = 1, 1
    stride = (1, 1)
    dilation = (1, 1)
    pad = (0, 0)

    kernel_shape = (out_channels, in_channels, kernel_h, kernel_w)
    torch_kernel = randomize_torch_tensor(torch_tensor_map, kernel_shape)
    tt_kernel = ttnn.from_torch(torch_kernel, dtype=ttnn.bfloat16)

    torch_input = randomize_torch_tensor(torch_tensor_map, input_shape)
    if input_on_device:
        tt_input = ttnn.from_torch(torch_input, device=device)
        tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))
        tt_input = ttnn.reshape(tt_input, (1, 1, batch_size * img_h * img_w, in_channels))
    else:
        torch_input_nhwc = torch.permute(torch_input, (0, 2, 3, 1))
        tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)

    if tiled_input:
        tt_input = ttnn.to_layout(tt_input, ttnn.TILE_LAYOUT)

    tt_out = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_kernel,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel_h, kernel_w),
        stride=stride,
        padding=pad,
        dilation=dilation,
        batch_size=batch_size,
        input_height=img_h,
        input_width=img_w,
    )

    assert tt_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED

    tt_output_tensor = ttnn.from_device(tt_out)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, img_h, img_w, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :out_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input, torch_kernel, bias=None, stride=stride, padding=pad, dilation=dilation, groups=1
    )

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    logger.info(f"PCC = {pcc_msg}. Threshold = 0.99")
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    ((16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, HS, {"act_block_h": 32 * 49}),),
)
def test_split_reader_regression(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        (pad_h, pad_w),
        config_override=config_override,
        has_bias=False,
        shard_layout=shard_layout,
        enable_split_reader=True,
        input_layout=ttnn.TILE_LAYOUT,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_small_in_large_out_channels_auto_shard(device, torch_tensor_map):
    batch_size = 2
    in_channels = 16
    out_channels = 1536
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    height = 128
    width = 128

    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat16,
        ttnn.bfloat16,
        batch_size,
        out_channels,
        in_channels,
        height,
        width,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding,
        None,
        auto_shard=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_silu_auto_shard_mm_conv(device, torch_tensor_map):
    batch_size = 1
    in_channels = 64
    out_channels = 64
    kernel_size = (1, 1)
    stride = (1, 1)
    padding = (0, 0)
    height = 160
    width = 160

    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat16,
        ttnn.bfloat16,
        batch_size,
        out_channels,
        in_channels,
        height,
        width,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding,
        None,
        auto_shard=True,
        activation="silu",
    )


# fmt: off
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, kernel, stride, padding",
    (
        (1, 64, 64, 128, 128, (3, 3), (1, 1), (1, 1)),
    ),
)
#fmt: on

@pytest.mark.parametrize("shard_layout", [BS])
@pytest.mark.parametrize("activation", ["relu"])

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384*2}], indirect=True)
def test_block_sharding_relu_act_block_h(
    device,
    torch_tensor_map,
    batch,
    input_channels,
    output_channels,
    input_height,
    input_width,
    kernel,
    stride,
    padding,
    shard_layout,
    activation,
):
    config_override = {}
    config_override["act_block_h"] = 32
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat16,
        ttnn.bfloat16,
        batch,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config_override=config_override,
        shard_layout=shard_layout,
        activation=activation,
    )

# fmt: off
@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, weights_dtype, output_dtype, input_dtype, input_layout, groups, kernel, stride, padding, dilation, auto_shard, act_block_h_override, act_block_w_div, deallocate_activation, math_fidelity, fp32_accum, packer_l1_acc, enable_split_reader, enable_act_double_buffer",
    (
        # # model strawberry
        (1, 10,   64, 1024, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (4, 4), (2, 2), [1, 1, 1, 1], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 64,   64, 512,   64, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (4, 4), (2, 2), [1, 1, 1, 1], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 64,   64, 256,   32, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (4, 4), (2, 2), [1, 1, 1, 1], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 64,   64, 128,   16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (4, 4), (2, 2), [1, 1, 1, 1], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1,  2,    1, 1024, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (1, 1), (1, 1), [0, 0, 0, 0], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),

        # # model kiwi
        (1,  4,   32, 288, 288, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (5, 5), (1, 1), [0, 0, 0, 0], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 32,   48, 284, 284, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (3, 3), (1, 1), [0, 0, 0, 0], (2, 2), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 48,   56, 280, 280, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (3, 3), (1, 1), [0, 0, 0, 0], (4, 4), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 56,   64, 272, 272, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (3, 3), (1, 1), [0, 0, 0, 0], (8, 8), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 64,  128, 256, 256, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (2, 2), (1, 1), [0, 0, 0, 0], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 128, 256, 255, 255, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (1, 1), (1, 1), [0, 0, 0, 0], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),
        (1, 256,   1, 255, 255, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 1, (1, 1), (1, 1), [0, 0, 0, 0], (1, 1), True, 0, 1, True, ttnn.MathFidelity.LoFi, False, False, False, False),

        # model kiwi new convs
        ( 1,  4,    32, 1024,  128, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 1, (5, 5), (1, 1), [2, 2, 2, 2], (1, 1), True, 32 * 16, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
        ( 4,  32,   48,  512,   64, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 1, (3, 3), (1, 1), [1, 1, 1, 1], (1, 1), True, 32 * 16, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
        (16,  48,   56,  256,   32, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 1, (3, 3), (1, 1), [1, 1, 1, 1], (1, 1), True, 32 * 16, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
        (64,  56,   64,  128,   16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 1, (3, 3), (1, 1), [1, 1, 1, 1], (1, 1), True, 32 * 16, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
        ( 1,  64,  128,  1024, 128, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 1, (2, 2), (1, 1), [1, 0, 1, 0], (1, 1), True, 32 * 16, 1, True, ttnn.MathFidelity.LoFi, False, False, True, True),
    ),
)
 #fmt: on

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384*2}], indirect=True)
def test_conv2d_model_fruit(
    device,
    torch_tensor_map,
    batch,
    input_channels,
    output_channels,
    input_height,
    input_width,
    weights_dtype,
    output_dtype,
    groups,
    kernel,
    stride,
    padding,
    dilation,
    auto_shard,
    act_block_h_override,
    act_block_w_div,
    deallocate_activation,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
    enable_split_reader,
    enable_act_double_buffer,
    input_dtype,
    input_layout,
):

    if (
        device.core_grid.y < 8
        and is_wormhole_b0()
        and batch == 1
        and input_channels == 64
        and output_channels == 128
        and input_height == 1024
        and input_width == 128
    ):
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    config_override = {}
    config_override["act_block_h"] = act_block_h_override
    config_override["act_block_w_div"] = act_block_w_div

    run_conv(
        device=device,
        torch_tensor_map=torch_tensor_map,
        math_fidelity=math_fidelity,
        output_dtype=output_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=kernel[0],
        filter_width=kernel[1],
        stride_h=stride[0],
        stride_w=stride[1],
        padding=(padding[0], padding[1], padding[2], padding[3]),
        config_override=config_override,
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        output_layout=ttnn.TILE_LAYOUT,
        deallocate_activation=deallocate_activation,
        groups=groups,
        has_bias=True,
        shard_layout=None,
        auto_shard=auto_shard,
        memory_config=None,
        input_mesh_mapper=None,
        weight_mesh_mapper=None,
        output_mesh_composer=None,
        enable_split_reader=enable_split_reader,
        input_layout= input_layout,
        activation="relu",
        enable_act_double_buffer=enable_act_double_buffer,
        input_dtype = input_dtype,
    )



@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, weights_dtype, output_dtype, groups, kernel, stride, padding, dilation, shard_layout, act_block_h_override, act_block_w_div, deallocate_activation, math_fidelity, fp32_accum, packer_l1_acc, enable_split_reader, split_input_channels_factor, split_output_channels_factor, act_db, w_db",
    (
        # 1024x1024 resolution

        # UNet
        # kernel 3x3
        (1, 1280, 1280, 32, 32, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 0,  1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 1280, 1280, 64, 64, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 256,  1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 1280, 640, 64, 64,  ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 128,  1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 1920, 1280, 32, 32, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), WS, 512, 1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 1920, 640, 64, 64,  ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 128, 1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 2560, 1280, 32, 32, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), WS, 256, 1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 320, 640, 64, 64,   ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 512, 1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 320, 320, 128, 128, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 512, 1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 640, 1280, 32, 32,  ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 0,   1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 640, 640, 128, 128, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 128,  1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 640, 640, 64, 64,   ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 0, 1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 640, 320, 128, 128, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 128,  1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 960, 640, 64, 64,   ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 128,  1, True, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 960, 320, 128, 128, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), BS, 512,   1, True, ttnn.MathFidelity.HiFi2, True, False, False, 2, 1, True, True),

        # stride 2x2
        (1, 320, 320, 128, 128, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (2, 2), (1, 1), (1, 1), BS, 512, 1, False, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),
        (1, 640, 640, 64, 64,   ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (2, 2), (1, 1), (1, 1), BS, 0, 1, False, ttnn.MathFidelity.HiFi2, True, False, False, 1, 1, True, True),

        # output_channels 4
        (1, 320, 4, 128, 128,   ttnn.bfloat16, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, 128,  1, True, ttnn.MathFidelity.HiFi2, False, False, True, 1, 1, False, False),

        # input_channels 4
        (1, 4, 320, 128, 128,   ttnn.bfloat16, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), HS, 256,  1, True, ttnn.MathFidelity.HiFi2, False, False, True, 1, 1, True, False),

        # kernel 1x1
        (1, 1280, 640, 64, 64,  ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),
        (1, 1920, 1280, 32, 32, ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),
        (1, 1920, 640, 64, 64,  ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),
        (1, 2560, 1280, 32, 32, ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),
        (1, 320, 640, 64, 64,   ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),
        (1, 640, 1280, 32, 32,  ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),
        (1, 640, 320, 128, 128, ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),
        (1, 960, 640, 64, 64,   ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),
        (1, 960, 320, 128, 128, ttnn.bfloat16, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), None, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False, 1, 1, False, False),

    ),
)

@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 16384}], indirect=True)
def test_conv2d_sdxl(
    device,
    torch_tensor_map,
    batch,
    input_channels,
    output_channels,
    input_height,
    input_width,
    weights_dtype,
    output_dtype,
    groups,
    kernel,
    stride,
    padding,
    dilation,
    shard_layout,
    act_block_h_override,
    act_block_w_div,
    deallocate_activation,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
    enable_split_reader,
    split_input_channels_factor,
    split_output_channels_factor,
    act_db,
    w_db,
):

    # Skip all on N300
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    config_override = {}
    config_override["act_block_h"] = act_block_h_override
    config_override["act_block_w_div"] = act_block_w_div
    config_override["enable_act_double_buffer"] = act_db
    config_override["enable_weights_double_buffer"] = w_db

    if split_input_channels_factor > 1 or split_output_channels_factor > 1:
        run_conv_with_split(
            device,
            torch_tensor_map,
            math_fidelity,
            output_dtype,
            weights_dtype,
            batch,
            output_channels,
            input_channels,
            input_height,
            input_width,
            kernel[0],
            kernel[1],
            stride[0],
            stride[1],
            padding,
            config_override,
            shard_layout=shard_layout,
            split_input_channels_factor=split_input_channels_factor,
            split_output_channels_factor=split_output_channels_factor,
            fp32_accum=fp32_accum,
            packer_l1_acc=packer_l1_acc,
            auto_shard=True if shard_layout is None else False,
            input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
        )
    else:
        run_conv(
            device=device,
            torch_tensor_map=torch_tensor_map,
            math_fidelity=math_fidelity,
            output_dtype=output_dtype,
            weights_dtype=weights_dtype,
            batch_size=batch,
            output_channels=output_channels,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            filter_height=kernel[0],
            filter_width=kernel[1],
            stride_h=stride[0],
            stride_w=stride[1],
            padding=padding,
            config_override=config_override,
            dilation_h=dilation[0],
            dilation_w=dilation[1],
            fp32_accum=fp32_accum,
            packer_l1_acc=packer_l1_acc,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=deallocate_activation,
            groups=groups,
            has_bias=True,
            shard_layout=shard_layout,
            auto_shard=True if shard_layout is None else False,
            memory_config=None,
            input_mesh_mapper=None,
            weight_mesh_mapper=None,
            output_mesh_composer=None,
            enable_split_reader=enable_split_reader,
            input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
        )


@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, weights_dtype, output_dtype, groups, kernel, stride, padding, dilation, auto_shard, deallocate_activation, slice_type, num_slices, act_block_h_override",
    (
        # 1024x1024 resolution

        # VAE
        # kernel 3x3
        (1, 128, 128, 1024, 1024, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, ttnn.Conv2dSliceWidth, 8, 32),
        (1, 256, 128, 1024, 1024, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, ttnn.Conv2dSliceWidth, 16, 32),
        (1, 256, 256, 1024, 1024, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, ttnn.Conv2dSliceWidth, 16, 128),
        (1, 256, 256, 512, 512, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, ttnn.Conv2dSliceWidth, 4, 64),
        (1, 512, 512, 128, 128, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, None, 1, 128),
        (1, 512, 512, 256, 256, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, ttnn.Conv2dSliceWidth, 2, 32),
        (1, 512, 256, 512, 512, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, ttnn.Conv2dSliceWidth, 8, 64),
        (1, 512, 512, 512, 512, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, ttnn.Conv2dSliceWidth, 8, 32),

        # output_channels 3
        (1, 128, 3, 1024, 1024, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, ttnn.Conv2dSliceWidth, 16, 512),

        # input_channels 4
        (1, 4, 512, 128, 128, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, False, None, 1, 256),

        # kernel 1x1
        (1, 256, 128, 1024, 1024, ttnn.bfloat8_b, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), True, False, None, 1, 0),
        (1, 512, 256, 512, 512, ttnn.bfloat8_b, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), True, False, None, 1, 0),

        # channels 4
        # Skip specific test case for Blackhole devices
        (1, 4, 4, 128, 128, ttnn.bfloat8_b, ttnn.bfloat16, 1, (1, 1), (1, 1), (0, 0), (1, 1), True, False, None, 1, 0),
    ),
)

@skip_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 16384}], indirect=True)
def test_conv2d_vae_sdxl(
    device,
    torch_tensor_map,
    batch,
    input_channels,
    output_channels,
    input_height,
    input_width,
    weights_dtype,
    output_dtype,
    groups,
    kernel,
    stride,
    padding,
    dilation,
    auto_shard,
    deallocate_activation,
    slice_type,
    num_slices,
    act_block_h_override
):

    # Skip all on N300
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")
    # Skip specific test case for Blackhole devices
    if is_blackhole() and (batch, input_channels, output_channels, input_height, input_width, weights_dtype) == (1, 4, 4, 128, 128, ttnn.bfloat8_b):
        pytest.skip("Skipping this test case for Blackhole devices due to PCC issue, tracked in ISSUE-24463")

    config_override = {}
    config_override["act_block_h"] = act_block_h_override
    config_override["act_block_w_div"] = 1

    slice_config = ttnn.Conv2dSliceConfig(
        slice_type=slice_type,
        num_slices=num_slices,
    ) if num_slices > 1 and slice_type is not None else None


    run_conv(
        device=device,
        torch_tensor_map=torch_tensor_map,
        math_fidelity=ttnn.MathFidelity.LoFi,
        output_dtype=output_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=kernel[0],
        filter_width=kernel[1],
        stride_h=stride[0],
        stride_w=stride[1],
        padding=padding,
        config_override=config_override,
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        fp32_accum=False,
        packer_l1_acc=False,
        output_layout=ttnn.TILE_LAYOUT,
        deallocate_activation=deallocate_activation,
        groups=groups,
        has_bias=True,
        shard_layout=None,
        auto_shard=auto_shard,
        memory_config=None,
        input_mesh_mapper=None,
        weight_mesh_mapper=None,
        output_mesh_composer=None,
        enable_split_reader=False,
        slice_config=slice_config,
        input_layout=ttnn.TILE_LAYOUT,
    )


@pytest.mark.parametrize(
    "batch, input_channels, output_channels, input_height, input_width, weights_dtype, output_dtype, groups, kernel, stride, padding, dilation, auto_shard, act_block_h_override, act_block_w_div, deallocate_activation, math_fidelity, fp32_accum, packer_l1_acc, enable_split_reader",
    (
        (1, (1920, 1280, 1920), 1280, 32, 32, ttnn.bfloat8_b, ttnn.bfloat16, 1, (3, 3), (1, 1), (1, 1), (1, 1), True, 0, 1, False, ttnn.MathFidelity.LoFi, False, False, False),
    ),
)

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_ws_program_cache(
    device,
    torch_tensor_map,
    batch,
    input_channels,
    output_channels,
    input_height,
    input_width,
    weights_dtype,
    output_dtype,
    groups,
    kernel,
    stride,
    padding,
    dilation,
    auto_shard,
    act_block_h_override,
    act_block_w_div,
    deallocate_activation,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
    enable_split_reader,

):

    config_override = {}
    config_override["act_block_h"] = act_block_h_override
    config_override["act_block_w_div"] = act_block_w_div

    for i in input_channels:
        run_conv(
            device=device,
            torch_tensor_map=torch_tensor_map,
            math_fidelity=math_fidelity,
            output_dtype=output_dtype,
            weights_dtype=weights_dtype,
            batch_size=batch,
            output_channels=output_channels,
            input_channels=i,
            input_height=input_height,
            input_width=input_width,
            filter_height=kernel[0],
            filter_width=kernel[1],
            stride_h=stride[0],
            stride_w=stride[1],
            padding=padding,
            config_override=config_override,
            dilation_h=dilation[0],
            dilation_w=dilation[1],
            fp32_accum=fp32_accum,
            packer_l1_acc=packer_l1_acc,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=deallocate_activation,
            groups=groups,
            has_bias=True,
            shard_layout=None,
            auto_shard=auto_shard,
            memory_config=None,
            input_mesh_mapper=None,
            weight_mesh_mapper=None,
            output_mesh_composer=None,
            enable_split_reader=enable_split_reader,
            run_twice=False,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_sharded_non_tile(device):
    batch = 1
    input_channels = 32
    output_channels = 32
    input_height = 528
    input_width = 80
    filter = 3
    stride = 1
    padding = 0
    shard_height = 671
    shard_width = 32
    input_shape = (batch, input_channels, input_height, input_width)
    weights_shape = (output_channels, input_channels, filter, filter)

    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    torch_input_nhwc = torch.permute(torch_input, (0, 2, 3, 1))

    num_cores = math.ceil((batch * input_height * input_width) / shard_height)
    input_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=ttnn.num_cores_to_corerangeset(target_num_cores=num_cores, grid_size=device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16, device=device, memory_config=input_mem_cfg)

    torch_weights = torch.randn(weights_shape, dtype=torch.bfloat16)
    tt_weights = ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16)
    [tt_out, [oh, ow]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weights,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        kernel_size=(filter, filter),
        stride=(stride, stride),
        padding=(padding, padding),
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        return_output_dim=True,
    )

    torch_output_tensor = ttnn.to_torch(tt_out)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch, oh, ow, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input, torch_weights, bias=None, stride=stride, padding=padding, groups=1
    )

    passing, _ = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    assert passing


@pytest.mark.parametrize("enable_act_double_buffer", [True, False])
@pytest.mark.parametrize("enable_split_reader", [True, False])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_channel_padding(device, enable_act_double_buffer, enable_split_reader):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    patch_size = 7
    stride = 4
    num_channels = 3
    hidden_size = 32
    batch_size = 1
    height = 512
    width = 512
    torch.manual_seed(20250416)

    # In case of output dtype is bfloat8_b, we need to pad the input channels to be divisible by 8.
    required_padding = (8 - num_channels % 8) % 8
    padded_num_channels = num_channels + required_padding

    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)
    torch_input_tensor = torch.nn.functional.pad(torch_input_tensor, (0, 0, 0, 0, 0, required_padding), mode="constant", value=0)

    torch_weights = torch.randn((hidden_size, num_channels, patch_size, patch_size), dtype=torch.bfloat16).float()
    torch_weights = torch.nn.functional.pad(torch_weights, (0, 0, 0, 0, 0, required_padding), mode="constant", value=0)
    torch_bias = torch.randn((1, 1, 1, hidden_size), dtype=torch.bfloat16).float()

    torch_output_tensor = (
        torch.nn.functional.conv2d(
            torch_input_tensor,
            torch_weights,
            bias=torch_bias.reshape(-1),
            stride=(stride, stride),
            padding=(patch_size // 2, patch_size // 2),
            dilation=(1, 1),
            groups=1,
        )
        .flatten(start_dim=2, end_dim=3)
        .transpose(1, 2)
    )

    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn_bias = ttnn.from_torch(
        torch_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    conv_config = ttnn.Conv2dConfig(
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_split_reader=enable_split_reader,
    )

    ttnn_output_tensor = ttnn.conv2d(
        input_tensor=ttnn_input_tensor,
        weight_tensor=ttnn_weights,
        bias_tensor=ttnn_bias,
        in_channels=padded_num_channels,
        out_channels=hidden_size,
        batch_size=batch_size,
        input_height=height,
        input_width=width,
        kernel_size=(patch_size, patch_size),
        stride=(stride, stride),
        device=device,
        padding=(patch_size // 2, patch_size // 2),
        conv_config=conv_config,
        dtype=ttnn.bfloat8_b,
    )
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    _, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output_tensor[0], pcc=0.99)
    print(pcc_message)

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "input_channels, output_channels, input_height, input_width, kernel_height, kernel_width, stride_height, stride_width",
    [
        (3, 32, 224, 224, 16, 16, 16, 16),
        (3, 32, 224, 224, 32, 32, 32, 32),
        (3, 32, 224, 224, 16, 16, 2, 2),
        (3, 32, 224, 224, 7, 7, 2, 2),
        (3, 32, 224, 224, 6, 6, 2, 2),
        (3, 32, 1280, 1280, 6, 6, 2, 2),
        (3, 32, 512, 672, 16, 16, 16, 16),
        (3, 32, 512, 672, 32, 32, 32, 32),
        (320, 32, 224, 224, 16, 16, 16, 16),
        (320, 32, 224, 224, 32, 32, 32, 32),
        (320, 32, 512, 672, 16, 16, 16, 16),
        (320, 32, 512, 672, 32, 32, 32, 32),
    ]
)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("has_bias", [True, False])
def test_conv2d_with_fold(
    device,
    torch_tensor_map,
    batch_size,
    input_channels,
    output_channels,
    input_height,
    input_width,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    input_layout,
    has_bias,
):
    run_conv(
        device=device,
        torch_tensor_map=torch_tensor_map,
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc = True,
        output_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=kernel_height,
        filter_width=kernel_width,
        stride_h=stride_height,
        stride_w=stride_width,
        padding=(0, 0),
        config_override=None,
        input_layout=input_layout,
        has_bias=has_bias,
        enable_kernel_stride_folding=True,
    )

@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant, shard_layout, activation",
    (
        (1, 320, 320, 20, 20, 3, 3, 1, 1, 1, 1, 320, True, None, False, BS, ""),
        (1, 320, 320, 40, 40, 3, 3, 1, 1, 1, 1, 320, True, None, False, HS, "silu"),
        (1, 320, 320, 80, 80, 3, 3, 2, 2, 1, 1, 320, True, None, False, HS, "silu"),
        (1, 640, 640, 20, 20, 3, 3, 1, 1, 1, 1, 640, True, None, False, BS, "silu"),
        (1, 640, 640, 40, 40, 3, 3, 1, 1, 1, 1, 640, True, None, False, BS, "silu"),
        (1, 640, 640, 40, 40, 3, 3, 2, 2, 1, 1, 640, True, None, False, BS, ""),
        (1, 640, 640, 80, 80, 3, 3, 2, 2, 1, 1, 640, True, None, False, None, ""),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_yolov10x(
    device,
    torch_tensor_map,

    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
    memory_config,
    shard_layout,
    activation,
):
    run_conv(
    device=device,
    torch_tensor_map=torch_tensor_map,
    math_fidelity=math_fidelity,
    output_dtype=ttnn.bfloat8_b,
    weights_dtype=weights_dtype,
    batch_size=batch_size,
    output_channels=output_channels,
    input_channels=input_channels,
    input_height=input_height,
    input_width=input_width,
    filter_height=filter_height,
    filter_width=filter_width,
    stride_h=stride_h,
    stride_w=stride_w,
    padding=(pad_h, pad_w),
    config_override=config_override,
    groups=groups,
    output_layout=output_layout,
    memory_config=memory_config,
    fp32_accum=False,
    packer_l1_acc=False,
    input_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    shard_layout = shard_layout,
    activation = activation,
    math_approx_mode = False,
)

@pytest.mark.parametrize(
    "batch, input_height, input_width, weights_dtype, output_dtype, groups, padding, dilation",
    (
        (1, 64, 64, ttnn.bfloat8_b, ttnn.bfloat16, 1, (0, 0, 0, 0), (1, 1)),
    ),
)
@pytest.mark.parametrize( "input_channels, output_channels", [
    (512, 512),
    (64,64)
    ])
@pytest.mark.parametrize("kernel,", [
    (1, 1),
    (2, 2)]
    )
@pytest.mark.parametrize("stride", [
    (1, 1),
    (2, 2)]
)
@pytest.mark.parametrize(
    "input_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "output_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)

@pytest.mark.parametrize(
    "shard_layout",
    [
        None,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ]
)
@pytest.mark.parametrize(
    "enable_fenable_kernel_stride_folding",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("slice_type, num_slices", [
    (None,1), # no slicing
    (SliceHeight, 2),
])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_act_dealloc(
    device,
    torch_tensor_map,
    batch,
    input_channels,
    output_channels,
    input_height,
    input_width,
    weights_dtype,
    output_dtype,
    groups,
    kernel,
    stride,
    padding,
    dilation,
    input_layout,
    output_layout,
    shard_layout,
    enable_fenable_kernel_stride_folding,
    slice_type,
    num_slices,
):
    if shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED and input_layout == ttnn.ROW_MAJOR_LAYOUT and stride == (1,1):
        pytest.skip("Block sharded conv2d does not support input with row major layout")
    if enable_fenable_kernel_stride_folding and stride != kernel:
        pytest.skip("Kernel stride folding is only supported when stride == kernel size")
    if enable_fenable_kernel_stride_folding and shard_layout is not None:
        pytest.skip("Kernel stride folding is not supported for non L1 inputs")
    if (input_channels > 64 and shard_layout is not None) or kernel > stride:
        pytest.skip("OOM for this given core_grid (expected)")
    if slice_type is not None and shard_layout is not None:
        pytest.skip("Slicing is not supported for sharded conv2d")
    if slice_type is not None and enable_fenable_kernel_stride_folding:
        pytest.skip("Skip slicing when folding is enabled")
    if slice_type is SliceHeight and kernel == (1,1) and stride == (1,1) and input_channels == 512:
        pytest.skip("Skip due to assertion in conv2d implementation in tilize op")

    input_shape = (batch, input_channels, input_height, input_width)
    weight_shape = (output_channels, input_channels // groups, kernel[0], kernel[1])
    torch_input_tensor = randomize_torch_tensor(torch_tensor_map, input_shape)
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    torch_weight_tensor = randomize_torch_tensor(torch_tensor_map, weight_shape)

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        ttnn.bfloat16 if weights_dtype == ttnn.bfloat16 else ttnn.float32,
    )

    input_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores = 2
        input_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(input_height*input_width*batch // num_cores, input_channels),
                core_grid=ttnn.num_cores_to_corerangeset(target_num_cores=num_cores, grid_size=device.compute_with_storage_grid_size(), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
        )
    elif shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        num_cores = 4
        input_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(input_height*input_width*batch // 2, input_channels // 2),
                core_grid=ttnn.CoreGrid(x=2,y=2),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
        )
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        output_dtype,
        layout=input_layout,
        device=device, # set the tensor to be on device because is_allocated() returns true for host tensors
        memory_config=input_mem_cfg,
    )

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        deallocate_activation=True,
        output_layout=output_layout,
        enable_kernel_stride_folding=enable_fenable_kernel_stride_folding,
    )
    slice_config = None
    if slice_type is not None:
        slice_config = ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        )
    _ = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=None,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        batch_size=batch,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        groups=groups,
        slice_config=slice_config,
        dtype=output_dtype,
    )
    assert not tt_input_tensor.is_allocated(), "Input tensor is allocated"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout",
    (
        (32, 32, 8, 8, WS),
        (16, 16, 8, 4, BS),
    ),
)
@pytest.mark.parametrize(
    "filter, padding",
    [
        [3, (1, 1)],
    ],
)
def test_conv_single_core(
    device,
    torch_tensor_map,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    filter,
    padding,
):

    run_conv(
        device = device,
        torch_tensor_map = torch_tensor_map,
        math_fidelity = ttnn.MathFidelity.HiFi4,
        output_dtype = ttnn.bfloat16,
        weights_dtype = ttnn.bfloat16,
        batch_size = 1,
        output_channels = output_channels,
        input_channels = input_channels,
        input_height = input_height,
        input_width = input_width,
        filter_height=filter,
        filter_width=filter,
        stride_h = 1,
        stride_w = 1,
        padding = padding,
        shard_layout=shard_layout,
        input_dtype=ttnn.bfloat16,
        config_override = None,
    )
