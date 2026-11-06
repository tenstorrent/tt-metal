# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from loguru import logger
from ttnn.operations.activations import get_golden_function_for_activation

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, is_wormhole_b0
from tt_py_test_utils_common.utils_for_testing import assert_equal, check_with_pcc_without_tensor_printout

BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def randomize_torch_tensor(
    torch_tensor_map,
    tensor_shape,
    generate_positive_numbers=False,
    dtype=torch.bfloat16,
):
    if generate_positive_numbers:
        torch_tensor = torch.randn(tensor_shape, dtype=dtype).float()
        torch_tensor = torch.abs(torch_tensor)
        return torch_tensor
    else:
        cache_key = (tensor_shape, dtype)
        if cache_key in torch_tensor_map.keys():
            torch_tensor = torch_tensor_map[cache_key]
        else:
            torch_tensor = torch.randn(tensor_shape, dtype=dtype).float()
            torch_tensor_map[cache_key] = torch_tensor

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
    has_bias=True,
    shard_layout=None,
    auto_shard=False,
    memory_config=None,
    input_mesh_mapper=None,
    weight_mesh_mapper=None,
    output_mesh_composer=None,
    activation=None,
    run_twice=False,
    fast_compare=False,
    use_dram_slicing=False,
    slice_config=None,
    enable_kernel_stride_folding=False,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
    bs_full_inner_dim=False,
    sharded_cfg=None,
    throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    enable_activation_reuse=False,
    config_tensors_in_dram=False,
    custom_pcc=None,
    force_split_reader=None,
    core_grid=None,
    perf_test_mode=False,
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
        logger.debug(f"Using {num_devices} devices for this test")
    else:
        total_batch_size = batch_size

    if type(throttle_level) == int:
        throttle_level = ttnn.ThrottleLevel(throttle_level)

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
    sqrt_act_function = activation is not None and activation.op_type == ttnn.UnaryOpType.SQRT
    torch_input_tensor_nchw = randomize_torch_tensor(
        torch_tensor_map, conv_input_shape, generate_positive_numbers=sqrt_act_function
    )

    torch_weight_tensor = randomize_torch_tensor(
        torch_tensor_map, conv_weight_shape, generate_positive_numbers=sqrt_act_function
    )
    torch_bias_tensor = (
        randomize_torch_tensor(torch_tensor_map, conv_bias_shape, generate_positive_numbers=sqrt_act_function) * 10
        if has_bias
        else None
    )

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    if not perf_test_mode:
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
        # Handle UnaryWithParam activation type with direct enum mapping
        act_func = get_golden_function_for_activation(activation)
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
    if slice_config is not None:
        use_dram_slicing = True
    requires_device_placement = input_dtype == ttnn.bfloat8_b or sharded_cfg is not None or use_dram_slicing

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        input_dtype,
        mesh_mapper=input_mesh_mapper,
        layout=input_layout,
        device=device if requires_device_placement else None,
    )

    if sharded_cfg:
        tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_cfg)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        shard_layout=shard_layout if not auto_shard else None,
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        output_layout=output_layout,
        activation=activation,
        transpose_shards=transpose_shards,
        enable_kernel_stride_folding=enable_kernel_stride_folding,
        full_inner_dim=bs_full_inner_dim,
        enable_activation_reuse=enable_activation_reuse,
        config_tensors_in_dram=config_tensors_in_dram,
        force_split_reader=force_split_reader,
        core_grid=core_grid,
        override_output_sharding_config=core_grid is not None,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_approx_mode=math_approx_mode,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        throttle_level=throttle_level,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override and not auto_shard:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "reshard_if_not_optimal" in config_override:
        conv_config.reshard_if_not_optimal = config_override["reshard_if_not_optimal"]

    if not use_dram_slicing:
        slice_config = ttnn.Conv2dL1FullSliceConfig

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
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=output_dtype,
        slice_config=slice_config,
    )

    if run_twice and not perf_test_mode:
        del tt_output_tensor_on_device
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
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=output_dtype,
            slice_config=slice_config,
        )
    ttnn.synchronize_device(device)

    if not perf_test_mode:
        tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
        out = ttnn.to_torch(tt_output_tensor, mesh_composer=output_mesh_composer)
        # out is in row major layout and NHWC shape
        # NHWC to NCHW
        out = out.reshape(total_batch_size, out_height, out_width, out.shape[-1])
        out = out[:, :, :, :output_channels]

        ref = torch.permute(ref, (0, 2, 3, 1))

        if custom_pcc is not None:
            pcc = custom_pcc
        else:
            if not fp32_accum:
                pcc = 0.985
                if input_channels * filter_height * filter_width > 10000:
                    pcc = 0.97
            elif math_fidelity == ttnn.MathFidelity.LoFi and output_dtype == ttnn.bfloat8_b:
                pcc = 0.996
            elif activation is not None and activation.op_type == ttnn.UnaryOpType.SIGMOID:
                # Scale down PCC for sigmoid.
                # The sigmoid function relies on the exp approximation, which can introduce small discrepancies in output values.
                # This necessitates a slightly lower PCC threshold, similar to the adjustment for tanh.
                pcc = 0.995
            else:
                pcc = 0.997

            # Check if activation is tanh
            is_tanh = activation is not None and activation.op_type == ttnn.UnaryOpType.TANH
            if is_tanh:
                # Scale down PCC for tanh.
                # tanh has a range of -1 to 1. So discrepancies in output values which are close to 0 tend to disproportionately affect the PCC.
                pcc = pcc * 0.99

        torch.set_printoptions(precision=3, sci_mode=False)
        if fast_compare:
            if (
                fp32_accum
                and output_dtype != ttnn.bfloat8_b
                and input_dtype != ttnn.bfloat8_b
                and weights_dtype != ttnn.bfloat8_b
            ):
                threshold = 3e-1 + 5e-3 * math.log(input_channels * filter_height * filter_width, 2)
            else:
                threshold = 3e-1 + 1e-1 * math.log(input_channels * filter_height * filter_width, 2)
            logger.debug(f"Threshold: {threshold}")
            diff = torch.abs(ref - out) / ref.abs().mean()
            assert torch.all(diff < threshold), f"Max diff: {diff.max()}, Threshold: {threshold} "
        else:
            passing, pcc_msg = check_with_pcc_without_tensor_printout(out, ref, pcc=pcc)
            logger.debug(f"PCC = {pcc_msg}. Threshold = {pcc}")
            assert passing, pcc_msg
            if pcc_msg == 1:
                # Conv2d with randomized input and weights can't legitimately return PCC of 1
                # Edge case can happen rarely if activation function like ReLU zeros out all values
                # In this case, tensors have to match.
                assert_equal(out, ref)

        if memory_config:
            output_memory_config = ttnn.get_memory_config(tt_output_tensor_on_device)
            logger.debug(f"Output Memory Config : {output_memory_config}")
            assert output_memory_config == memory_config


def run_sdxl_conv2d(
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
    act_db,
    w_db,
    perf_test_mode=False,
):
    core_grid = (
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(4, 7),
                ),
            }
        )
        if shard_layout == BS and is_wormhole_b0()
        else None
    )
    # Skip all on N300
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")

    config_override = {}
    config_override["act_block_h"] = act_block_h_override
    config_override["act_block_w_div"] = act_block_w_div
    config_override["reshard_if_not_optimal"] = True

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
        input_layout=ttnn.TILE_LAYOUT if output_dtype == ttnn.bfloat8_b else None,
        enable_act_double_buffer=act_db,
        enable_weights_double_buffer=w_db,
        core_grid=core_grid,
        perf_test_mode=perf_test_mode,
    )


def run_sdxl_conv2d_vae(
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
    deallocate_activation,
    slice_type,
    num_slices,
    act_block_h_override,
    throttle,
    auto_slice,
    perf_test_mode=False,
):
    # Skip all on N300
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 grid for wormhole_b0")
    # Skip specific test case for Blackhole devices
    if is_blackhole() and (batch, input_channels, output_channels, input_height, input_width, weights_dtype) == (
        1,
        4,
        4,
        128,
        128,
        ttnn.bfloat8_b,
    ):
        pytest.skip("Skipping this test case for Blackhole devices due to PCC issue, tracked in ISSUE-24463")

    config_override = {}
    config_override["act_block_h"] = act_block_h_override
    config_override["act_block_w_div"] = 1
    if auto_slice:
        use_dram_slicing = True
        slice_config = None
    elif slice_type is None:
        slice_config = None
        use_dram_slicing = False
    else:
        slice_config = ttnn.Conv2dSliceConfig(slice_type=slice_type, num_slices=num_slices)
        use_dram_slicing = True

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
        shard_layout=shard_layout,
        memory_config=None,
        input_mesh_mapper=None,
        weight_mesh_mapper=None,
        output_mesh_composer=None,
        use_dram_slicing=use_dram_slicing,
        slice_config=slice_config,
        input_layout=ttnn.TILE_LAYOUT,
        enable_act_double_buffer=False,  # TODO: this is set to true in SDXL, need to adapt tests
        throttle_level=throttle,
        perf_test_mode=perf_test_mode,
    )
