# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from loguru import logger
from ttnn.operations.conv2d import get_torch_act_func_from_string

import ttnn
from models.demos.yolov7.common import YOLOV7_L1_SMALL_SIZE
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


def randomize_torch_tensor(
    torch_tensor_map,
    tensor_shape,
    generate_positive_numbers=False,
):
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV7_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("fp32_accum", [False])
@pytest.mark.parametrize("packer_l1_acc", [False])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("act_block_h", [32 * 10, 32 * 50])
@pytest.mark.parametrize("enable_act_double_buffer", [True])
@pytest.mark.parametrize("enable_weights_double_buffer", [True])
def test_conv(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    input_dtype,
    fp32_accum,
    packer_l1_acc,
    output_layout,
    act_block_h,
    enable_act_double_buffer,
    enable_weights_double_buffer,
    math_approx_mode=True,
    transpose_shards=False,
    deallocate_activation=True,
    groups=1,
    has_bias=True,
    shard_layout=None,
    auto_shard=False,
    memory_config=None,
    input_mesh_mapper=None,
    weight_mesh_mapper=None,
    output_mesh_composer=None,
    enable_split_reader=True,
    activation="silu",
    in_place=False,
    run_twice=False,
    fast_compare=False,
    slice_config=None,
    enable_kernel_stride_folding=False,
    bs_full_inner_dim=False,
    sharded_cfg=None,
):
    # Setup test configuration
    conv_params = {
        "batch_size": 1,
        "input_height": 640,
        "input_width": 640,
        "input_channels": 16,
        "output_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
    }
    total_batch_size = _setup_device_config(
        device, conv_params["batch_size"], input_mesh_mapper, weight_mesh_mapper, output_mesh_composer
    )

    torch_tensors = _create_torch_tensors(torch_tensor_map, conv_params, has_bias)

    reference_output = _compute_reference_output(torch_tensors, conv_params, activation, groups)

    tt_tensors = _create_tt_tensors(
        torch_tensors,
        input_dtype,
        weights_dtype,
        conv_params,
        input_mesh_mapper,
        weight_mesh_mapper,
        device,
        sharded_cfg,
    )

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        activation=activation,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
        act_block_h_override=act_block_h,
        reshard_if_not_optimal=False,
        override_sharding_config=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        output_layout=output_layout,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        enable_split_reader=enable_split_reader,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        dst_full_sync_en=True,
    )

    tt_output = _run_conv2d(
        tt_tensors, conv_params, conv_config, compute_config, device, output_dtype, groups, memory_config, slice_config
    )

    # Process output and validate
    final_output = _process_output(tt_output, total_batch_size, conv_params["output_channels"], output_mesh_composer)

    _validate_results(
        final_output,
        reference_output,
        fp32_accum,
        math_fidelity,
        output_dtype,
        input_dtype,
        conv_params,
        activation,
        fast_compare,
    )

    _validate_memory_config(tt_output["tensor_on_device"], memory_config)


def _setup_device_config(device, batch_size, input_mesh_mapper, weight_mesh_mapper, output_mesh_composer):
    """Setup device configuration for single or multi-device"""
    if isinstance(device, ttnn.MeshDevice) and len(device.get_device_ids()) > 1:
        assert input_mesh_mapper is not None, "Expected mesh mapper for input tensor when running on multiple devices"
        assert (
            weight_mesh_mapper is not None
        ), "Expected mesh mapper for weight tensors when running on multiple devices"
        assert (
            output_mesh_composer is not None
        ), "Expected mesh composer for output tensor when running on multiple devices"

        num_devices = len(device.get_device_ids())
        total_batch_size = num_devices * batch_size
        logger.info(f"Using {num_devices} devices for this test")
        return total_batch_size
    else:
        return batch_size


def _create_torch_tensors(torch_tensor_map, conv_params, has_bias):
    """Create PyTorch tensors for input, weights, and bias"""
    torch.manual_seed(0)

    # Define tensor shapes
    conv_input_shape = (1, conv_params["input_channels"], conv_params["input_height"], conv_params["input_width"])
    conv_weight_shape = (
        conv_params["output_channels"],
        conv_params["input_channels"],
        conv_params["kernel_size"],
        conv_params["kernel_size"],
    )
    conv_bias_shape = (1, 1, 1, conv_params["output_channels"])

    sqrt_act_function = False

    # Create input tensor (NCHW -> NHWC)
    torch_input_tensor_nchw = randomize_torch_tensor(
        torch_tensor_map, conv_input_shape, generate_positive_numbers=sqrt_act_function
    )
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    # Create weight tensor
    torch_weight_tensor = randomize_torch_tensor(
        torch_tensor_map, conv_weight_shape, generate_positive_numbers=sqrt_act_function
    )

    # Create bias tensor
    torch_bias_tensor = None
    if has_bias:
        torch_bias_tensor = (
            randomize_torch_tensor(torch_tensor_map, conv_bias_shape, generate_positive_numbers=sqrt_act_function) * 10
        )

    return {
        "input_nchw": torch_input_tensor_nchw,
        "input_nhwc": torch_input_tensor,
        "weight": torch_weight_tensor,
        "bias": torch_bias_tensor,
    }


def _compute_reference_output(torch_tensors, conv_params, activation, groups):
    """Compute reference output using PyTorch"""
    torch_padded_input = torch.nn.functional.pad(
        torch_tensors["input_nchw"],
        (conv_params["padding"], conv_params["padding"], conv_params["padding"], conv_params["padding"]),
        mode="constant",
        value=0,
    )

    ref = torch.nn.functional.conv2d(
        torch_padded_input,
        torch_tensors["weight"],
        bias=torch_tensors["bias"].reshape(-1) if torch_tensors["bias"] is not None else None,
        stride=(conv_params["stride"], conv_params["stride"]),
        padding=(0, 0),
        dilation=(conv_params["dilation"], conv_params["dilation"]),
        groups=groups,
    )

    # Apply activation if specified
    act_func = get_torch_act_func_from_string(activation)
    if act_func:
        ref = act_func(ref)

    return ref


def _create_tt_tensors(
    torch_tensors, input_dtype, weights_dtype, conv_params, input_mesh_mapper, weight_mesh_mapper, device, sharded_cfg
):
    input_layout = ttnn.ROW_MAJOR_LAYOUT if input_dtype == ttnn.bfloat16 else ttnn.TILE_LAYOUT

    tt_weight_tensor = ttnn.from_torch(
        torch_tensors["weight"],
        ttnn.bfloat16 if weights_dtype == ttnn.bfloat16 else ttnn.float32,
        mesh_mapper=weight_mesh_mapper,
    )

    tt_bias_tensor = None
    if torch_tensors["bias"] is not None:
        tt_bias_tensor = ttnn.from_torch(
            torch_tensors["bias"],
            ttnn.bfloat16 if weights_dtype == ttnn.bfloat16 else ttnn.float32,
            mesh_mapper=weight_mesh_mapper,
        )

    requires_device_placement = input_dtype == ttnn.bfloat8_b or sharded_cfg is not None
    tt_input_tensor = ttnn.from_torch(
        torch_tensors["input_nhwc"],
        input_dtype,
        mesh_mapper=input_mesh_mapper,
        layout=input_layout,
        device=device if requires_device_placement else None,
    )

    if sharded_cfg:
        tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, sharded_cfg)

    return {"input": tt_input_tensor, "weight": tt_weight_tensor, "bias": tt_bias_tensor}


def _run_conv2d(
    tt_tensors, conv_params, conv_config, compute_config, device, output_dtype, groups, memory_config, slice_config
):
    """Run the convolution operation"""
    result = ttnn.conv2d(
        input_tensor=tt_tensors["input"],
        weight_tensor=tt_tensors["weight"],
        in_channels=conv_params["input_channels"],
        out_channels=conv_params["output_channels"],
        device=device,
        bias_tensor=tt_tensors["bias"],
        kernel_size=(conv_params["kernel_size"], conv_params["kernel_size"]),
        stride=(conv_params["stride"], conv_params["stride"]),
        padding=(conv_params["padding"], conv_params["padding"], conv_params["padding"], conv_params["padding"]),
        dilation=(conv_params["dilation"], conv_params["dilation"]),
        batch_size=conv_params["batch_size"],
        input_height=conv_params["input_height"],
        input_width=conv_params["input_width"],
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        memory_config=memory_config,
        slice_config=slice_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=output_dtype,
    )

    tensor_on_device, output_dims, weights_and_bias = result
    return {"tensor_on_device": tensor_on_device, "output_dims": output_dims, "weights_and_bias": weights_and_bias}


def _process_output(tt_output, total_batch_size, output_channels, output_mesh_composer):
    """Process TT output tensor to PyTorch format"""
    out_height, out_width = tt_output["output_dims"]

    tt_output_tensor = ttnn.from_device(tt_output["tensor_on_device"])
    out = ttnn.to_torch(tt_output_tensor, mesh_composer=output_mesh_composer)

    # Reshape and slice output (NHWC format)
    out = out.reshape(total_batch_size, out_height, out_width, out.shape[-1])
    out = out[:, :, :, :output_channels]

    return out


def _validate_results(
    output, reference, fp32_accum, math_fidelity, output_dtype, input_dtype, conv_params, activation, fast_compare
):
    """Validate the convolution results"""
    # Convert reference to NHWC format
    ref = torch.permute(reference, (0, 2, 3, 1))

    # Determine PCC threshold
    pcc = _calculate_pcc_threshold(fp32_accum, math_fidelity, output_dtype, conv_params, activation)

    torch.set_printoptions(precision=3, sci_mode=False)

    if fast_compare:
        threshold = _calculate_fast_compare_threshold(fp32_accum, output_dtype, input_dtype, conv_params)
        logger.info(f"Threshold: {threshold}")
        diff = torch.abs(ref - output) / ref.abs().mean()
        assert torch.all(diff < threshold), f"Max diff: {diff.max()}, Threshold: {threshold}"
    else:
        passing, pcc_msg = check_with_pcc_without_tensor_printout(output, ref, pcc=pcc)
        logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
        assert passing, pcc_msg
        assert pcc_msg != 1, "Conv2d with randomized input and weights can't legitimately return PCC of 1"


def _calculate_pcc_threshold(fp32_accum, math_fidelity, output_dtype, conv_params, activation):
    """Calculate PCC threshold based on configuration"""
    if not fp32_accum:
        pcc = 0.985
        filter_size = conv_params["input_channels"] * conv_params["kernel_size"] * conv_params["kernel_size"]
        if filter_size > 10000:
            pcc = 0.97
    elif math_fidelity == ttnn.MathFidelity.LoFi and output_dtype == ttnn.bfloat8_b:
        pcc = 0.996
    else:
        pcc = 0.997

    if activation == "tanh":
        # Scale down PCC for tanh activation
        pcc = pcc * 0.99

    return pcc


def _calculate_fast_compare_threshold(fp32_accum, output_dtype, input_dtype, conv_params):
    """Calculate threshold for fast comparison"""
    filter_size = conv_params["input_channels"] * conv_params["kernel_size"] * conv_params["kernel_size"]

    if fp32_accum and output_dtype != ttnn.bfloat8_b and input_dtype != ttnn.bfloat8_b:
        threshold = 3e-1 + 5e-3 * math.log(filter_size, 2)
    else:
        threshold = 3e-1 + 1e-1 * math.log(filter_size, 2)

    return threshold


def _validate_memory_config(tensor_on_device, expected_memory_config):
    """Validate memory configuration if specified"""
    if expected_memory_config:
        output_memory_config = ttnn.get_memory_config(tensor_on_device)
        logger.info(f"Output Memory Config : {output_memory_config}")
        assert output_memory_config == expected_memory_config
