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


# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from loguru import logger
from ttnn.operations.conv2d import get_torch_act_func_from_string

import ttnn
from models.demos.yolov7.common import YOLOV7_L1_SMALL_SIZE
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout


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
@pytest.mark.parametrize("fp32_accum", [True])
@pytest.mark.parametrize("packer_l1_acc", [True])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("act_block_h", [32 * 10, 32 * 4])
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
        "nhw": 1,
        "input_height": 80,
        "input_width": 80,
        "input_channels": 128,
        "output_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
    }
    total_nhw = _setup_device_config(
        device, conv_params["nhw"], input_mesh_mapper, weight_mesh_mapper, output_mesh_composer
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
    final_output = _process_output(tt_output, total_nhw, conv_params["output_channels"], output_mesh_composer)

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


def _setup_device_config(device, nhw, input_mesh_mapper, weight_mesh_mapper, output_mesh_composer):
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
        total_nhw = num_devices * nhw
        logger.info(f"Using {num_devices} devices for this test")
        return total_nhw
    else:
        return nhw


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
        nhw=conv_params["nhw"],
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


def _process_output(tt_output, total_nhw, output_channels, output_mesh_composer):
    """Process TT output tensor to PyTorch format"""
    out_height, out_width = tt_output["output_dims"]

    tt_output_tensor = ttnn.from_device(tt_output["tensor_on_device"])
    out = ttnn.to_torch(tt_output_tensor, mesh_composer=output_mesh_composer)

    # Reshape and slice output (NHWC format)
    out = out.reshape(total_nhw, out_height, out_width, out.shape[-1])
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


@pytest.mark.parametrize("in0_block_w", [1, 2, 4, 8])
@pytest.mark.parametrize("out_subblock", [[1, 4]])
@pytest.mark.parametrize("out_block", [[1, 4]])
@pytest.mark.parametrize("num_cores", [62, 64])  # original value 62
@pytest.mark.parametrize("pad_to_multiple_of_2048", [False])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_bias_dtype", [ttnn.bfloat8_b])  # original value ttnn.bfloat16
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "compute_config_params", [[ttnn.MathFidelity.LoFi, True, False, False, False]]
)  # original value [ttnn.MathFidelity.LoFi, False, False, True, False]
def test_linear_conv_height_sharded(
    device,
    in0_block_w,
    out_subblock,
    out_block,
    num_cores,
    pad_to_multiple_of_2048,
    input_dtype,
    weights_bias_dtype,
    output_dtype,
    compute_config_params,
):
    torch.manual_seed(0)
    nhw = 6400
    input_channels = 512
    output_channels = 512

    if pad_to_multiple_of_2048:
        nhw = ((nhw + 2047) // 2048) * 2048

    input_shape = [1, 1, nhw, input_channels]
    torch_input_tensor = torch.randn([1, 1, nhw, input_channels], dtype=torch.bfloat16)  # Original size
    torch_weight_tensor = torch.randn([1, 1, output_channels, input_channels], dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn([1, 1, 1, output_channels], dtype=torch.bfloat16)
    torch_out_golden_tensor = torch.nn.functional.linear(
        torch_input_tensor[0, 0, :, :], torch_weight_tensor[0, 0, :, :], bias=torch_bias_tensor[0, 0, :, :]
    )
    torch_out_golden_tensor = torch.nn.functional.silu(torch_out_golden_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    tt_weight_tensor = ttnn.from_torch(
        torch.permute(torch_weight_tensor, (0, 1, 3, 2)),
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor,
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=compute_config_params[0],
        math_approx_mode=compute_config_params[1],
        fp32_dest_acc_en=compute_config_params[2],
        packer_l1_acc=compute_config_params[3],
        dst_full_sync_en=compute_config_params[4],
    )
    grid_size = (8, 8)
    per_core_M = (nhw + 32 * num_cores - 1) // (32 * num_cores)
    per_core_N = output_channels // 32
    print(f"per_core_M: {per_core_M}, per_core_N: {per_core_N}")

    matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock[0],
        out_subblock_w=out_subblock[1],
        out_block_h=out_block[0],
        out_block_w=out_block[1],
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        mcast_in0=False,
    )

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),  # 8 rows x 6 cols = 48 cores
            ),
        }
    )

    # shard height is equal to nhw / num_cores but rounded to the next multiple of 32
    shard_height = (nhw + num_cores * 32 - 1) // (num_cores * 32) * 32
    x = tt_input_tensor
    in0_shard_shape = [shard_height, input_channels]
    print(f"shard_height: {shard_height}, in0_shard_shape: {in0_shard_shape}")
    in0_shard_spec = ttnn.ShardSpec(shard_grid, in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    height_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )
    x = ttnn.to_memory_config(x, height_sharded_mem_config)

    # Create output memory config to match target
    out_shard_shape = [shard_height, output_channels]
    print(f"out_shard_shape: {out_shard_shape}")
    out_shard_spec = ttnn.ShardSpec(shard_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    tt_output_tensor_on_device = ttnn.linear(
        x,
        tt_weight_tensor,
        bias=tt_bias_tensor,
        program_config=matmul_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
        compute_kernel_config=compute_config,
    )
    print(x.padded_shape, tt_weight_tensor.padded_shape, tt_bias_tensor.padded_shape)
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_out_golden_tensor, torch_output_tensor[0, 0, :, :], pcc=0.99)


@pytest.mark.parametrize("in0_block_w", [4])
@pytest.mark.parametrize("out_subblock", [[5, 1]])
@pytest.mark.parametrize("out_block", [[25, 1]])
@pytest.mark.parametrize("num_cores", [62, 64])  # original value 62
@pytest.mark.parametrize("pad_to_multiple_of_2048", [False])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_bias_dtype", [ttnn.bfloat8_b])  # original value ttnn.bfloat16
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "compute_config_params", [[ttnn.MathFidelity.LoFi, True, False, False, False]]
)  # original value [ttnn.MathFidelity.LoFi, False, False, True, False]
def test_linear_conv_dram(
    device,
    in0_block_w,
    out_subblock,
    out_block,
    num_cores,
    pad_to_multiple_of_2048,
    input_dtype,
    weights_bias_dtype,
    output_dtype,
    compute_config_params,
):
    torch.manual_seed(0)
    nhw = 6400
    input_channels = 512
    output_channels = 512

    if pad_to_multiple_of_2048:
        nhw = ((nhw + 2047) // 2048) * 2048

    input_shape = [1, 1, nhw, input_channels]
    torch_input_tensor = torch.randn([1, 1, nhw, input_channels], dtype=torch.bfloat16)  # Original size
    torch_weight_tensor = torch.randn([1, 1, output_channels, input_channels], dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn([1, 1, 1, output_channels], dtype=torch.bfloat16)
    torch_out_golden_tensor = torch.nn.functional.linear(
        torch_input_tensor[0, 0, :, :], torch_weight_tensor[0, 0, :, :], bias=torch_bias_tensor[0, 0, :, :]
    )
    torch_out_golden_tensor = torch.nn.functional.silu(torch_out_golden_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    tt_weight_tensor = ttnn.from_torch(
        torch.permute(torch_weight_tensor, (0, 1, 3, 2)),
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor,
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=compute_config_params[0],
        math_approx_mode=compute_config_params[1],
        fp32_dest_acc_en=compute_config_params[2],
        packer_l1_acc=compute_config_params[3],
        dst_full_sync_en=compute_config_params[4],
    )
    grid_size = (8, 8)
    per_core_M = (nhw + 32 * grid_size[0] - 1) // (32 * grid_size[0])
    per_core_N = (output_channels + 32 * grid_size[1] - 1) // (32 * grid_size[1])
    print(f"per_core_M: {per_core_M}, per_core_N: {per_core_N}")

    matmul_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock[0],
        out_subblock_w=out_subblock[1],
        out_block_h=out_block[0],
        out_block_w=out_block[1],
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        ranspose_mcast=False,
    )

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),  # 8 rows x 8 cols = 64 cores
            ),
        }
    )

    tt_output_tensor_on_device = ttnn.linear(
        tt_input_tensor,
        tt_weight_tensor,
        bias=tt_bias_tensor,
        program_config=matmul_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=output_dtype,
        compute_kernel_config=compute_config,
    )
    print(tt_input_tensor.padded_shape, tt_weight_tensor.padded_shape, tt_bias_tensor.padded_shape)
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_out_golden_tensor, torch_output_tensor[0, 0, :, :], pcc=0.99)


@pytest.mark.parametrize("in0_block_w", [1, 2])
@pytest.mark.parametrize("out_subblock", [[1, 1], [7, 1]])
@pytest.mark.parametrize("out_block", [[7, 1]])
@pytest.mark.parametrize("num_cores", [64])  # original value 62
@pytest.mark.parametrize("pad_to_multiple_of_2048", [False])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_bias_dtype", [ttnn.bfloat8_b])  # original value ttnn.bfloat16
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "compute_config_params", [[ttnn.MathFidelity.LoFi, True, False, False, False]]
)  # original value [ttnn.MathFidelity.LoFi, False, False, True, False]
def test_linear_conv_block(
    device,
    in0_block_w,
    out_subblock,
    out_block,
    num_cores,
    pad_to_multiple_of_2048,
    input_dtype,
    weights_bias_dtype,
    output_dtype,
    compute_config_params,
):
    torch.manual_seed(0)
    nhw = 1600
    input_channels = 512
    output_channels = 256

    if pad_to_multiple_of_2048:
        nhw = ((nhw + 2047) // 2048) * 2048

    input_shape = [1, 1, nhw, input_channels]
    torch_input_tensor = torch.randn([1, 1, nhw, input_channels], dtype=torch.bfloat16)  # Original size
    torch_weight_tensor = torch.randn([1, 1, output_channels, input_channels], dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn([1, 1, 1, output_channels], dtype=torch.bfloat16)
    torch_out_golden_tensor = torch.nn.functional.linear(
        torch_input_tensor[0, 0, :, :], torch_weight_tensor[0, 0, :, :], bias=torch_bias_tensor[0, 0, :, :]
    )
    torch_out_golden_tensor = torch.nn.functional.silu(torch_out_golden_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    tt_weight_tensor = ttnn.from_torch(
        torch.permute(torch_weight_tensor, (0, 1, 3, 2)),
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor,
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=compute_config_params[0],
        math_approx_mode=compute_config_params[1],
        fp32_dest_acc_en=compute_config_params[2],
        packer_l1_acc=compute_config_params[3],
        dst_full_sync_en=compute_config_params[4],
    )
    grid_size = (8, 8)
    per_core_M = (nhw + 32 * grid_size[0] - 1) // (32 * grid_size[0])
    per_core_N = (output_channels + 32 * grid_size[1] - 1) // (32 * grid_size[1])
    print(f"per_core_M: {per_core_M}, per_core_N: {per_core_N}")

    matmul_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock[0],
        out_subblock_w=out_subblock[1],
        out_block_h=out_block[0],
        out_block_w=out_block[1],
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        transpose_mcast=False,
    )

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),  # 8 rows x 7 cols = 56 cores
            ),
        }
    )
    x = tt_input_tensor
    in0_shard_height = (nhw + grid_size[0] * 32 - 1) // (grid_size[0] * 32) * 32
    in0_shard_width = (input_channels + grid_size[1] * 32 - 1) // (grid_size[1] * 32) * 32
    in0_shard_shape = [in0_shard_height, in0_shard_width]
    in0_shard_spec = ttnn.ShardSpec(shard_grid, in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    height_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )
    tt_input_tensor = ttnn.to_memory_config(x, height_sharded_mem_config)
    print(x.memory_config())

    # Create output memory config to match target
    out_shard_width = (output_channels + grid_size[1] * 32 - 1) // (grid_size[1] * 32) * 32
    out_shard_shape = [in0_shard_height, out_shard_width]
    out_shard_spec = ttnn.ShardSpec(shard_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    print(output_mem_config)

    tt_output_tensor_on_device = ttnn.linear(
        tt_input_tensor,
        tt_weight_tensor,
        bias=tt_bias_tensor,
        program_config=matmul_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
        compute_kernel_config=compute_config,
    )
    print(tt_input_tensor.padded_shape, tt_weight_tensor.padded_shape, tt_bias_tensor.padded_shape)
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_out_golden_tensor, torch_output_tensor[0, 0, :, :], pcc=0.99)
