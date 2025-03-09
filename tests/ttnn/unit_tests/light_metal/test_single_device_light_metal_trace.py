# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import inspect
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


# Helper function to write binary file to a path
def write_binary_to_file(binary_data, dir, filename):
    bin_file = dir / f"{filename}.bin"
    logger.info(f"Writing light metal binary {binary_data.size()} bytes to {bin_file}")
    binary_data.save_to_file(str(bin_file))


# Helper function to reset a device, and run light metal binary.
def reset_device_and_replay_binary(reset_device, device, binary_data):
    logger.info("Resetting device, and running Light Metal Binary via LightMetalReplay now...")
    device = reset_device(device)
    lm_replay = ttnn.LightMetalReplay.create(binary_data, device)
    success = lm_replay.run()

    if success == 0:
        logger.info("Light Metal Binary failed to execute or encountered errors.")
    else:
        logger.info("Light Metal Binary executed successfully!")

    assert success == 1, "Light Metal Binary replay failure"


# Test that buffer can be written to and read back same way during capture and replay.
@pytest.mark.parametrize("shape", [[1, 1, 2, 2], [1, 1, 32, 32], (1, 3, 256, 512)])
def test_buffer_read_write(device, reset_device, shape, tmp_path):
    ttnn.light_metal_begin_capture()

    # Create single buffer on device with random values:
    input_0_torch = torch.rand(tuple(shape), dtype=torch.float32)
    input0 = ttnn.from_torch(input_0_torch, dtype=ttnn.float32)
    input_0_dev = ttnn.to_device(input0, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Read back the buffer to host and print for debug
    input_0_host = ttnn.to_torch(input_0_dev)
    logger.info("input_0_dev: shape: {} dtype: {} data: {}", input_0_host.shape, input_0_host.dtype, input_0_host)

    # End Light Metal capture, write binary and replay from binary.
    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)


# Simple bringup single op test to see if everything uses host APIs and if it can be light-metal traced.
@pytest.mark.parametrize("shape", [[1, 3, 1024, 1024], (1, 1, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("enable_async", [False])
@pytest.mark.parametrize("blocking", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}])
def test_single_op_test_light_metal_capture(device, reset_device, shape, enable_async, blocking, tmp_path):
    ttnn.light_metal_begin_capture()

    device.enable_async(enable_async)
    device.enable_program_cache()

    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    output_tensor = ttnn.add(input_0_dev, input_1_dev)

    ttnn_torch_output_tensor = ttnn.to_torch(output_tensor)
    # logger.info("ttnn_torch_output_tensor: {}".format(ttnn_torch_output_tensor))

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)

    # TODO (kmabee) - Add capture vs replay verification check


# Simple bringup, multiple ops in chain
@pytest.mark.parametrize("shape", [[1, 3, 1024, 1024], (1, 1, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("enable_async", [False])
@pytest.mark.parametrize("blocking", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}])
def test_chain_op_test_light_metal_capture(device, reset_device, shape, enable_async, blocking, tmp_path):
    ttnn.light_metal_begin_capture()

    device.enable_async(enable_async)
    device.enable_program_cache()
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # Op chain to be traced
    def run_op_chain(input_0, input_1):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))

    output_tensor = run_op_chain(input_0_dev, input_1_dev)

    ttnn_torch_output_tensor = ttnn.to_torch(output_tensor)
    # logger.info("ttnn_torch_output_tensor: {}".format(ttnn_torch_output_tensor))

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)

    # TODO (kmabee) - Add capture vs replay verification check


# TODO (kmabee) - Add more tests, including version with metal-trace.


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_ttnn_conv2d(batch_size, device, reset_device, tmp_path):
    ttnn.light_metal_begin_capture()

    # Create input tensor in NCHW
    torch_input_tensor_nchw = torch.rand((batch_size, 3, 32, 32), dtype=torch.bfloat16)
    # Permute to NHWC, as expected by TTNN
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))  # (N, H, W, C)
    # Convert input tensor to TTNN tensor with proper dtype.
    input_tt = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    # Create weight tensor in OIHW format:
    # Weight shape: (out_channels, in_channels, filter_height, filter_width)
    weight_torch = torch.rand((8, 3, 3, 3), dtype=torch.bfloat16)
    # Create bias tensor shaped as (1, 1, 1, out_channels) as used in the reference test.
    bias_torch = torch.rand((1, 1, 1, 8), dtype=torch.bfloat16)

    weight_tt = ttnn.from_torch(weight_torch, ttnn.bfloat16)
    bias_tt = ttnn.from_torch(bias_torch, ttnn.bfloat16)

    # Set up conv2d configuration similar to the reference test.
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
        transpose_shards=True,
    )

    # Initialize compute config using an appropriate MathFidelity enum.
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # Use a default L1 memory configuration.
    memory_config = ttnn.L1_MEMORY_CONFIG

    # Note: We must provide explicit dimensions.
    conv_kwargs = {
        "in_channels": 3,
        "out_channels": 8,
        "batch_size": batch_size,
        "input_height": 32,  # from our NCHW input tensor
        "input_width": 32,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
    }

    # Run conv2d using the public API (which internally calls the same routines as in the reference test).
    output_tt, [out_height, out_width] = ttnn.conv2d(
        input_tensor=input_tt,
        weight_tensor=weight_tt,
        bias_tensor=bias_tt,
        **conv_kwargs,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        memory_config=memory_config,
        return_output_dim=True,
    )

    # Read back the output tensor from device to host.
    output_host = ttnn.to_torch(output_tt)
    print("Output tensor (host):")
    print("  Shape:", output_host.shape)
    print("  Stats: min =", output_host.min(), "max =", output_host.max(), "mean =", output_host.mean())

    # Verify that the last dimension (channels) matches our expected output channels.
    assert output_host.shape[-1] == 8, "Output channel dimension mismatch."

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_ttnn_conv2d_complex(batch_size, device, reset_device, tmp_path):
    ttnn.light_metal_begin_capture()

    # --- First Conv2D Op ---
    # Create input tensor in NCHW and convert to NHWC for TTNN.
    torch_input_tensor_nchw = torch.rand((batch_size, 3, 32, 32), dtype=torch.bfloat16)
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))  # (N, H, W, C)
    input_tt = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    # Create weights & bias for first conv2d op in OIHW and (1,1,1,C) formats.
    weight1_torch = torch.rand((8, 3, 3, 3), dtype=torch.bfloat16)
    bias1_torch = torch.rand((1, 1, 1, 8), dtype=torch.bfloat16)
    weight1_tt = ttnn.from_torch(weight1_torch, ttnn.bfloat16)
    bias1_tt = ttnn.from_torch(bias1_torch, ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
        transpose_shards=True,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    memory_config = ttnn.L1_MEMORY_CONFIG

    conv_kwargs1 = {
        "in_channels": 3,
        "out_channels": 8,
        "batch_size": batch_size,
        "input_height": 32,
        "input_width": 32,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
    }

    # Run the first conv2d op.
    output1_tt, [out_height1, out_width1] = ttnn.conv2d(
        input_tensor=input_tt,
        weight_tensor=weight1_tt,
        bias_tensor=bias1_tt,
        **conv_kwargs1,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        memory_config=memory_config,
        return_output_dim=True,
    )

    # Read back and print the first conv2d output.
    output1_host = ttnn.to_torch(output1_tt)
    print("First conv2d output:")
    print("  Shape:", output1_host.shape)
    print("  Stats: min =", output1_host.min(), "max =", output1_host.max(), "mean =", output1_host.mean())

    # --- Residual Addition ---
    # Simulate a residual connection by adding the conv output to itself.
    added_tt = ttnn.add_(output1_tt, output1_tt)
    added_host = ttnn.to_torch(added_tt)
    print("After addition (residual):")
    print("  Shape:", added_host.shape)
    print("  Stats: min =", added_host.min(), "max =", added_host.max(), "mean =", added_host.mean())

    # --- Second Conv2D Op ---
    # Create weights & bias for a second conv2d op.
    # Now input channels are 8 (from the first op) and let's output 16 channels.
    weight2_torch = torch.rand((16, 8, 3, 3), dtype=torch.bfloat16)
    bias2_torch = torch.rand((1, 1, 1, 16), dtype=torch.bfloat16)
    weight2_tt = ttnn.from_torch(weight2_torch, ttnn.bfloat16)
    bias2_tt = ttnn.from_torch(bias2_torch, ttnn.bfloat16)

    conv_kwargs2 = {
        "in_channels": 8,
        "out_channels": 16,
        "batch_size": batch_size,
        "input_height": out_height1,
        "input_width": out_width1,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
    }

    # Run the second conv2d op on the result of the addition.
    output2_tt, [out_height2, out_width2] = ttnn.conv2d(
        input_tensor=added_tt,
        weight_tensor=weight2_tt,
        bias_tensor=bias2_tt,
        **conv_kwargs2,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        memory_config=memory_config,
        return_output_dim=True,
    )

    output2_host = ttnn.to_torch(output2_tt)
    print("Second conv2d output:")
    print("  Shape:", output2_host.shape)
    print("  Stats: min =", output2_host.min(), "max =", output2_host.max(), "mean =", output2_host.mean())

    # End capture and replay.
    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)
