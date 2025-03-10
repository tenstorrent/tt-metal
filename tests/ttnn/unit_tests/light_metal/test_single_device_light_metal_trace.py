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
    if os.getenv("LIGHTMETAL_DISABLE_RUN") == "1":
        logger.info("Skipping Light Metal Binary replay as LIGHTMETAL_DISABLE_RUN is set.")
        return

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


# This TTNN Move Op exposed issue with Buffer deallocate before use, leading to change to need
# to capture/replay buffer deallocate separate from buffer delete.
@pytest.mark.parametrize("shape", [[1, 1, 32, 32]])
@pytest.mark.parametrize("enable_async", [False])
@pytest.mark.parametrize("blocking", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}])
def test_sharded_move_op(device, reset_device, shape, enable_async, blocking, tmp_path):
    ttnn.light_metal_begin_capture()

    run_move_op(shape, device, tmp_path)

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)


# TODO (kmabee) - Add more tests, including version with metal-trace.

#####################################################################
# Conv2D Tests - These caught some issues in resnet50.              #
#####################################################################


def create_input_tt(batch_size, channels, height, width):
    """
    Create a TTNN tensor input.
    Input is generated in NCHW, then permuted to NHWC.
    """
    torch_input_nchw = torch.rand((batch_size, channels, height, width), dtype=torch.bfloat16)
    torch_input = torch.permute(torch_input_nchw, (0, 2, 3, 1))  # convert to NHWC
    return ttnn.from_torch(torch_input, ttnn.bfloat16)


def create_weight_bias_tt(out_channels, in_channels, kernel_h, kernel_w):
    """
    Create TTNN weight and bias tensors.
    We use OIHW for weights and shape (1,1,1,out_channels) for bias.
    """
    weight_torch = torch.rand((out_channels, in_channels, kernel_h, kernel_w), dtype=torch.bfloat16)
    bias_torch = torch.rand((1, 1, 1, out_channels), dtype=torch.bfloat16)
    weight_tt = ttnn.from_torch(weight_torch, ttnn.bfloat16)
    bias_tt = ttnn.from_torch(bias_torch, ttnn.bfloat16)
    return weight_tt, bias_tt


def create_common_conv_configs(device):
    """
    Create and return common conv2d configurations:
      - conv_config: general configuration for conv2d ops.
      - compute_config: device-specific compute configuration.
      - memory_config: memory configuration to use.
    """
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="relu",
        deallocate_activation=True,
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
    return conv_config, compute_config, memory_config


# Test: Single Conv2D operation. Passes.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_ttnn_conv2d_simple(batch_size, device, reset_device, tmp_path):
    ttnn.light_metal_begin_capture()

    # Create input tensor (3 channels, 32x32)
    input_tt = create_input_tt(batch_size, 3, 32, 32)

    # Create weight and bias for a 3->8 conv2d with 3x3 kernel.
    weight_tt, bias_tt = create_weight_bias_tt(8, 3, 3, 3)

    # Create common conv2d configuration.
    conv_config, compute_config, memory_config = create_common_conv_configs(device)

    conv_kwargs = {
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

    output_host = ttnn.to_torch(output_tt)
    print("Output tensor (host):")
    print("  Shape:", output_host.shape)
    print("  Stats: min =", output_host.min(), "max =", output_host.max(), "mean =", output_host.mean())

    assert output_host.shape[-1] == 8, "Output channel dimension mismatch."

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)


# Test: Loop stress test with residuals and channel growth.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_ttnn_conv2d_loop_stress(batch_size, device, reset_device, tmp_path):
    ttnn.light_metal_begin_capture()

    # Create initial input tensor (3 channels, 64x64).
    current_tt = create_input_tt(batch_size, 3, 64, 64)

    conv_config, compute_config, memory_config = create_common_conv_configs(device)

    # Loop settings.
    num_iterations = 10
    in_channels = 3
    out_channels = 8

    for i in range(num_iterations):
        weight_tt, bias_tt = create_weight_bias_tt(out_channels, in_channels, 3, 3)
        conv_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "batch_size": batch_size,
            "input_height": current_tt.shape[1],
            "input_width": current_tt.shape[2],
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
        }

        conv_output_tt, [h_out, w_out] = ttnn.conv2d(
            input_tensor=current_tt,
            weight_tensor=weight_tt,
            bias_tensor=bias_tt,
            **conv_kwargs,
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache={},
            memory_config=memory_config,
            return_output_dim=True,
        )

        conv_output_host = ttnn.to_torch(conv_output_tt)
        print(
            f"Iteration {i}: Conv output shape: {conv_output_host.shape}, "
            f"min: {conv_output_host.min()}, max: {conv_output_host.max()}, mean: {conv_output_host.mean()}"
        )

        # Attempt residual addition if shapes match.
        if i > 0:
            if conv_output_tt.shape == current_tt.shape:
                conv_output_tt = ttnn.add_(conv_output_tt, current_tt)
                print(f"Iteration {i}: Performed residual addition.")
            else:
                print(
                    f"Iteration {i}: Skipping residual addition due to shape mismatch: "
                    f"conv_output {conv_output_tt.shape} vs current {current_tt.shape}"
                )

        current_tt = conv_output_tt

        # Increase channel count every 3 iterations.
        if (i + 1) % 3 == 0:
            in_channels = out_channels
            out_channels *= 2

    final_output = ttnn.to_torch(current_tt)
    print("Final output shape:", final_output.shape)

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)


# -----------------------------------------------------------------------------
# Dummy class to extract the failing conv2d op from resnet50.
# -----------------------------------------------------------------------------


class DummyResnet50FirstConv:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        # Dummy parameters – adjust these values to match your ResNet50 config.
        self.conv1_input_channels = 3
        self.conv1_output_channels = 64
        self.conv1_input_height = 115
        self.conv1_input_width = 115
        self.conv1_kernel_size = (7, 7)
        self.conv1_stride = (2, 2)
        self.conv1_padding = (3, 3)
        self.conv1_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            activation="relu",
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
            transpose_shards=True,
        )
        # Use the default device's arch to initialize a compute config.
        self.conv1_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # Dummy fold parameters. Adjust as needed.
        self.fold_stride_h = 1
        self.fold_stride_w = 1
        self.fold_pad_c = 0
        self.fold_pad_h = 0
        self.fold_pad_w = 0
        # For simplicity, create a dummy core grid.
        self.fold_compute_grid_size = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 7))})
        self.override_fold_mem_config = ttnn.L1_MEMORY_CONFIG

        # Create dummy conv1 weight and bias tensors.
        self.conv1_weight_tensor = ttnn.from_torch(
            torch.rand(
                (
                    self.conv1_output_channels,
                    self.conv1_input_channels,
                    self.conv1_kernel_size[0],
                    self.conv1_kernel_size[1],
                ),
                dtype=torch.bfloat16,
            ),
            ttnn.bfloat16,
        )
        self.conv1_bias_tensor = ttnn.from_torch(
            torch.rand((1, 1, 1, self.conv1_output_channels), dtype=torch.bfloat16), ttnn.bfloat16
        )

    def run_first_conv(self, input_tensor, device, conv_op_cache={}):
        logger.info("==== Running dummy resnet50 first conv2d op")
        # Log the initial input.
        input_host = ttnn.to_torch(input_tensor)
        print("KCM INTERMEDIATE_0_input:", input_host)

        # KCM - this seems important too, maybe.
        # Convert the input tensor to a layout supported by fold (e.g. TILE_LAYOUT).
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)

        # KCM - This exposes failure (was incorrect storage assert without this)
        input_tensor = ttnn.to_device(input_tensor, device)

        # Run fold.
        fold_output_tensor = ttnn.fold(
            input_tensor,
            self.fold_stride_h,
            self.fold_stride_w,
            use_transpose_as_fold=True,
            pad_c=self.fold_pad_c,
            pad_h=self.fold_pad_h,
            pad_w=self.fold_pad_w,
            grid_size=self.fold_compute_grid_size,
            override_memory_config=self.override_fold_mem_config,
        )
        fold_output_host = ttnn.to_torch(fold_output_tensor)
        print("KCM INTERMEDIATE_1_fold:", fold_output_host)

        # Reshape the folded tensor.
        n, c, h, w = fold_output_tensor.shape
        fold_output_tensor = ttnn.reshape(fold_output_tensor, (1, 1, n * c * h, w))
        fold_output_host = ttnn.to_torch(fold_output_tensor)
        print("KCM INTERMEDIATE_2_reshape:", fold_output_host)

        ttnn.deallocate(input_tensor)

        logger.info("==== First conv2d op")
        conv_kwargs = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": self.batch_size,
            "input_height": self.conv1_input_height,
            "input_width": self.conv1_input_width,
            "kernel_size": self.conv1_kernel_size,
            "stride": self.conv1_stride,
            "padding": self.conv1_padding,
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": self.conv1_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.conv1_weight_tensor):
            self.conv1_weight_tensor = ttnn.prepare_conv_weights(
                weight_tensor=self.conv1_weight_tensor,
                weights_format="OIHW",
                input_memory_config=fold_output_tensor.memory_config(),
                input_layout=fold_output_tensor.get_layout(),
                has_bias=True,
                **conv_kwargs,
            )
            self.conv1_bias_tensor = ttnn.prepare_conv_bias(
                bias_tensor=self.conv1_bias_tensor,
                input_memory_config=fold_output_tensor.memory_config(),
                input_layout=fold_output_tensor.get_layout(),
                **conv_kwargs,
            )
            self.conv1_weight_tensor = ttnn.to_device(self.conv1_weight_tensor, device)
            self.conv1_bias_tensor = ttnn.to_device(self.conv1_bias_tensor, device)

            conv1_weight_tensor_host = ttnn.to_torch(self.conv1_weight_tensor)
            print("KCM INTERMEDIATE_3_weight_tensor:", conv1_weight_tensor_host)
            conv1_bias_tensor_host = ttnn.to_torch(self.conv1_bias_tensor)
            print("KCM INTERMEDIATE_4_bias_tensor:", conv1_bias_tensor_host)

        x, [x_height, x_width] = ttnn.conv2d(
            input_tensor=fold_output_tensor,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs,
            compute_config=self.conv1_compute_config,
            conv_op_cache=conv_op_cache,
            return_output_dim=True,
            return_weights_and_bias=False,
        )

        intermediate_host = ttnn.to_torch(x)
        print("KCM INTERMEDIATE_5_conv2d:", intermediate_host)

        # For debugging, exit early. (Return a dummy output so that downstream code doesn't fail.)
        # Here we return the conv2d output. In debug mode you could also return a dummy tensor.
        return x


# This test extracted ops up until first conv2d that is failing.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_resnet50_until_first_conv2d(batch_size, device, reset_device, tmp_path):
    ttnn.light_metal_begin_capture()

    # Create a dummy input tensor with shape matching the expected input.
    # Here we assume the expected input face shape is (batch, 3, 115, 115)
    input_tt = ttnn.from_torch(torch.rand((batch_size, 3, 115, 115), dtype=torch.bfloat16), ttnn.bfloat16)

    # Instantiate the dummy ResNet50 first conv handler.
    dummy_resnet = DummyResnet50FirstConv(batch_size, device)

    # Run the first conv2d op.
    x = dummy_resnet.run_first_conv(input_tt, device)

    # In debug mode, we want to exit early. For downstream compatibility,
    # we return a dummy tensor with shape (batch, 1000). (This is just a hack.)
    debug_mode = True
    if debug_mode:
        # Create a dummy tensor on host with the expected final shape.
        dummy_out = torch.zeros((x.shape[0], 1000), dtype=torch.bfloat16)
        # Optionally, convert to a TTNN tensor.
        final_output = ttnn.from_torch(dummy_out, ttnn.bfloat16)
    else:
        final_output = x

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)

    output_host = ttnn.to_torch(final_output)
    print("Final output shape:", output_host.shape)


# This test aims to extract just the conv2d op from resnet that was failing.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
def test_resnet50_first_conv2d_only(batch_size, device, reset_device, tmp_path):
    ttnn.light_metal_begin_capture()

    # Create a dummy activation tensor that represents the already-folded activation.
    # In production, the folded activation (after fold+reshape) has shape [1, 1, 211600, 16].
    # For simplicity, we simulate a small unfolded activation with shape (1, 1, 100, 16).
    # That means the activation matrix’s width is 16.
    dummy_activation = torch.rand((batch_size, 1, 100, 16), dtype=torch.bfloat16)
    act_tt = ttnn.from_torch(dummy_activation, ttnn.bfloat16)
    act_tt = ttnn.to_device(act_tt, device)

    # Use production conv_kwargs (as printed in your debug output):
    #   in_channels: 16, out_channels: 64, batch_size: 16, input_height: 115, input_width: 115,
    #   kernel_size: (4,4), stride: (1,1), padding: (0,0), dilation: (1,1), groups: 1.
    # Note: We intentionally use these production numbers even though our act_tt is small.
    conv_kwargs = {
        "in_channels": 16,
        "out_channels": 64,
        "batch_size": 16,
        "input_height": 115,
        "input_width": 115,
        "kernel_size": (4, 4),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
    }

    # Define conv_config separately (using production settings).
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        activation="relu",
        input_channels_alignment=16,
        deallocate_activation=True,
        reallocate_halo_output=False,
        act_block_h_override=1568,
        act_block_w_div=1,
        reshard_if_not_optimal=False,
        override_sharding_config=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        core_grid=None,
        transpose_shards=False,
        output_layout=ttnn.TILE_LAYOUT,
        enable_act_double_buffer=True,
        enable_weights_double_buffer=False,
        enable_split_reader=True,
        enable_subblock_padding=False,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    mem_config = ttnn.L1_MEMORY_CONFIG

    # Create dummy weight and bias tensors as in production.
    # Weight shape (OIHW): (64, 16, 4, 4); Bias shape: (1,1,1,64)
    weight = torch.rand((64, 16, 4, 4), dtype=torch.bfloat16)
    bias = torch.rand((1, 1, 1, 64), dtype=torch.bfloat16)
    weight_tt = ttnn.from_torch(weight, ttnn.bfloat16)
    bias_tt = ttnn.from_torch(bias, ttnn.bfloat16)

    # Prepare the weights using act_tt's memory config.
    if not ttnn.is_tensor_storage_on_device(weight_tt):
        weight_tt = ttnn.prepare_conv_weights(
            weight_tensor=weight_tt,
            weights_format="OIHW",
            input_memory_config=act_tt.memory_config(),
            input_layout=act_tt.get_layout(),
            has_bias=True,
            **conv_kwargs,
        )
        bias_tt = ttnn.prepare_conv_bias(
            bias_tensor=bias_tt,
            input_memory_config=act_tt.memory_config(),
            input_layout=act_tt.get_layout(),
            **conv_kwargs,
        )
    weight_tt = ttnn.to_device(weight_tt, device)
    bias_tt = ttnn.to_device(bias_tt, device)

    # Now, call conv2d. The conv2d op will check that the activation matrix’s width (16)
    # matches the weight matrix’s height (expected to be 115, per conv_kwargs).
    # This mismatch should trigger the error "act_matrix_width == weight_matrix_height".
    output_tt, dims = ttnn.conv2d(
        input_tensor=act_tt,
        weight_tensor=weight_tt,
        bias_tensor=bias_tt,
        **conv_kwargs,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        memory_config=mem_config,
        return_output_dim=True,
    )

    output = ttnn.to_torch(output_tt)
    print("Output shape:", output.shape)

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)

    # Downstream expects output channel dimension to be 64.
    assert output.shape[-1] == 64, "Output channel dimension mismatch"


# Trimmed - also reproduces failure after reset.
# This test aims to extract just the conv2d op from resnet that was failing.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("trim_params", [1])
@pytest.mark.parametrize("increase_stride", [1])
def test_resnet50_first_conv2d_only_repro(batch_size, trim_params, increase_stride, device, reset_device, tmp_path):
    ttnn.light_metal_begin_capture()

    # Create a dummy activation tensor that represents the already-folded activation.
    # In production, the folded activation has shape [1, 1, 211600, 16].
    # For simplicity, we simulate a small unfolded activation with shape (1, 1, 100, 16).
    dummy_activation = torch.rand((batch_size, 1, 100, 16), dtype=torch.bfloat16)
    act_tt = ttnn.from_torch(dummy_activation, ttnn.bfloat16)
    act_tt = ttnn.to_device(act_tt, device)

    # Use trimmed conv_kwargs that match the dummy activation.
    conv_kwargs = {
        "in_channels": 16,
        "out_channels": 32,
        "batch_size": 1,  # trimmed batch_size
        "input_height": 100,  # matching dummy activation height
        "input_width": 16,  # matching dummy activation width
        "kernel_size": (4, 4),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
    }

    # KCM - Quick change to reduce output size greatly.
    if increase_stride:
        conv_kwargs["stride"] = (2, 4)

    # KCM - Made no difference here, both fail.
    if trim_params == 1:
        logger.info("KCM Using trimmed conv2d parameters")

        # Trimmed conv_config with only the essential parameters.
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            activation="relu",
            input_channels_alignment=16,
            act_block_h_override=100,
            act_block_w_div=2,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )

    else:
        logger.info("KCM Using full conv2d parameters")

        # Define conv_config separately (using production settings for the conv op)
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            activation="relu",
            input_channels_alignment=16,
            deallocate_activation=True,
            reallocate_halo_output=False,
            act_block_h_override=100,  # adjusted to match the activation height
            act_block_w_div=2,
            reshard_if_not_optimal=False,
            override_sharding_config=False,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            core_grid=None,
            transpose_shards=False,
            output_layout=ttnn.TILE_LAYOUT,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=False,
            enable_split_reader=True,
            enable_subblock_padding=False,
        )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    mem_config = ttnn.L1_MEMORY_CONFIG

    # Create dummy weight and bias tensors as in production.
    # Weight shape (OIHW): (64, 16, 4, 4); Bias shape: (1,1,1,64)
    weight = torch.rand((32, 16, 4, 4), dtype=torch.bfloat16)
    bias = torch.rand((1, 1, 1, 32), dtype=torch.bfloat16)
    weight_tt = ttnn.from_torch(weight, ttnn.bfloat16)
    bias_tt = ttnn.from_torch(bias, ttnn.bfloat16)

    # Prepare the weights using act_tt's memory config.
    if not ttnn.is_tensor_storage_on_device(weight_tt):
        weight_tt = ttnn.prepare_conv_weights(
            weight_tensor=weight_tt,
            weights_format="OIHW",
            input_memory_config=act_tt.memory_config(),
            input_layout=act_tt.get_layout(),
            has_bias=True,
            **conv_kwargs,
        )
        bias_tt = ttnn.prepare_conv_bias(
            bias_tensor=bias_tt,
            input_memory_config=act_tt.memory_config(),
            input_layout=act_tt.get_layout(),
            **conv_kwargs,
        )
    weight_tt = ttnn.to_device(weight_tt, device)
    bias_tt = ttnn.to_device(bias_tt, device)

    # KCM - Reads to verify matching between capture / replay
    act_host = ttnn.to_torch(act_tt)
    weight_host = ttnn.to_torch(weight_tt)
    bias_host = ttnn.to_torch(bias_tt)

    # Now, call conv2d. With the trimmed conv_kwargs the op should run on a much smaller output tensor.
    output_tt, dims = ttnn.conv2d(
        input_tensor=act_tt,
        weight_tensor=weight_tt,
        bias_tensor=bias_tt,
        **conv_kwargs,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        memory_config=mem_config,
        return_output_dim=True,
    )

    output = ttnn.to_torch(output_tt)
    print("Output shape:", output.shape)

    binary_data = ttnn.light_metal_end_capture()
    write_binary_to_file(binary_data, tmp_path, inspect.currentframe().f_code.co_name)
    reset_device_and_replay_binary(reset_device, device, binary_data)

    # Downstream expects the channel dimension to be 64.
    assert output.shape[-1] == 64, "Output channel dimension mismatch"
