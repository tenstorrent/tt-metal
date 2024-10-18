# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

from tests.didt.matmul_test_base import MatmulTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole


class ResnetConvTest(MatmulTestBase):
    def __init__(
        self,
        mesh_device,  # device
        in0_shape,  # [batch, in_channels, in_height, in_width]
        in1_shape,  # [out_channels, in_channels // groups, filter_height, filter_width]
        in0_mem_config,  # activation memory config - set to dram interleaved RM
        in1_mem_config,  # weight memory config - set to dram interleaved
        out_mem_config,  #
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_layout,
        in1_layout,
        program_config,
        compute_config,
        input_channels,
        out_channels,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
        batch_size,
        input_height,
        input_width,
        groups,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
    ):
        super().__init__(
            mesh_device,
            in0_shape,
            in1_shape,
            in0_mem_config,
            in1_mem_config,
            out_mem_config,
            in0_dtype,
            in1_dtype,
            out_dtype,
            in0_layout,
            in1_layout,
            program_config,
            compute_config,
            loop_count,
            determinism_check_enabled,
            determinism_check_iterations,
        )
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation = dilation
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.groups = groups
        self.reader_patterns_cache = {}

    # Remove weights shape
    def generate_weights(self, shape):
        return torch.randn(self.in1_shape, dtype=torch.bfloat16).float()

    # Remove weights shape
    def generate_activations(self, shape):
        torch_input_tensor_nchw = torch.randn(self.in0_shape, dtype=torch.bfloat16).float()
        torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
        return torch_input_tensor

    def generate_tt_activations_from_torch(self, torch_tensor):
        return ttnn.from_torch(torch_tensor, self.in0_dtype)

    def generate_tt_weights_from_torch(self, torch_tensor):
        return ttnn.from_torch(torch_tensor, self.in1_dtype)

    def convert_activations_to_memory_config(self, activations):
        # For this case, activations need to be on device :(
        return activations

    def deallocate_activations(self):
        # Do nothing in conv case
        pass

    def run_device_operation(self):
        # print("Conv config:")
        # print("Dtype: ", self.program_config.dtype)
        # print("Weights dtype: ", self.program_config.weights_dtype)
        # print("Math fidelity: ", self.program_config.math_fidelity)
        # print("Shard layout: ", self.program_config.shard_layout)
        # print("Input channels alignment: ", self.program_config.input_channels_alignment)
        # print("Deallocate activation: ", self.program_config.deallocate_activation)
        # print("FP32 dest acc enabled: ", self.program_config.fp32_dest_acc_enabled)
        # print("Packer L1 accum enabled: ", self.program_config.packer_l1_accum_enabled)
        # print("Enable act double buffer: ", self.program_config.enable_act_double_buffer)
        # print("Enable split reader: ", self.program_config.enable_split_reader)
        # print("Enable subblock padding: ", self.program_config.enable_subblock_padding)
        # print("Act block H override: ", self.program_config.act_block_h_override)
        # print("Math approx mode: ", self.program_config.math_approx_mode_enabled)
        # print("String activation", self.program_config.activation)
        # print("Reallocate halo output", self.program_config.reallocate_halo_output)
        # print("Act block W div", self.program_config.act_block_w_div)
        # print("Reshard if not optimal: ", self.program_config.reshard_if_not_optimal)
        # print("Override sharding config: ", self.program_config.override_sharding_config)
        # print("Core grid: ", self.program_config.core_grid)
        # print("Transpose shards: ", self.program_config.transpose_shards)
        # print("Output layout: ", self.program_config.output_layout)

        # print("Act: ", self.activations)
        # print("Weights: ", self.weights)
        # print("In channels: ", self.input_channels)
        # print("Out channels: ", self.out_channels)
        # print("Device: -")
        # print("Bias = none")
        # print("Kernel size: ", (self.filter_height, self.filter_width))
        # print("Stride: ", (self.stride_h, self.stride_w))
        # print("Padding: ", (self.pad_h, self.pad_w))
        # print("Dilation: ", (self.dilation, self.dilation))
        # print("Batch size: ", self.batch_size)
        # print("Input height: ", self.input_height)
        # print("Input width: ", self.input_width)
        # print("Conv config: ", self.program_config)
        # print("Conv op cache: ", self.reader_patterns_cache)
        # print("Groups: ", self.groups)

        [tt_output_tensor_on_device, _, _, _, _] = ttnn.conv2d(
            input_tensor=self.activations,
            weight_tensor=self.weights,
            in_channels=self.input_channels,
            out_channels=self.out_channels,
            device=self.mesh_device,
            bias_tensor=None,
            kernel_size=(self.filter_height, self.filter_width),
            stride=(self.stride_h, self.stride_w),
            padding=(self.pad_h, self.pad_w),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            conv_config=self.program_config,
            conv_op_cache=self.reader_patterns_cache,
            debug=False,
            groups=self.groups,
        )
        self.reader_patterns_cache.clear()
        return tt_output_tensor_on_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
def test_resnet_conv(mesh_device, iterations, determinism_check_iterations, use_program_cache, simulate_bh_harvesting):
    if simulate_bh_harvesting and is_blackhole() == False:
        pytest.skip("Blackhole harvesting simulation is only supported for Blackhole devices")

    math_fidelity = ttnn.MathFidelity.LoFi
    batch_size = 16
    output_channels = 64
    input_channels = 16
    input_height = 115
    input_width = 115
    filter_height = 4
    filter_width = 4
    stride_h = 1
    stride_w = 1
    pad_h = 0
    pad_w = 0
    config_override_act_block_h = 256
    dilation = 1
    fp32_accum = False
    packer_l1_acc = True
    deallocate_activation = False
    groups = 1

    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]

    in0_shape = conv_input_shape
    in1_shape = conv_weight_shape

    activations_dtype = ttnn.DataType.BFLOAT8_B
    weights_dtype = ttnn.DataType.BFLOAT8_B

    # Weird situation
    # act and weight dtype need to be bfp8, which is set through conv config
    # however when creating tensors they need to be different
    in0_dtype = ttnn.bfloat16
    in1_dtype = ttnn.float32

    shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        shard_layout=shard_layout,
        input_channels_alignment=(16),
        deallocate_activation=deallocate_activation,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        enable_act_double_buffer=True,
        enable_split_reader=True,
        enable_subblock_padding=False,
    )
    conv_config.act_block_h_override = config_override_act_block_h

    resnetConvTest = ResnetConvTest(
        mesh_device,
        in0_shape,
        in1_shape,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),  # see what this does
        in0_dtype,
        in1_dtype,
        None,  # out_dtype
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
        conv_config,  # program config
        None,  # compute config
        input_channels,
        output_channels,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
        batch_size,
        input_height,
        input_width,
        groups,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
    )

    resnetConvTest.run_matmul()


@skip_for_blackhole("Multi-chip Blackhole has not been tested")
@pytest.mark.parametrize("logical_chip_id", range(32), ids=[f"logical_chip_{i}_" for i in range(32)])
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
def test_specific_chip_resnet_conv(
    mesh_device, logical_chip_id, iterations, determinism_check_iterations, use_program_cache
):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_resnet_conv(
        mesh_device.get_device(logical_chip_id), iterations, determinism_check_iterations, use_program_cache, False
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize(
    "board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["board_mesh_device"],
)
def test_specific_board_resnet_conv(board_mesh_device, iterations, determinism_check_iterations, use_program_cache):
    test_resnet_conv(board_mesh_device, iterations, determinism_check_iterations, use_program_cache, False)
