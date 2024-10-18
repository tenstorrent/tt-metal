# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole


class ResnetConvTest(OpTestBase):
    def __init__(
        self,
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
        weights_block_h,
        weights_block_w,
        weights_df_on_device,
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
        self.weights_block_h = weights_block_h
        self.weights_block_w = weights_block_w
        self.weights_df_on_device = weights_df_on_device
        self.reader_patterns_cache = {}

    # Remove weights shape
    def generate_torch_weights(self, shape):
        return torch.randn(self.in1_shape, dtype=torch.bfloat16).float()

    # Remove weights shape
    def generate_torch_activations(self, shape):
        torch_input_tensor_nchw = torch.randn(self.in0_shape, dtype=torch.bfloat16).float()
        torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
        return torch_input_tensor

    def generate_tt_activations_from_torch(self, torch_tensor):
        return ttnn.from_torch(
            torch_tensor,
            dtype=self.in0_dtype,
            layout=self.in0_layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=self.mesh_device,
            mesh_mapper=self.from_torch_mesh_mapper,
        )

    def generate_tt_weights_from_torch(self, torch_tensor):
        tt_weights = ttnn.Tensor(torch_tensor.flatten().tolist(), self.in1_shape, self.in1_dtype, ttnn.ROW_MAJOR_LAYOUT)
        tt_weights_tiled_host = ttnn.operations.conv2d.convert_conv_weight_tensor_to_tiled_layout(
            tt_weights, self.weights_block_h, self.weights_block_w, self.weights_df_on_device
        )
        return tt_weights_tiled_host.to(self.mesh_device, self.in1_mem_config)

    def convert_activations_to_memory_config(self, activations):
        return ttnn.to_memory_config(activations, self.in0_mem_config)

    def deallocate_activations(self):
        # Do nothing in conv case as activations are on device
        pass

    def run_device_operation(self):
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
    if is_blackhole():
        pytest.skip("Blackhole is not supported for this test")
    if simulate_bh_harvesting and is_blackhole() == False:
        pytest.skip("Blackhole harvesting simulation is only supported for Blackhole devices")

    groups = 1
    dilation = 1
    pad_w = 0
    pad_h = 0
    stride_w = 1
    stride_h = 1
    filter_height = 4
    filter_width = 4
    input_height = 115
    input_width = 115
    batch_size = 16
    output_channels = 64
    input_channels = 16
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    weights_block_h = 2
    weights_block_w = 2

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
        math_fidelity=ttnn.MathFidelity.LoFi,
        shard_layout=shard_layout,
        input_channels_alignment=(16),
        deallocate_activation=False,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=True,
        enable_act_double_buffer=True,
        enable_split_reader=True,
        enable_subblock_padding=False,
    )
    # This sets subblocks to [2, 4] in underlying matmul
    conv_config.act_block_h_override = 256

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
        weights_block_h,
        weights_block_w,
        weights_dtype,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
    )

    resnetConvTest.run_op_test()


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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_specific_board_resnet_conv(board_mesh_device, iterations, determinism_check_iterations, use_program_cache):
    test_resnet_conv(board_mesh_device, iterations, determinism_check_iterations, use_program_cache, False)
