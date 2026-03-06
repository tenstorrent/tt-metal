# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from loguru import logger
import pytest
import torch
import random

from tests.didt.op_test_base import OpTestBase, OpParameter, get_mesh_grid_size
import ttnn
from models.common.utility_functions import skip_for_blackhole, is_blackhole

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)


class ResnetConvTest(OpTestBase):
    def __init__(
        self,
        *args,
        compute_with_storage_grid_size=(8, 8),
        input_channels=None,
        out_channels=None,
        filter_height=None,
        filter_width=None,
        stride_h=None,
        stride_w=None,
        pad_h=None,
        pad_w=None,
        dilation=None,
        batch_size=None,
        input_height=None,
        input_width=None,
        groups=None,
        weights_block_h=None,
        weights_block_w=None,
        weights_df_on_device=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_with_storage_grid_size = compute_with_storage_grid_size
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
    def generate_torch_activations(self, shape):
        torch_input_tensor_nchw = torch.randn(shape, dtype=torch.bfloat16).float()
        torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
        return torch_input_tensor

    # Remove weights shape
    def generate_torch_input(self, shape):
        return torch.randn(shape, dtype=torch.bfloat16)

    def deallocate_activations(self):
        # Do nothing in conv case as activations are on device
        pass

    def run_device_operation(self):
        tt_output_tensor_on_device = ttnn.conv2d(
            input_tensor=self.activations,
            weight_tensor=self.inputs[0],
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
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            compute_config=self.compute_config,
            groups=self.groups,
            dtype=self.out_dtype,
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
        pytest.param((MESH_X, MESH_Y), id="all"),  # run on all available devices
    ],
    indirect=["mesh_device"],
)
def test_resnet_conv(mesh_device, didt_workload_iterations, determinism_check_interval):
    groups = 1
    dilation = 1
    pad_w = 0
    pad_h = 0
    stride_w = 1
    stride_h = 1
    filter_height = 4
    filter_width = 4
    input_height = 35
    input_width = 83

    compute_grid = get_mesh_grid_size(mesh_device)
    compute_with_storage_grid_size = (compute_grid.x, compute_grid.y)
    logger.info(f"Running on {compute_with_storage_grid_size} cores")
    # scale batch_size with num cores to keep sub_block dims
    batch_size = compute_with_storage_grid_size[0] * compute_with_storage_grid_size[1]

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
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        deallocate_activation=False,
        enable_act_double_buffer=True,
    )
    # This sets subblocks to [2, 4] in underlying matmul
    conv_config.act_block_h_override = 40 * 32

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_kernel_config = ComputeConfigClass(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    resnetConvTest = ResnetConvTest(
        mesh_device,
        OpParameter(in0_shape, in0_dtype, ttnn.ROW_MAJOR_LAYOUT, mem_config),  # activations
        [
            OpParameter(in1_shape, in1_dtype, ttnn.ROW_MAJOR_LAYOUT, mem_config),  # inputs
        ],
        out_mem_config=mem_config,
        out_dtype=activations_dtype,
        program_config=conv_config,
        compute_config=compute_kernel_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
        compute_with_storage_grid_size=compute_with_storage_grid_size,
        input_channels=input_channels,
        out_channels=output_channels,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation=dilation,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        groups=groups,
        weights_block_h=weights_block_h,
        weights_block_w=weights_block_w,
        weights_df_on_device=weights_dtype,
    )

    resnetConvTest.run_op_test()


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
def test_specific_chip_resnet_conv(mesh_device, logical_chip_id, didt_workload_iterations, determinism_check_interval):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_resnet_conv(mesh_device.get_device(logical_chip_id), didt_workload_iterations, determinism_check_interval)


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize(
    "t3k_single_board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["t3k_single_board_mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_specific_board_resnet_conv(t3k_single_board_mesh_device, didt_workload_iterations, determinism_check_interval):
    test_resnet_conv(t3k_single_board_mesh_device, didt_workload_iterations, determinism_check_interval)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((MESH_X, MESH_Y), id="all"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "sub_mesh_shape",
    [
        pytest.param((4, 1), id="4x1"),
        pytest.param((4, 2), id="4x2"),
        pytest.param((8, 1), id="8x1"),
        pytest.param((8, 2), id="8x2"),
        pytest.param((8, 3), id="8x3"),
        pytest.param((6, 4), id="6x4"),
        pytest.param((8, 4), id="8x4"),
    ],
)
@pytest.mark.parametrize(
    "mesh_coordinate",
    [
        pytest.param((0, 0), id="0-0"),
        pytest.param((4, 0), id="4-0"),
        pytest.param((0, 1), id="0-1"),
        pytest.param((4, 1), id="4-1"),
        pytest.param((0, 2), id="0-2"),
        pytest.param((4, 2), id="4-2"),
        pytest.param((0, 3), id="0-3"),
        pytest.param((4, 3), id="4-3"),
    ],
)
def test_mesh_size_resnet_conv(
    mesh_device, sub_mesh_shape, mesh_coordinate, didt_workload_iterations, determinism_check_interval
):
    # check that sub-mesh with sub_mesh_shape and mesh_coordinate can fit within the parent mesh of MESH_X by MESH_Y
    if mesh_coordinate[0] + sub_mesh_shape[0] > MESH_X or mesh_coordinate[1] + sub_mesh_shape[1] > MESH_Y:
        pytest.skip(
            f"Sub-mesh {sub_mesh_shape} at mesh coordinate {mesh_coordinate} does not fit within parent mesh-device: {MESH_X} by {MESH_Y}"
        )
    sub_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(sub_mesh_shape), ttnn.MeshCoordinate(mesh_coordinate))
    logger.info(f"Running on {sub_mesh_shape} sub-mesh at mesh coordinate {mesh_coordinate}")
    test_resnet_conv(sub_mesh_device, didt_workload_iterations, determinism_check_interval)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((MESH_X, MESH_Y), id="all"),
    ],
    indirect=["mesh_device"],
)
def test_random_mesh_size_resnet_conv(mesh_device, didt_workload_iterations, determinism_check_interval):
    # generate random sub-mesh shape and mesh coordinate
    valid_sub_mesh_shapes = [(x, y) for x in range(1, MESH_X + 1) for y in range(1, MESH_Y + 1)]
    sub_mesh_shape = random.choice(valid_sub_mesh_shapes)
    valid_mesh_coordinates = [
        (x, y) for x in range(0, MESH_X + 1 - sub_mesh_shape[0]) for y in range(0, MESH_Y + 1 - sub_mesh_shape[1])
    ]
    mesh_coordinate = random.choice(valid_mesh_coordinates)

    sub_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(sub_mesh_shape), ttnn.MeshCoordinate(mesh_coordinate))
    logger.info(f"Running on {sub_mesh_shape} sub-mesh at mesh coordinate {mesh_coordinate}")
    test_resnet_conv(sub_mesh_device, didt_workload_iterations, determinism_check_interval)
