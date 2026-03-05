# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch
import random

from tests.didt.op_test_base import OpTestBase, OpParameter, get_mesh_grid_size
import ttnn
from models.common.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)


class DutyCycleTest(OpTestBase):
    def __init__(self, *args, non_mm_loops=5, wl_loops=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_mm_loops = (non_mm_loops,)
        self.wl_loops = (wl_loops,)

    def run_device_operation(self):
        for i in range(self.wl_loops[0]):
            ttnn.matmul(
                self.activations,
                self.inputs[0],
                program_config=self.program_config,
                memory_config=self.out_mem_config,
                dtype=self.out_dtype,
                compute_kernel_config=self.compute_config,
            )
            for j in range(self.non_mm_loops[0]):
                ttnn.cos(self.inputs[0])

        return ttnn.matmul(
            self.activations,
            self.inputs[0],
            program_config=self.program_config,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_config,
        )


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
# number of workload repetitions (of alternating mm/non-mm workloads)
@pytest.mark.parametrize(
    "wl_loops",
    [100, 1000, 10000],
    ids=["rep-100x", "rep-1000x", "rep-10000x"],
)
# the number of non-mm loops within each iteration (increasing this lowers the duty cycle)
@pytest.mark.parametrize(
    "non_mm_loops",
    [1, 2, 3, 4, 5, 6],
    ids=["duty-1", "duty-2", "duty-3", "duty-4", "duty-5", "duty-6"],
)
def test_duty_cycle(
    mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
    non_mm_loops,
    wl_loops,
):
    per_core_M = 4
    per_core_N = 72

    # Initialize input configurations
    compute_grid = get_mesh_grid_size(mesh_device)
    logger.info(f"Running on {compute_grid} cores")

    # Initialize matmul configurations
    out_subblock_h = 1
    out_subblock_w = 8

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(compute_grid.x, compute_grid.y),
        in0_block_w=3,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    in0_shape = [1, 1, 32 * per_core_M * compute_grid.y, 576 * compute_grid.x]
    in1_shape = [1, 1, 576 * compute_grid.x, 32 * per_core_N * compute_grid.x]

    in0_mem_config = ttnn.create_sharded_memory_config(
        in0_shape,
        ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x),
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    duty_cycle_test = DutyCycleTest(
        mesh_device,
        OpParameter(in0_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, in0_mem_config),
        [
            OpParameter(in1_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT, in1_mem_config),
        ],
        out_mem_config=out_mem_config,
        out_dtype=ttnn.DataType.BFLOAT16,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
        non_mm_loops=non_mm_loops,
        wl_loops=wl_loops,
    )

    # Run test
    duty_cycle_test.run_op_test()


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
# number of workload repetitions (of alternating mm/non-mm workloads)
@pytest.mark.parametrize(
    "wl_loops",
    [100, 1000, 10000],
    ids=["rep-100x", "rep-1000x", "rep-10000x"],
)
# the number of non-mm loops within each iteration (increasing this lowers the duty cycle)
@pytest.mark.parametrize(
    "non_mm_loops",
    [
        (1),
        (2),
        (3),
        (4),
        (5),
        (6),
    ],
    ids=["duty-1", "duty-2", "duty-3", "duty-4", "duty-5", "duty-6"],
)
def test_mesh_size_duty_cycle(
    mesh_device,
    sub_mesh_shape,
    mesh_coordinate,
    didt_workload_iterations,
    determinism_check_interval,
    non_mm_loops,
    wl_loops,
):
    # check that sub-mesh with sub_mesh_shape and mesh_coordinate can fit within the parent mesh of MESH_X by MESH_Y
    if mesh_coordinate[0] + sub_mesh_shape[0] > MESH_X or mesh_coordinate[1] + sub_mesh_shape[1] > MESH_Y:
        pytest.skip(
            f"Sub-mesh {sub_mesh_shape} at mesh coordinate {mesh_coordinate} does not fit within parent mesh-device: {MESH_X} by {MESH_Y}"
        )
    sub_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(sub_mesh_shape), ttnn.MeshCoordinate(mesh_coordinate))
    logger.info(f"Running on {sub_mesh_shape} sub-mesh at mesh coordinate {mesh_coordinate}")
    test_duty_cycle(sub_mesh_device, didt_workload_iterations, determinism_check_interval, non_mm_loops, wl_loops)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((MESH_X, MESH_Y), id="all"),
    ],
    indirect=["mesh_device"],
)
# number of workload repetitions (of alternating mm/non-mm workloads)
@pytest.mark.parametrize(
    "wl_loops",
    [100, 1000, 10000],
    ids=["rep-100x", "rep-1000x", "rep-10000x"],
)
# the number of non-mm loops within each iteration (increasing this lowers the duty cycle)
@pytest.mark.parametrize(
    "non_mm_loops",
    [
        (1),
        (2),
        (3),
        (4),
        (5),
        (6),
    ],
    ids=["duty-1", "duty-2", "duty-3", "duty-4", "duty-5", "duty-6"],
)
def test_random_mesh_size_duty_cycle(
    mesh_device, didt_workload_iterations, determinism_check_interval, non_mm_loops, wl_loops
):
    # generate random sub-mesh shape and mesh coordinate
    valid_sub_mesh_shapes = [(x, y) for x in range(1, MESH_X + 1) for y in range(1, MESH_Y + 1)]
    sub_mesh_shape = random.choice(valid_sub_mesh_shapes)
    valid_mesh_coordinates = [
        (x, y) for x in range(0, MESH_X + 1 - sub_mesh_shape[0]) for y in range(0, MESH_Y + 1 - sub_mesh_shape[1])
    ]
    mesh_coordinate = random.choice(valid_mesh_coordinates)

    sub_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(sub_mesh_shape), ttnn.MeshCoordinate(mesh_coordinate))
    logger.info(f"Running on {sub_mesh_shape} sub-mesh at mesh coordinate {mesh_coordinate}")
    test_duty_cycle(sub_mesh_device, didt_workload_iterations, determinism_check_interval, non_mm_loops, wl_loops)
