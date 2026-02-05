# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch
import random

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.common.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)


class FF1Test(OpTestBase):
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
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_interval=False,
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
            determinism_check_interval,
        )


GELU_FIDELITY_PARAMETRIZATION = ((False, ttnn.MathFidelity.LoFi), (True, ttnn.MathFidelity.HiFi2))
GELU_FIDELITY_PARAMETRIZATION_IDS = ["without_gelu", "with_gelu"]


@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
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
def test_ff1_matmul(
    mesh_device,
    gelu,
    math_fidelity,
    didt_workload_iterations,
    determinism_check_interval,
    grid_size=(8, 8),
):
    per_core_M = 4
    per_core_N = 72

    # Initialize input configurations
    if is_blackhole():
        compute_grid = get_blackhole_grid_size(mesh_device)
    else:
        compute_grid = ttnn.CoreCoord(grid_size[0], grid_size[1])
    logger.info(f"Running on {compute_grid} cores")

    start_core = ttnn.CoreCoord(0, 0)
    end_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    core_range = ttnn.CoreRange(start_core, end_core)

    in0_block_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                core_range,
            }
        ),
        [
            128,
            576,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_block_shard_spec)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

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
        fused_activation=[ttnn.UnaryOpType.GELU, True] if gelu else None,
    )

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    in0_shape = [1, 1, 32 * per_core_M * compute_grid.y, 576 * compute_grid.x]
    in1_shape = [1, 1, 576 * compute_grid.x, 32 * per_core_N * compute_grid.x]

    ff1_test = FF1Test(
        mesh_device,
        in0_shape=in0_shape,
        in1_shape=in1_shape,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT16,
        in1_dtype=ttnn.DataType.BFLOAT8_B,
        out_dtype=ttnn.DataType.BFLOAT16,
        in0_layout=ttnn.TILE_LAYOUT,
        in1_layout=ttnn.TILE_LAYOUT,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=True if determinism_check_interval > 0 else False,
        determinism_check_interval=determinism_check_interval,
    )

    # Run test
    ff1_test.run_op_test()


def get_mesh_device_logical_chip_combinations():
    """Generate mesh_device and logical_chip_id combinations dynamically"""
    mesh_configs = [
        (1, "1chips"),
        (2, "2chips"),
        (8, "8chips"),
        ((8, 4), "galaxy"),
    ]

    combinations = []
    for mesh_config, mesh_id in mesh_configs:
        # Calculate total chips
        if isinstance(mesh_config, tuple):
            total_chips = mesh_config[0] * mesh_config[1]
        else:
            total_chips = mesh_config

        # Generate combinations for this mesh configuration
        for chip_id in range(total_chips):
            combinations.append(pytest.param(mesh_config, chip_id, id=f"{mesh_id}_chip_{chip_id}"))

    return combinations


@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
)
@pytest.mark.parametrize(
    "mesh_device, logical_chip_id",
    get_mesh_device_logical_chip_combinations(),
    indirect=["mesh_device"],
)
def test_specific_chip_ff1_matmul(
    mesh_device,
    logical_chip_id,
    gelu,
    math_fidelity,
    didt_workload_iterations,
    determinism_check_interval,
):
    submesh_devices = mesh_device.create_submeshes(ttnn.MeshShape((1, 1)))
    test_ff1_matmul(
        submesh_devices[logical_chip_id],
        gelu,
        math_fidelity,
        didt_workload_iterations,
        determinism_check_interval,
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
)
@pytest.mark.parametrize(
    "t3k_single_board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["t3k_single_board_mesh_device"],
)
def test_specific_board_ff1_matmul(
    t3k_single_board_mesh_device,
    gelu,
    math_fidelity,
    didt_workload_iterations,
    determinism_check_interval,
):
    test_ff1_matmul(
        t3k_single_board_mesh_device, gelu, math_fidelity, didt_workload_iterations, determinism_check_interval
    )


@skip_for_blackhole("Use test_blackhole_grid_size_ff1_matmul for blackhole!")
@pytest.mark.parametrize(
    "grid_size",
    [(i, 8) for i in range(1, 9)] + [(8, i) for i in range(1, 8)],
    ids=[f"{i}x8" for i in range(1, 9)] + [f"8x{i}" for i in range(1, 8)],  # 1x8, 2x8 ... 8x1, 8x2...
)
@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
)
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
def test_grid_size_ff1_matmul(
    mesh_device, gelu, math_fidelity, grid_size, didt_workload_iterations, determinism_check_interval
):
    test_ff1_matmul(
        mesh_device,
        gelu,
        math_fidelity,
        didt_workload_iterations,
        determinism_check_interval,
        grid_size=grid_size,
    )


@skip_for_wormhole_b0("Use test_grid_size_ff1_matmul for blackhole!")
@pytest.mark.parametrize(
    "grid_size",
    [(i, 10) for i in range(1, 14)] + [(13, i) for i in range(1, 10)],
    ids=[f"{i}x10" for i in range(1, 14)]
    + [f"13x{i}" for i in range(1, 10)],  # 1x10, 2x10 ..., 13x10, 13x1, 13x2, 13x9
)
@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
)
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
def test_blackhole_grid_size_ff1_matmul(
    mesh_device, gelu, math_fidelity, grid_size, didt_workload_iterations, determinism_check_interval
):
    test_ff1_matmul(
        mesh_device,
        gelu,
        math_fidelity,
        didt_workload_iterations,
        determinism_check_interval,
        grid_size=grid_size,
    )


@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
)
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
def test_mesh_size_ff1_matmul(
    mesh_device,
    sub_mesh_shape,
    mesh_coordinate,
    gelu,
    math_fidelity,
    didt_workload_iterations,
    determinism_check_interval,
):
    # check that sub-mesh with sub_mesh_shape and mesh_coordinate can fit within the parent mesh of MESH_X by MESH_Y
    if mesh_coordinate[0] + sub_mesh_shape[0] > MESH_X or mesh_coordinate[1] + sub_mesh_shape[1] > MESH_Y:
        pytest.skip(
            f"Sub-mesh {sub_mesh_shape} at mesh coordinate {mesh_coordinate} does not fit within parent mesh-device: {MESH_X} by {MESH_Y}"
        )
    sub_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(sub_mesh_shape), ttnn.MeshCoordinate(mesh_coordinate))
    logger.info(f"Running on {sub_mesh_shape} sub-mesh at mesh coordinate {mesh_coordinate}")
    test_ff1_matmul(sub_mesh_device, gelu, math_fidelity, didt_workload_iterations, determinism_check_interval)


@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((MESH_X, MESH_Y), id="all"),
    ],
    indirect=["mesh_device"],
)
def test_random_mesh_size_ff1_matmul(
    mesh_device, gelu, math_fidelity, didt_workload_iterations, determinism_check_interval
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
    test_ff1_matmul(sub_mesh_device, gelu, math_fidelity, didt_workload_iterations, determinism_check_interval)
