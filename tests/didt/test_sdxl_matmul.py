# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else NUM_DEVICES / MESH_X


class SdxlMmTest(OpTestBase):
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

    def deallocate_activations(self):
        if self.in0_mem_config != ttnn.DRAM_MEMORY_CONFIG:
            self.activations.deallocate(True)


# Test cases for matmuls that hang in SDXL UNet
mm_test_cases = [
    {
        "id": "geglu_wo_gelu",
        "M": 1024,
        "K": 1280,
        "N": 5120,
        "in0_mem_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "in0_block_w": 5,
        "out_subblock_h": 1,
        "out_subblock_w": 5,
        "with_gelu": False,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    },
    {
        "id": "geglu_w_gelu",
        "M": 1024,
        "K": 1280,
        "N": 5120,
        "in0_mem_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "in0_block_w": 5,
        "out_subblock_h": 1,
        "out_subblock_w": 5,
        "with_gelu": False,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    },
    {
        "id": "attn_in0_l1",
        "M": 1024,
        "K": 1280,
        "N": 1280,
        "in0_mem_config": ttnn.L1_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,
        "in0_block_w": 5,
        "out_subblock_h": 1,
        "out_subblock_w": 5,
        "with_gelu": False,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    },
    {
        "id": "attn_in0_dram",
        "M": 1024,
        "K": 1280,
        "N": 1280,
        "in0_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,
        "in0_block_w": 5,
        "out_subblock_h": 1,
        "out_subblock_w": 5,
        "with_gelu": False,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    },
    {
        "id": "ff",
        "M": 1024,
        "K": 5120,
        "N": 1280,
        "in0_mem_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "in0_block_w": 10,
        "out_subblock_h": 1,
        "out_subblock_w": 5,
        "with_gelu": False,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    },
    {
        "id": "resnet_640x1280",
        "M": 1024,
        "K": 640,
        "N": 1280,
        "in0_mem_config": ttnn.L1_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "in0_block_w": 4,
        "out_subblock_h": 1,
        "out_subblock_w": 5,
        "with_gelu": False,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    },
    {
        "id": "resnet_2560x1280",
        "M": 1024,
        "K": 2560,
        "N": 1280,
        "in0_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        "in0_block_w": 2,
        "out_subblock_h": 1,
        "out_subblock_w": 5,
        "with_gelu": False,
        "math_fidelity": ttnn.MathFidelity.HiFi2,
    },
]


@skip_for_blackhole("Blackhole has not been tested, see #25544")
@pytest.mark.parametrize("test_config", mm_test_cases, ids=[c["id"] for c in mm_test_cases])
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
def test_sdxl_matmul(
    mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
    test_config,
    grid_size=(8, 8),
):
    # Initialize input configurations
    if is_blackhole():
        compute_grid = get_blackhole_grid_size(mesh_device)
    else:
        compute_grid = ttnn.CoreCoord(grid_size[0], grid_size[1])
    logger.info(f"Running on {compute_grid} cores")

    in0_shape = [1, 1, test_config["M"], test_config["K"]]
    in1_shape = [1, 1, test_config["K"], test_config["N"]]

    per_core_M = test_config["M"] // compute_grid.y // 32
    per_core_N = test_config["N"] // compute_grid.x // 32

    if test_config["in0_mem_config"].is_sharded():
        assert (
            test_config["in0_mem_config"].memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        ), "Test assumes block sharding"

        in0_mem_config = ttnn.create_sharded_memory_config(
            shape=in0_shape,
            core_grid=ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in0_mem_config = test_config["in0_mem_config"]

    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = test_config["out_mem_config"]

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(compute_grid.x, compute_grid.y),
        in0_block_w=test_config["in0_block_w"],
        out_subblock_h=test_config["out_subblock_h"],
        out_subblock_w=test_config["out_subblock_w"],
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=[ttnn.UnaryOpType.GELU, True] if test_config["with_gelu"] else None,
        fuse_batch=True,
    )

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=test_config["math_fidelity"],
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    sdxl_matmul_test = SdxlMmTest(
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
    sdxl_matmul_test.run_op_test()


@pytest.mark.parametrize("test_config", mm_test_cases, ids=[c["id"] for c in mm_test_cases])
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
def test_specific_chip_sdxl_matmul(
    mesh_device,
    logical_chip_id,
    didt_workload_iterations,
    determinism_check_interval,
    test_config,
):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_sdxl_matmul(
        mesh_device.get_device(logical_chip_id),
        didt_workload_iterations,
        determinism_check_interval,
        test_config,
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize("test_config", mm_test_cases, ids=[c["id"] for c in mm_test_cases])
@pytest.mark.parametrize(
    "t3k_single_board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["t3k_single_board_mesh_device"],
)
def test_specific_board_sdxl_matmul(
    t3k_single_board_mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
    test_config,
):
    test_sdxl_matmul(
        t3k_single_board_mesh_device,
        didt_workload_iterations,
        determinism_check_interval,
        test_config,
    )


@skip_for_blackhole("Use test_blackhole_grid_size_ff1_matmul for blackhole!")
@pytest.mark.parametrize("test_config", mm_test_cases, ids=[c["id"] for c in mm_test_cases])
@pytest.mark.parametrize(
    "grid_size",
    [(i, 8) for i in range(1, 9)] + [(8, i) for i in range(1, 8)],
    ids=[f"{i}x8" for i in range(1, 9)] + [f"8x{i}" for i in range(1, 8)],  # 1x8, 2x8 ... 8x1, 8x2...
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
def test_grid_size_sdxl_matmul(
    mesh_device, grid_size, didt_workload_iterations, determinism_check_interval, test_config
):
    test_sdxl_matmul(
        mesh_device,
        didt_workload_iterations,
        determinism_check_interval,
        test_config,
        grid_size=grid_size,
    )


@skip_for_blackhole("Blackhole has not been tested, see #25544")
@skip_for_wormhole_b0("Use test_grid_size_ff1_matmul for blackhole!")
@pytest.mark.parametrize("test_config", mm_test_cases, ids=[c["id"] for c in mm_test_cases])
@pytest.mark.parametrize(
    "grid_size",
    [(i, 10) for i in range(1, 14)] + [(13, i) for i in range(1, 10)],
    ids=[f"{i}x10" for i in range(1, 14)]
    + [f"13x{i}" for i in range(1, 10)],  # 1x10, 2x10 ..., 13x10, 13x1, 13x2, 13x9
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
def test_blackhole_grid_size_sdxl_matmul(
    mesh_device, grid_size, didt_workload_iterations, determinism_check_interval, test_config
):
    test_sdxl_matmul(
        mesh_device,
        didt_workload_iterations,
        determinism_check_interval,
        test_config,
        grid_size=grid_size,
    )
