# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0


class LMHeadTest(OpTestBase):
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

    def generate_torch_weights(self, shape):
        return torch.randn(shape) - 0.95


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
def test_lm_head_matmul(
    mesh_device, didt_workload_iterations, determinism_check_interval, use_program_cache, grid_size=(8, 8)
):
    if is_blackhole() and mesh_device.get_num_devices() > 1:
        pytest.skip("Multi-chip Blackhole has not been tested")

    # Initialize input configurations
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Initialize matmul configurations
    if is_blackhole():
        compute_grid = get_blackhole_grid_size(mesh_device)
    else:
        compute_grid = ttnn.CoreCoord(grid_size[0], grid_size[1])
    logger.info(f"Running on {compute_grid} cores")

    in1_dtype = ttnn.DataType.BFLOAT8_B
    seq_len = 32
    per_core_M = seq_len // 32
    per_core_N = 32
    weights_n = (per_core_N * (compute_grid.x * compute_grid.y) * 32) - 512

    out_subblock_h = 1
    out_subblock_w = 8
    assert per_core_M % out_subblock_h == 0
    assert per_core_N % out_subblock_w == 0

    math_fidelity = ttnn.MathFidelity.LoFi

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(compute_grid.x, compute_grid.y),
        in0_block_w=2,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    in0_shape = [1, 1, seq_len, 4544]
    in1_shape = [1, 1, 4544, weights_n]

    lm_head_test = LMHeadTest(
        mesh_device,
        in0_shape=in0_shape,
        in1_shape=in1_shape,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT8_B,
        in1_dtype=in1_dtype,
        out_dtype=ttnn.DataType.BFLOAT8_B,
        in0_layout=ttnn.TILE_LAYOUT,
        in1_layout=ttnn.TILE_LAYOUT,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=True if determinism_check_interval > 0 else False,
        determinism_check_interval=determinism_check_interval,
    )

    # Run test
    lm_head_test.run_op_test()


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
def test_specific_chip_lm_head_matmul(
    mesh_device, logical_chip_id, didt_workload_iterations, determinism_check_interval, use_program_cache
):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_lm_head_matmul(
        mesh_device.get_device(logical_chip_id),
        didt_workload_iterations,
        determinism_check_interval,
        use_program_cache,
        False,
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize(
    "t3k_single_board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["t3k_single_board_mesh_device"],
)
def test_specific_board_lm_head_matmul(
    t3k_single_board_mesh_device, didt_workload_iterations, determinism_check_interval, use_program_cache
):
    test_lm_head_matmul(
        t3k_single_board_mesh_device, didt_workload_iterations, determinism_check_interval, use_program_cache, False
    )


@skip_for_blackhole("Use test_blackhole_grid_size_lm_head_matmul test for blackhole!")
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
def test_grid_size_lm_head_matmul(
    mesh_device, grid_size, didt_workload_iterations, determinism_check_interval, use_program_cache
):
    test_lm_head_matmul(
        mesh_device, didt_workload_iterations, determinism_check_interval, use_program_cache, False, grid_size=grid_size
    )


@skip_for_wormhole_b0("Use test_grid_size_lm_head_matmul for blackhole!")
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
def test_blackhole_grid_size_lm_head_matmul(
    mesh_device, grid_size, didt_workload_iterations, determinism_check_interval, use_program_cache
):
    test_lm_head_matmul(
        mesh_device, didt_workload_iterations, determinism_check_interval, use_program_cache, False, grid_size=grid_size
    )
