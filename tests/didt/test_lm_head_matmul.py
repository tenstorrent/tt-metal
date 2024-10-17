# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

from tests.didt.matmul_test_base import MatmulTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole


class LMHeadTest(MatmulTestBase):
    def __init__(
        self,
        mesh_device,
        seq_len,
        inner_dim,
        weights_n,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        in0_dtype,
        in1_dtype,
        out_dtype,
        program_config,
        compute_config,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
    ):
        super().__init__(
            mesh_device,
            seq_len,
            inner_dim,
            weights_n,
            in0_mem_config,
            in1_mem_config,
            out_mem_config,
            in0_dtype,
            in1_dtype,
            out_dtype,
            program_config,
            compute_config,
            loop_count,
            determinism_check_enabled,
            determinism_check_iterations,
        )

    def generate_weights(self, shape):
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
    mesh_device, iterations, determinism_check_iterations, use_program_cache, simulate_bh_harvesting
):
    if simulate_bh_harvesting and is_blackhole() == False:
        pytest.skip("Blackhole harvesting simulation is only supported for Blackhole devices")
    # Initialize input configurations
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Initialize matmul configurations
    compute_grid = get_blackhole_grid_size(simulate_bh_harvesting) if is_blackhole() else ttnn.CoreCoord(8, 8)

    seq_len = 32
    per_core_M = seq_len // 32
    per_core_N = 32
    weights_n = (per_core_N * (compute_grid.x * compute_grid.y) * 32) - 512

    out_subblock_h = 1
    out_subblock_w = 8
    assert per_core_M % out_subblock_h == 0
    assert per_core_N % out_subblock_w == 0

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
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    lm_head_test = LMHeadTest(
        mesh_device,
        seq_len=seq_len,
        inner_dim=4544,
        weights_n=weights_n,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT8_B,
        in1_dtype=ttnn.DataType.BFLOAT8_B,
        out_dtype=ttnn.DataType.BFLOAT8_B,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
    )

    # Run test
    lm_head_test.run_matmul()


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
    mesh_device, logical_chip_id, iterations, determinism_check_iterations, use_program_cache
):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_lm_head_matmul(
        mesh_device.get_device(logical_chip_id), iterations, determinism_check_iterations, use_program_cache, False
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize(
    "board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["board_mesh_device"],
)
def test_specific_board_lm_head_matmul(board_mesh_device, iterations, determinism_check_iterations, use_program_cache):
    test_lm_head_matmul(board_mesh_device, iterations, determinism_check_iterations, use_program_cache, False)
