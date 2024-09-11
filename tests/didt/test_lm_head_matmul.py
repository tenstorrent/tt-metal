# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

from tests.didt.matmul_test_base import MatmulTestBase
import ttnn


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
def test_lm_head_matmul(mesh_device, iterations, determinism_check_iterations, use_program_cache):
    # Initialize input configurations
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Initialize matmul configurations
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=1,
        per_core_N=32,
        out_subblock_h=1,
        out_subblock_w=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    lm_head_test = LMHeadTest(
        mesh_device,
        seq_len=32,
        inner_dim=4544,
        weights_n=65024,
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
        mesh_device.get_device(logical_chip_id), iterations, determinism_check_iterations, use_program_cache
    )


@pytest.mark.parametrize(
    "board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["board_mesh_device"],
)
def test_specific_board_lm_head_matmul(board_mesh_device, iterations, determinism_check_iterations, use_program_cache):
    test_lm_head_matmul(board_mesh_device, iterations, determinism_check_iterations, use_program_cache)
