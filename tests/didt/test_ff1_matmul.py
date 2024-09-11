# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch

from tests.didt.matmul_test_base import MatmulTestBase
import ttnn

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
    ],
    indirect=["mesh_device"],
)
def test_ff1_matmul(mesh_device, gelu, math_fidelity, iterations, determinism_check_iterations, use_program_cache):
    # Initialize input configurations
    in0_block_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 7),
                ),
            }
        ),
        [
            128,
            576,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_block_shard_spec)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    # Initialize matmul configurations
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=4,
        per_core_N=72,
        transpose_mcast=False,
        fused_activation=[ttnn.UnaryOpType.GELU, True] if gelu else None,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    ff1_test = MatmulTestBase(
        mesh_device,
        seq_len=1024,
        inner_dim=4608,
        weights_n=18432,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT16,
        in1_dtype=ttnn.DataType.BFLOAT8_B,
        out_dtype=ttnn.DataType.BFLOAT16,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
    )

    # Run test
    ff1_test.run_matmul()


@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
)
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
def test_specific_chip_ff1_matmul(
    mesh_device, logical_chip_id, gelu, math_fidelity, iterations, determinism_check_iterations, use_program_cache
):
    assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_ff1_matmul(
        mesh_device.get_device(logical_chip_id),
        gelu,
        math_fidelity,
        iterations,
        determinism_check_iterations,
        use_program_cache,
    )


@pytest.mark.parametrize(
    "gelu, math_fidelity",
    GELU_FIDELITY_PARAMETRIZATION,
    ids=GELU_FIDELITY_PARAMETRIZATION_IDS,
)
@pytest.mark.parametrize(
    "board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["board_mesh_device"],
)
def test_specific_board_ff1_matmul(
    board_mesh_device, gelu, math_fidelity, iterations, determinism_check_iterations, use_program_cache
):
    test_ff1_matmul(board_mesh_device, gelu, math_fidelity, iterations, determinism_check_iterations, use_program_cache)
