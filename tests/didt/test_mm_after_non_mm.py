# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0


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
        determinism_check_iterations=False,
        non_mm_loops=1000,
        mm_loops=10,
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
        self.non_mm_loops = (non_mm_loops,)
        self.mm_loops = (mm_loops,)

    def run_device_operation(self):
        # run matmul once to cache kernel
        ttnn.matmul(
            self.activations,
            self.weights,
            program_config=self.program_config,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_config,
        )

        for _ in range(self.non_mm_loops[0]):
            ttnn.cos(
                self.weights,
                # self.weights,
                # activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)],
            )

        for _ in range(self.mm_loops[0] - 1):
            ttnn.matmul(
                self.activations,
                self.weights,
                program_config=self.program_config,
                memory_config=self.out_mem_config,
                dtype=self.out_dtype,
                compute_kernel_config=self.compute_config,
            )

        return ttnn.matmul(
            self.activations,
            self.weights,
            program_config=self.program_config,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_config,
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
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "loop_counts",
    [(1000, 20)],
    ids=["loop0"],
)
def test_ff1_matmul(
    mesh_device,
    gelu,
    math_fidelity,
    iterations,
    determinism_check_iterations,
    use_program_cache,
    simulate_bh_harvesting,
    loop_counts,
    grid_size=(13, 10),
):
    if is_blackhole() and mesh_device.get_num_devices() > 1:
        pytest.skip("Multi-chip Blackhole has not been tested")
    if simulate_bh_harvesting and is_blackhole() == False:
        pytest.skip("Blackhole harvesting simulation is only supported for Blackhole devices")

    per_core_M = 4
    per_core_N = 72

    # Initialize input configurations
    if is_blackhole():
        compute_grid = get_blackhole_grid_size(simulate_bh_harvesting)
    else:
        compute_grid = ttnn.CoreCoord(grid_size[0], grid_size[1])

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
        False,
    )

    # Initialize matmul configurations
    out_subblock_h = 1
    out_subblock_w = 8

    subblock_1x1 = os.getenv("TT_USE_1X1_SUBBLOCK") == "1"
    if subblock_1x1:
        out_subblock_h = 1
        out_subblock_w = 1

    fidelity_env = int(os.getenv("TT_MATH_FIDELITY", default=1))
    if fidelity_env == 2:
        math_fidelity = ttnn.MathFidelity.HiFi2
    elif fidelity_env == 3:
        math_fidelity = ttnn.MathFidelity.HiFi3
    elif fidelity_env == 4:
        math_fidelity = ttnn.MathFidelity.HiFi4

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

    # in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_block_shard_spec)
    in0_mem_config = ttnn.create_sharded_memory_config(
        in0_shape,
        ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x),
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

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
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
        non_mm_loops=loop_counts[0],
        mm_loops=loop_counts[1],
    )

    # Run test
    ff1_test.run_op_test()
