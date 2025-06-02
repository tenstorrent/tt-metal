# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0


# This test was created to perform temperature readings on BH chip
# The workload starts with loops of a non-matmul OP to bring the chip to
# steady state, followed by sharded matmul which draws max power
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
        loop_count=1,
        determinism_check_enabled=False,
        determinism_check_interval=False,
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
            determinism_check_interval,
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
    didt_workload_iterations,
    determinism_check_interval,
    use_program_cache,
    loop_counts,
    grid_size=(8, 8),
):
    if is_blackhole() and mesh_device.get_num_devices() > 1:
        pytest.skip("Multi-chip Blackhole has not been tested")

    per_core_M = 4
    per_core_N = 72

    # Initialize input configurations
    if is_blackhole():
        compute_grid = get_blackhole_grid_size(mesh_device)
    else:
        compute_grid = ttnn.CoreCoord(grid_size[0], grid_size[1])
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
        loop_count=didt_workload_iterations,
        determinism_check_enabled=True if determinism_check_interval > 0 else False,
        determinism_check_interval=determinism_check_interval,
        non_mm_loops=loop_counts[0],
        mm_loops=loop_counts[1],
    )

    # Run test
    ff1_test.run_op_test()
