# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, OpParameter, get_mesh_grid_size
import ttnn
from models.common.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)


# This test was created to perform temperature readings on BH chip
# The workload starts with loops of a non-matmul OP to bring the chip to
# steady state, followed by sharded matmul which draws max power
class FF1Test(OpTestBase):
    def __init__(self, *args, non_mm_loops=1000, mm_loops=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_mm_loops = (non_mm_loops,)
        self.mm_loops = (mm_loops,)

    def run_device_operation(self):
        # run matmul once to cache kernel
        ttnn.matmul(
            self.activations,
            self.inputs[0],
            program_config=self.program_config,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_config,
        )

        for _ in range(self.non_mm_loops[0]):
            ttnn.cos(
                self.inputs[0],
            )

        for _ in range(self.mm_loops[0] - 1):
            ttnn.matmul(
                self.activations,
                self.inputs[0],
                program_config=self.program_config,
                memory_config=self.out_mem_config,
                dtype=self.out_dtype,
                compute_kernel_config=self.compute_config,
            )

        return ttnn.matmul(
            self.activations,
            self.inputs[0],
            program_config=self.program_config,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_config,
        )


@pytest.mark.parametrize(
    "gelu, math_fidelity",
    [
        (False, ttnn.MathFidelity.LoFi),
        (True, ttnn.MathFidelity.HiFi2),
    ],
    ids=["without_gelu", "with_gelu"],
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
    loop_counts,
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
        OpParameter(in0_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, in0_mem_config),  # activations
        [
            OpParameter(in1_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT, in1_mem_config),  # inputs
        ],
        out_mem_config=out_mem_config,
        out_dtype=ttnn.DataType.BFLOAT16,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
        non_mm_loops=loop_counts[0],
        mm_loops=loop_counts[1],
    )

    # Run test
    ff1_test.run_op_test()
