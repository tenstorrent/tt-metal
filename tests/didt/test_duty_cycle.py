# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import random
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.common.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)


class DutyCycleTest(OpTestBase):
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
        non_mm_loops=5,
        wl_loops=100,
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
        self.wl_loops = (wl_loops,)
        self.trace_id = None

    def run_device_operation(self):
        if self.trace_id is None:
            # Ensure all binaries are compiled/loaded before starting trace capture.
            # Trace capture does not allow device writes (e.g. binary loading) on fast dispatch paths.
            ttnn.matmul(
                self.activations,
                self.weights,
                program_config=self.program_config,
                memory_config=self.out_mem_config,
                dtype=self.out_dtype,
                compute_kernel_config=self.compute_config,
            )
            ttnn.cos(
                self.weights,
            )

            self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            for i in range(self.wl_loops[0]):
                ttnn.matmul(
                    self.activations,
                    self.weights,
                    program_config=self.program_config,
                    memory_config=self.out_mem_config,
                    dtype=self.out_dtype,
                    compute_kernel_config=self.compute_config,
                )
                for j in range(self.non_mm_loops[0]):
                    ttnn.cos(
                        self.weights,
                    )
            ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        else:
            ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)

        return None


class DutyCycleRandomizedTest(OpTestBase):
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
        non_mm_loops=5,
        wl_loops=100,
        random_seed=42,
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
        self.wl_loops = (wl_loops,)
        self.random_seed = random_seed
        self.trace_id = None

    def run_device_operation(self):
        if self.trace_id is None:
            # Ensure all binaries are compiled/loaded before starting trace capture.
            # Trace capture does not allow device writes (e.g. binary loading) on fast dispatch paths.
            ttnn.matmul(
                self.activations,
                self.weights,
                program_config=self.program_config,
                memory_config=self.out_mem_config,
                dtype=self.out_dtype,
                compute_kernel_config=self.compute_config,
            )
            ttnn.cos(
                self.weights,
            )

            # Build list of operations maintaining the same ratio
            # wl_loops matmul operations and wl_loops * non_mm_loops cos operations
            operations = ["matmul"] * self.wl_loops[0] + ["cos"] * (self.wl_loops[0] * self.non_mm_loops[0])

            # Shuffle operations using seeded random number generator for reproducibility
            rng = random.Random(self.random_seed)
            rng.shuffle(operations)

            self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            for op in operations:
                if op == "matmul":
                    ttnn.matmul(
                        self.activations,
                        self.weights,
                        program_config=self.program_config,
                        memory_config=self.out_mem_config,
                        dtype=self.out_dtype,
                        compute_kernel_config=self.compute_config,
                    )
                elif op == "cos":
                    ttnn.cos(
                        self.weights,
                    )
            ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        else:
            ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)

        return None


@pytest.mark.parametrize("device_params", [{"trace_region_size": 491520000}], indirect=True)
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
def test_duty_cycle(
    mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
    non_mm_loops,
    wl_loops,
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
        non_mm_loops=non_mm_loops,
        wl_loops=wl_loops,
    )

    # Run test
    duty_cycle_test.run_op_test()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 491520000}], indirect=True)
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
    [10, 100, 1000, 10000],
    ids=["rep-10x", "rep-100x", "rep-1000x", "rep-10000x"],
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
def test_duty_cycle_randomized(
    mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
    non_mm_loops,
    wl_loops,
    grid_size=(8, 8),
):
    # Read random seed from environment variable, default to 31
    random_seed = int(os.getenv("TT_DUTY_CYCLE_SEED", "31"))

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

    duty_cycle_test = DutyCycleRandomizedTest(
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
        non_mm_loops=non_mm_loops,
        wl_loops=wl_loops,
        random_seed=random_seed,
    )

    # Run test
    duty_cycle_test.run_op_test()
