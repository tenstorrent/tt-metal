# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, OpParameter, get_mesh_grid_size
import ttnn
from models.common.utility_functions import skip_for_blackhole, is_blackhole

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)


class MinimalMatmulWorstCaseTest(OpTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_torch_activations(self, shape):
        return torch.randn(shape, dtype=torch.bfloat16).float()

    def generate_torch_input(self, shape):
        return torch.randn(shape, dtype=torch.bfloat16).float()

    def run_device_operation(self):
        return ttnn.experimental.minimal_matmul(
            input_tensor=self.activations,
            weight_tensor=self.inputs[0],
            bias_tensor=None,
            compute_kernel_config=self.compute_config,
            config=self.program_config,
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
    "dtype, math_fidelity, fp32_acc, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False, 8, 16, 8, 2, 4),
        (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False, 8, 16, 4, 2, 4),
        (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, False, 8, 16, 16, 2, 4),
        (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, False, 8, 16, 16, 2, 4),
    ],
    ids=["bf16_HiFi2", "bf8b_HiFi2", "bf8b_LoFi", "bf4b_LoFi"],
)
def test_minimal_matmul(
    mesh_device,
    dtype,
    math_fidelity,
    fp32_acc,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    didt_workload_iterations,
    determinism_check_interval,
):
    M, K, N = 16384, 16384, 16384

    compute_grid = get_mesh_grid_size(mesh_device)
    compute_with_storage_grid_size = (compute_grid.x, compute_grid.y)
    logger.info(f"Running on {compute_with_storage_grid_size} cores")

    in0_shape = [M, K]
    in1_shape = [K, N]

    in0_dtype = dtype
    in1_dtype = dtype
    out_dtype = dtype

    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Create compute config
    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    # Create core grid for minimal_matmul config
    core_grid = ttnn.CoreCoord(compute_with_storage_grid_size[0], compute_with_storage_grid_size[1])

    # Create matmul config
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )

    minimalMatmulTest = MinimalMatmulWorstCaseTest(
        mesh_device,
        OpParameter(in0_shape, in0_dtype, ttnn.TILE_LAYOUT, in0_mem_config),  # activations
        [
            OpParameter(in1_shape, in1_dtype, ttnn.TILE_LAYOUT, in1_mem_config),  # inputs
        ],
        out_mem_config,
        out_dtype,
        matmul_config,  # program_config
        compute_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
    )

    minimalMatmulTest.run_op_test()
