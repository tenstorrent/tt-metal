# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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


class SdpaOpTest(OpTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_torch_activations(self, shape):
        return torch.randn(shape, dtype=torch.bfloat16)

    def generate_torch_input(self, shape):
        return torch.randn(shape, dtype=torch.bfloat16)

    def run_device_operation(self):
        return ttnn.transformer.scaled_dot_product_attention(
            self.activations,
            self.inputs[0],
            self.inputs[1],
            is_causal=False,
            program_config=self.program_config,
            compute_kernel_config=self.compute_config,
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
    "dtype, math_fidelity",
    [
        (ttnn.bfloat16, ttnn.MathFidelity.LoFi),
        (ttnn.bfloat16, ttnn.MathFidelity.HiFi2),
        (ttnn.bfloat16, ttnn.MathFidelity.HiFi3),
        (ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
    ],
    ids=["bf16_LoFi", "bf16_HiFi2", "bf16_HiFi3", "bf16_HiFi4"],
)
def test_sdpa_op(
    mesh_device,
    dtype,
    math_fidelity,
    didt_workload_iterations,
    determinism_check_interval,
):
    compute_grid = get_mesh_grid_size(mesh_device)
    compute_with_storage_grid_size = (compute_grid.x, compute_grid.y)
    logger.info(f"Running on {compute_with_storage_grid_size} cores")

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Create core grid for SDPA config
    core_grid = ttnn.CoreCoord(compute_with_storage_grid_size[0], compute_with_storage_grid_size[1])

    full_grid = mesh_device.compute_with_storage_grid_size()
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=core_grid,
        q_chunk_size=256,
        k_chunk_size=256,
        exp_approx_mode=False,  # NOTE: False is more correct
    )

    sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
    )

    shape = (1, 10, 9472, 128)

    sdpa_test = SdpaOpTest(
        mesh_device,
        OpParameter(shape, dtype, ttnn.TILE_LAYOUT, mem_config),  # activations
        [
            OpParameter(shape, dtype, ttnn.TILE_LAYOUT, mem_config),  # inputs
            OpParameter(shape, dtype, ttnn.TILE_LAYOUT, mem_config),
        ],
        mem_config,  # out
        dtype,  # out
        sdpa_program_config,
        sdpa_compute_kernel_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
    )

    sdpa_test.run_op_test()
