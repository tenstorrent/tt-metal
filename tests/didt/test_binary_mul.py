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


# This test was created to measure power consumption of BH chip on non-matmul workload.
# The underlying workload is binary eltwise multiplication.
class BinaryMulTest(OpTestBase):
    def __init__(self, *args, gelu=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gelu = gelu

    def run_device_operation(self):
        return ttnn.mul(
            self.activations,
            self.inputs[0],
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)] if self.gelu else [],
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
def test_binary_mul(
    mesh_device,
    gelu,
    math_fidelity,
    didt_workload_iterations,
    determinism_check_interval,
):
    # Initialize input configurations
    compute_grid = get_mesh_grid_size(mesh_device)
    logger.info(f"Running on {compute_grid} cores")

    in0_shape = [1, 1, 640 * compute_grid.y, 576 * compute_grid.x]
    in1_shape = in0_shape

    in0_mem_config = ttnn.L1_MEMORY_CONFIG
    in1_mem_config = ttnn.L1_MEMORY_CONFIG
    out_mem_config = ttnn.L1_MEMORY_CONFIG
    program_config = None
    compute_config = None

    binary_mul_test = BinaryMulTest(
        mesh_device,
        OpParameter(in0_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT, in0_mem_config),
        [
            OpParameter(in1_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT, in1_mem_config),
        ],
        out_mem_config=out_mem_config,
        out_dtype=ttnn.DataType.BFLOAT8_B,
        program_config=program_config,
        compute_config=compute_config,
        gelu=gelu,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
    )

    # Run test
    binary_mul_test.run_op_test()
