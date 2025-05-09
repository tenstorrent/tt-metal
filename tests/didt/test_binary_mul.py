# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.utility_functions import skip_for_blackhole, is_blackhole


class BinaryMulTest(OpTestBase):
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
        compute_config,
        gelu,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
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
            compute_config,
            gelu,
            loop_count,
            determinism_check_enabled,
            determinism_check_iterations,
        )
        self.gelu = gelu

    def run_device_operation(self):
        return ttnn.mul(
            self.activations,
            self.weights,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)] if self.gelu else None,
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
def test_binary_mul(
    mesh_device,
    gelu,
    math_fidelity,
    iterations,
    determinism_check_iterations,
    use_program_cache,
    simulate_bh_harvesting,
    grid_size=(13, 10),
):
    if is_blackhole() and mesh_device.get_num_devices() > 1:
        pytest.skip("Multi-chip Blackhole has not been tested")
    if simulate_bh_harvesting and is_blackhole() == False:
        pytest.skip("Blackhole harvesting simulation is only supported for Blackhole devices")

    # Initialize input configurations
    if is_blackhole():
        compute_grid = get_blackhole_grid_size(simulate_bh_harvesting)
    else:
        compute_grid = ttnn.CoreCoord(grid_size[0], grid_size[1])

    in0_shape = [1, 1, 640 * compute_grid.y, 576 * compute_grid.x]
    in1_shape = in0_shape

    # in0_mem_config = ttnn.create_sharded_memory_config(in0_shape, ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR)
    # in1_mem_config = ttnn.create_sharded_memory_config(in0_shape, ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR)
    # out_mem_config = ttnn.create_sharded_memory_config(in0_shape, ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR)
    in0_mem_config = ttnn.L1_MEMORY_CONFIG
    in1_mem_config = ttnn.L1_MEMORY_CONFIG
    out_mem_config = ttnn.L1_MEMORY_CONFIG
    compute_config = None

    binary_mul_test = BinaryMulTest(
        mesh_device,
        in0_shape=in0_shape,
        in1_shape=in1_shape,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT8_B,
        in1_dtype=ttnn.DataType.BFLOAT8_B,
        out_dtype=ttnn.DataType.BFLOAT8_B,
        in0_layout=ttnn.TILE_LAYOUT,
        in1_layout=ttnn.TILE_LAYOUT,
        compute_config=compute_config,
        gelu=gelu,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
    )

    # Run test
    binary_mul_test.run_op_test()
