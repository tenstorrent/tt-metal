# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pytest
import torch
import random

from tests.didt.op_test_base import OpTestBase, get_blackhole_grid_size
import ttnn
from models.common.utility_functions import skip_for_blackhole, is_blackhole, skip_for_wormhole_b0

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)


class MLA_SDPATest(OpTestBase):
    def __init__(
        self,
        mesh_device,
        in0_shape,
        in1_shape,
        in2_shape,
        in0_mem_config,
        in1_mem_config,
        in2_mem_config,
        out_mem_config,
        in0_dtype,
        in1_dtype,
        in2_dtype,
        out_dtype,
        in0_layout,
        in1_layout,
        in2_layout,
        scale,
        program_config,
        compute_config,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_interval=False,
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
        self.in2_shape = in2_shape
        self.in2_mem_config = in2_mem_config
        self.in2_dtype = in2_dtype
        self.in2_layout = in2_layout
        self.scale = scale

    def generate_tt_weights_from_torch(self, torch_tensor):
        tt_weights = ttnn.from_torch(
            torch_tensor,
            dtype=self.in1_dtype,
            layout=self.in1_layout,
            memory_config=self.in1_mem_config,
            device=self.mesh_device,
            mesh_mapper=self.from_torch_mesh_mapper,
        )
        torch_v = torch.randn(self.in2_shape, dtype=torch.bfloat16).float()
        self.tt_v = ttnn.from_torch(
            torch_v,
            dtype=self.in2_dtype,
            layout=self.in2_layout,
            memory_config=self.in2_mem_config,
            device=self.mesh_device,
            mesh_mapper=self.from_torch_mesh_mapper,
        )
        return tt_weights

    def run_device_operation(self):
        out = ttnn.transformer.flash_mla_prefill(
            self.activations,
            self.weights,
            self.tt_v,
            scale=self.scale,
            program_config=self.program_config,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            is_causal=False,  # TMP
            attn_mask=None,
        )
        return out

    def deallocate_activations(self):
        # No need to deallocate activations in this case, as they are on dram
        pass


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
@skip_for_wormhole_b0("This test is for blackhole")
def test_mla_sdpa(
    mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
    grid_size=(8, 8),
):
    # Initialize input configurations
    if is_blackhole():
        compute_grid = get_blackhole_grid_size(mesh_device)
    else:
        compute_grid = ttnn.CoreCoord(grid_size[0], grid_size[1])
    logger.info(f"Running on {compute_grid} cores")

    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    seq_len_q = 4 * 1024
    seq_len_kv = 128 * 1024
    num_heads = 32
    qk_head_dim = 576
    v_head_dim = 128

    q_shape = [1, num_heads, seq_len_q, qk_head_dim]
    k_shape = [1, 1, seq_len_kv, qk_head_dim]
    v_shape = [1, num_heads, seq_len_kv, v_head_dim]

    scale = qk_head_dim**-0.5
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(compute_grid.x, compute_grid.y),
        q_chunk_size=128,
        k_chunk_size=512,
        exp_approx_mode=False,
    )

    mla_sdpa_test = MLA_SDPATest(
        mesh_device,
        in0_shape=q_shape,
        in1_shape=k_shape,
        in2_shape=v_shape,
        in0_mem_config=in0_mem_config,
        in1_mem_config=in1_mem_config,
        in2_mem_config=in2_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=ttnn.DataType.BFLOAT16,
        in1_dtype=ttnn.DataType.BFLOAT8_B,
        in2_dtype=ttnn.DataType.BFLOAT8_B,
        in0_layout=ttnn.TILE_LAYOUT,
        in1_layout=ttnn.TILE_LAYOUT,
        in2_layout=ttnn.TILE_LAYOUT,
        out_dtype=ttnn.DataType.BFLOAT16,
        compute_config=compute_config,
        scale=scale,
        program_config=sdpa_program_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=True if determinism_check_interval > 0 else False,
        determinism_check_interval=determinism_check_interval,
    )

    # Run test
    mla_sdpa_test.run_op_test()


def get_mesh_device_logical_chip_combinations():
    """Generate mesh_device and logical_chip_id combinations dynamically"""
    mesh_configs = [
        (1, "1chips"),
        (2, "2chips"),
        (8, "8chips"),
        ((8, 4), "galaxy"),
    ]

    combinations = []
    for mesh_config, mesh_id in mesh_configs:
        # Calculate total chips
        if isinstance(mesh_config, tuple):
            total_chips = mesh_config[0] * mesh_config[1]
        else:
            total_chips = mesh_config

        # Generate combinations for this mesh configuration
        for chip_id in range(total_chips):
            combinations.append(pytest.param(mesh_config, chip_id, id=f"{mesh_id}_chip_{chip_id}"))

    return combinations


@pytest.mark.parametrize(
    "mesh_device, logical_chip_id",
    get_mesh_device_logical_chip_combinations(),
    indirect=["mesh_device"],
)
@skip_for_wormhole_b0("This test is for blackhole")
def test_specific_chip_mla_sdpa(
    mesh_device,
    logical_chip_id,
    didt_workload_iterations,
    determinism_check_interval,
):
    submesh_devices = mesh_device.create_submeshes(ttnn.MeshShape((1, 1)))
    test_mla_sdpa(
        submesh_devices[logical_chip_id],
        didt_workload_iterations,
        determinism_check_interval,
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((MESH_X, MESH_Y), id="all"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "sub_mesh_shape",
    [
        pytest.param((4, 1), id="4x1"),
        pytest.param((4, 2), id="4x2"),
        pytest.param((8, 1), id="8x1"),
        pytest.param((8, 2), id="8x2"),
        pytest.param((8, 3), id="8x3"),
        pytest.param((6, 4), id="6x4"),
        pytest.param((8, 4), id="8x4"),
    ],
)
@pytest.mark.parametrize(
    "mesh_coordinate",
    [
        pytest.param((0, 0), id="0-0"),
        pytest.param((4, 0), id="4-0"),
        pytest.param((0, 1), id="0-1"),
        pytest.param((4, 1), id="4-1"),
        pytest.param((0, 2), id="0-2"),
        pytest.param((4, 2), id="4-2"),
        pytest.param((0, 3), id="0-3"),
        pytest.param((4, 3), id="4-3"),
    ],
)
@skip_for_wormhole_b0("This test is for blackhole")
def test_mesh_size_mla_sdpa(
    mesh_device,
    sub_mesh_shape,
    mesh_coordinate,
    didt_workload_iterations,
    determinism_check_interval,
):
    # check that sub-mesh with sub_mesh_shape and mesh_coordinate can fit within the parent mesh of MESH_X by MESH_Y
    if mesh_coordinate[0] + sub_mesh_shape[0] > MESH_X or mesh_coordinate[1] + sub_mesh_shape[1] > MESH_Y:
        pytest.skip(
            f"Sub-mesh {sub_mesh_shape} at mesh coordinate {mesh_coordinate} does not fit within parent mesh-device: {MESH_X} by {MESH_Y}"
        )
    sub_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(sub_mesh_shape), ttnn.MeshCoordinate(mesh_coordinate))
    logger.info(f"Running on {sub_mesh_shape} sub-mesh at mesh coordinate {mesh_coordinate}")
    test_mla_sdpa(sub_mesh_device, didt_workload_iterations, determinism_check_interval)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((MESH_X, MESH_Y), id="all"),
    ],
    indirect=["mesh_device"],
)
@skip_for_wormhole_b0("This test is for blackhole")
def test_random_mesh_size_mla_sdpa(mesh_device, didt_workload_iterations, determinism_check_interval):
    # generate random sub-mesh shape and mesh coordinate
    valid_sub_mesh_shapes = [(x, y) for x in range(1, MESH_X + 1) for y in range(1, MESH_Y + 1)]
    sub_mesh_shape = random.choice(valid_sub_mesh_shapes)
    valid_mesh_coordinates = [
        (x, y) for x in range(0, MESH_X + 1 - sub_mesh_shape[0]) for y in range(0, MESH_Y + 1 - sub_mesh_shape[1])
    ]
    mesh_coordinate = random.choice(valid_mesh_coordinates)

    sub_mesh_device = mesh_device.create_submesh(ttnn.MeshShape(sub_mesh_shape), ttnn.MeshCoordinate(mesh_coordinate))
    logger.info(f"Running on {sub_mesh_shape} sub-mesh at mesh coordinate {mesh_coordinate}")
    test_mla_sdpa(sub_mesh_device, didt_workload_iterations, determinism_check_interval)
