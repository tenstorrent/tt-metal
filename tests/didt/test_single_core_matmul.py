# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Determinism probe for a single named compute core.

The matmul is confined to one logical worker core via ``allowed_worker_cores`` and
replicated byte-for-byte to every chip in the mesh, so the OpTestBase determinism
check flags any single device whose own output drifts run-to-run. The reduction dim
K is kept large (the accumulate axis is where a marginal FPU-accumulate bit shows up)
while M and N stay one core's worth of tiles.

A single core switching alone draws almost no current, so this is the di/dt negative
control: the marginal core on b06u02 (device 14, logical core (6,8)) is bit-exact here
and only drifts once a full grid of cores switches beside it. Pair a FAIL of the
full-grid deepseek matmul with a PASS here to separate "core computes wrong" from
"core is marginal under di/dt".
"""
import pytest
from loguru import logger

from tests.didt.op_test_base import OpTestBase, OpParameter
import ttnn
from models.common.utility_functions import is_blackhole, skip_for_wormhole_b0

# Reduction depth (accumulate axis). One core owns a per_core_M x per_core_N tile block.
K = 7168
PER_CORE_M = 6
PER_CORE_N = 6

MESH_DEVICE_PARAMS = [
    pytest.param(1, id="1chips"),
    pytest.param((8, 4), id="galaxy"),
]

# Logical worker core to confine the matmul to. (6,8) is the marginal core on b06u02;
# (0,0) is a healthy reference for contrast.
CORE_PARAMS = [
    pytest.param((6, 8), id="core_6_8"),
    pytest.param((0, 0), id="core_0_0"),
]


@skip_for_wormhole_b0("Grid position (6,8) and the fault under test are Blackhole-only")
@pytest.mark.parametrize("core", CORE_PARAMS)
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_single_core_matmul(mesh_device, core, didt_workload_iterations, determinism_check_interval):
    cx, cy = core
    core_coord = ttnn.CoreCoord(cx, cy)
    allowed_worker_cores = ttnn.CoreRangeSet({ttnn.CoreRange(core_coord, core_coord)})

    M = PER_CORE_M * 32
    N = PER_CORE_N * 32
    logger.info(f"Single-core matmul on logical core ({cx},{cy}): [1,1,{M},{K}] @ [1,1,{K},{N}]")

    dram = ttnn.DRAM_MEMORY_CONFIG
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=3,
        per_core_M=PER_CORE_M,
        per_core_N=PER_CORE_N,
        transpose_mcast=False,
        fused_activation=None,
        allowed_worker_cores=allowed_worker_cores,
    )

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    test = OpTestBase(
        mesh_device,
        OpParameter([1, 1, M, K], ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, dram),  # activations
        [
            OpParameter([1, 1, K, N], ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT, dram),  # weights
        ],
        out_mem_config=dram,
        out_dtype=ttnn.DataType.BFLOAT16,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
    )
    test.run_op_test()
