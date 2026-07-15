# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# SYNTHETIC matmul to demonstrate the compute-helper's TRM / larger-subblock lever WORKING on
# Blackhole. Clones the FF1 block-sharded 2D-mcast setup but sets per_core_N to a PRIME > DST
# (default 11). That forces main's SBM validator (out_subblock_w == per_core_N || out_subblock_h == 1)
# to out_subblock 1x1 (w=11 exceeds the DST=8 cap; 11's only other divisor is 1), i.e. one
# matmul_block+pack call per output tile. The helper's TRM (tile_pack_row_major) relaxes that
# constraint, so it can pack out_subblock 4x1 (h>1, w<per_core_N) — 4x the subblock volume, 4x
# fewer matmul_block/pack calls. Everything (shape, K, grid, l1_acc) is identical; only the
# subblock shape + TRM differ, isolating the lever.
#   env: SYN_PCN (per_core_N, default 11), SYN_SBH, SYN_SBW (out_subblock h/w), SYN_TRM (0/1)
import os
from loguru import logger
import pytest
import ttnn

from tests.didt.op_test_base import OpTestBase, OpParameter, get_mesh_grid_size
from models.common.utility_functions import is_blackhole


def _e(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


class SynthMMTest(OpTestBase):
    pass


@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chips")], indirect=["mesh_device"])
def test_synth_mm(mesh_device, didt_workload_iterations, determinism_check_interval):
    per_core_M = 4
    per_core_N = _e("SYN_PCN", 11, int)
    in0_block_w = _e("SYN_K", 3, int)
    out_subblock_h = _e("SYN_SBH", 1, int)
    out_subblock_w = _e("SYN_SBW", 1, int)
    tile_pack_row_major = _e("SYN_TRM", "0") == "1"

    grid = get_mesh_grid_size(mesh_device)
    logger.info(
        f"synth_mm on {grid} cores: per_core={per_core_M}x{per_core_N} "
        f"subblock={out_subblock_h}x{out_subblock_w} TRM={tile_pack_row_major}"
    )

    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))
    in0_shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet({core_range}), [32 * per_core_M, 32 * 18], ttnn.ShardOrientation.ROW_MAJOR
    )  # 32*18 = 576 datums K per core (in0_block_w=3 -> 6 K-blocks), mirrors FF1
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_shard)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )
    # Set TRM after construction (only when requested) so the test also loads on a main baseline
    # whose config type predates the tile_pack_row_major field. main runs TRM=off and never touches it.
    if tile_pack_row_major:
        program_config.tile_pack_row_major = True
    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    in0_shape = [1, 1, 32 * per_core_M * grid.y, 576 * grid.x]
    in1_shape = [1, 1, 576 * grid.x, 32 * per_core_N * grid.x]

    SynthMMTest(
        mesh_device,
        OpParameter(in0_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, in0_mem_config),
        [OpParameter(in1_shape, ttnn.DataType.BFLOAT8_B, ttnn.TILE_LAYOUT, in1_mem_config)],
        out_mem_config=out_mem_config,
        out_dtype=ttnn.DataType.BFLOAT16,
        program_config=program_config,
        compute_config=compute_config,
        loop_count=didt_workload_iterations,
        determinism_check_enabled=determinism_check_interval > 0,
        determinism_check_interval=determinism_check_interval,
    ).run_op_test()
