# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# mm_help profiling harness — isolated single matmul (mcast-reuse 2D) for the
# main-vs-mm_help5 A/B. Underscore-prefixed; run EXPLICITLY under Tracy. Every field
# overridable via MM_* env. Shape = per_core x grid envelope (matches the model's
# dispatched shape for explicit-config traces). Why-fields come from the Tracy CSV.
import os
import pytest
import ttnn
from loguru import logger
from tests.didt.op_test_base import OpTestBase, OpParameter

_DT = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "float32": ttnn.float32}
_FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


def _e(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


def _bool(v):
    return str(v).lower() in ("1", "true", "yes")


CFG = dict(
    gx=_e("MM_GX", 0, int), gy=_e("MM_GY", 0, int),  # 0 -> full device grid
    per_core_M=_e("MM_PCM", 4, int), per_core_N=_e("MM_PCN", 72, int),
    in0_block_w=_e("MM_K_BLK", 3, int),
    out_subblock_h=_e("MM_SBH", 1, int), out_subblock_w=_e("MM_SBW", 8, int),
    in0_shard_k=_e("MM_IN0_SHARD_K", 576, int),
    in0_dtype=_e("MM_IN0_DT", ttnn.bfloat16, lambda v: _DT[v]),
    in1_dtype=_e("MM_IN1_DT", ttnn.bfloat8_b, lambda v: _DT[v]),
    out_dtype=_e("MM_OUT_DT", ttnn.bfloat16, lambda v: _DT[v]),
    fidelity=_e("MM_FID", ttnn.MathFidelity.LoFi, lambda v: _FID[v]),
    fp32_acc=_e("MM_FP32_ACC", False, _bool), l1_acc=_e("MM_L1_ACC", True, _bool),
    gelu=_e("MM_GELU", False, _bool), loops=_e("MM_LOOPS", 8, int),
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_mmhelp_bench(device):
    c = CFG
    dg = device.compute_with_storage_grid_size()
    gx = c["gx"] if c["gx"] else dg.x
    gy = c["gy"] if c["gy"] else dg.y
    logger.info(f"mm_bench grid={gx}x{gy} pcm={c['per_core_M']} pcn={c['per_core_N']} k={c['in0_block_w']} sb={c['out_subblock_h']}x{c['out_subblock_w']}")
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))
    in0_shard = ttnn.ShardSpec(ttnn.CoreRangeSet({core_range}), [32 * c["per_core_M"], c["in0_shard_k"]], ttnn.ShardOrientation.ROW_MAJOR)
    in0_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_shard)
    in1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=c["in0_block_w"],
        out_subblock_h=c["out_subblock_h"], out_subblock_w=c["out_subblock_w"],
        per_core_M=c["per_core_M"], per_core_N=c["per_core_N"], transpose_mcast=False,
        fused_activation=[ttnn.UnaryOpType.GELU, True] if c["gelu"] else None,
    )
    compute_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=c["fidelity"], math_approx_mode=False,
        fp32_dest_acc_en=c["fp32_acc"], packer_l1_acc=c["l1_acc"],
    )
    in0_shape = [1, 1, 32 * c["per_core_M"] * gy, c["in0_shard_k"] * gx]
    in1_shape = [1, 1, c["in0_shard_k"] * gx, 32 * c["per_core_N"] * gx]
    OpTestBase(
        device,
        OpParameter(in0_shape, c["in0_dtype"], ttnn.TILE_LAYOUT, in0_mem),
        [OpParameter(in1_shape, c["in1_dtype"], ttnn.TILE_LAYOUT, in1_mem)],
        out_mem_config=out_mem, out_dtype=c["out_dtype"], program_config=program_config,
        compute_config=compute_config, loop_count=c["loops"],
        determinism_check_enabled=False, determinism_check_interval=0,
    ).run_op_test()
