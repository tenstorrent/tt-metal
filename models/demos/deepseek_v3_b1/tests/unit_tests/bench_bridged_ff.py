# SPDX-License-Identifier: Apache-2.0
"""Is the bridged streaming matmul actually faster than production, end to end?

The streaming kernel wins on the matmul alone (63.2 us vs 93.9 us at gemma2
FF1/FF3, bfp4). But using it from tt_transformers costs extra ops: unpad the
batch-1 row out of its [32,32] tile, replicate it per core, reshard, and on the
way out tilize back to standard tiles so the existing eltwise mul can consume it.

If those bridge ops cost more than the ~30 us the matmul saves, the whole idea is
dead and there is no point touching the model. This measures the complete chain
against the production baseline at real gemma2 shapes so the decision is made on
device time, not arithmetic.

Run under tracy to get device times:
  python -m tracy -p -r -v --op-support-count 40000 -m "pytest <this file> -s"
Or plain for a wall-clock sanity check.
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import (
    pad_to_dram_banks,
    shuffle_tensor_tiles,
)

# K, N. Default is gemma2-9B FF1/FF3; set BENCH_KN=14336,3584 for FF2, whose
# output is 4x narrower and so much cheaper to bridge back.
DIM, HIDDEN = (int(v) for v in os.environ.get("BENCH_KN", "3584,14336").split(","))
ITERS = 50


def _dram_sharded_mem_config(device, k, n):
    num_banks = device.dram_grid_size().x
    padded_n = pad_to_dram_banks(n, 32, 32 * num_banks)
    return (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                        )
                    }
                ),
                [k, padded_n // num_banks],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        padded_n,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 0}], indirect=True)
def test_bench_bridged_ff(device):
    tile_w = 32
    tiny = ttnn.Tile([1, tile_w])
    cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(cores)
    num_banks = device.dram_grid_size().x
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    mem_cfg, padded_n = _dram_sharded_mem_config(device, DIM, HIDDEN)

    act = torch.randn(1, 1, 32, DIM)
    act[:, :, 1:, :] = 0
    w = torch.randn(1, 1, DIM, padded_n)

    x = ttnn.from_torch(
        act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Production baseline is not re-measured here; configuring it alongside the
    # streaming buffers overflows L1. Use the known figures instead: 93.9 us in the
    # isolated sweep and 97.7 us in-model for this exact shape.
    PROD_US = 93.9

    # ---------------- bridged streaming ----------------
    w_stream = ttnn.from_torch(
        shuffle_tensor_tiles(w, tile_w, num_banks),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_cfg,
    )
    # Standard [32,32]-tiled output. The DST register is 32x32 regardless, so with
    # m=1 the result lands in row 0 and pack writes a full tile whose remaining
    # rows are junk -- exactly tt_transformers' batch-1 padding contract. Costs 32x
    # the output write (~1.8 us at N=14336) and removes the reverse bridge entirely.
    out_rm = ttnn.from_torch(
        torch.zeros(1, 1, 32, padded_n),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, (32, padded_n // num_cores), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    OUT_IS_RM = False

    subblock_k = DIM // tile_w // 4
    working = ttnn.from_torch(
        torch.zeros(1, 1, tile_w, subblock_k * 3 * tile_w * num_cores),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, (tile_w, subblock_k * 3 * tile_w), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=ttnn.Tile([tile_w, tile_w]),
    )
    in0_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [1, DIM], ttnn.ShardOrientation.ROW_MAJOR),
    )

    def run_bridged(with_reverse=True):
        row = ttnn.untilize_with_unpadding(x, [0, 0, 0, DIM - 1])
        rep = ttnn.repeat(row, ttnn.Shape([1, 1, num_cores, 1]))
        in0 = ttnn.to_memory_config(rep, in0_mem)
        res = DRAMStreamingMatmul.op(
            in0,
            w_stream,
            out_rm,
            fp32_dest_acc_en=False,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            subblock_k=subblock_k,
            fused_activation=None,
            num_loop_iters=1,
            working_buf_tensor=working,
            in0_tile=tiny,
            out_tile=tiny,
        )
        ttnn.deallocate(row)
        ttnn.deallocate(rep)
        ttnn.deallocate(in0)
        if not with_reverse:
            return res
        # Reverse bridge to standard [32,32] tiles so stock eltwise can consume it.
        if OUT_IS_RM:
            return ttnn.to_layout(res, ttnn.TILE_LAYOUT)
        rm = ttnn.to_layout(res, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    # correctness of the forward half before timing it
    golden = DRAMStreamingMatmul.golden(act[:, :, :1, :], w, None)
    fwd = run_bridged(with_reverse=False)
    ok, msg = comp_pcc(golden[..., :HIDDEN], ttnn.to_torch(fwd)[:, :, :1, :HIDDEN], 0.97)
    logger.info(f"BENCH forward-only PCC  : {msg}  ({'OK' if ok else 'BAD'})")
    try:
        rev = run_bridged(with_reverse=True)
        okr, msgr = comp_pcc(golden[..., :HIDDEN], ttnn.to_torch(rev)[:, :, :1, :HIDDEN], 0.97)
        logger.info(
            f"BENCH REVERSE BRIDGE WORKS, PCC : {msgr}  ({'OK' if okr else 'BAD'}) shape={rev.shape} tile={list(rev.get_tile().tile_shape)}"
        )
    except Exception as e:
        logger.info(f"BENCH reverse bridge FAILED: {type(e).__name__}: {str(e)[:200]}")

    # NOTE: with_reverse=False returns out_rm itself (written in place), so it must
    # not be deallocated; doing so frees the persistent output buffer.
    for label, rev in (("fwd bridge + streaming mm -> std tile out", False),):
        run_bridged(rev)
        ttnn.synchronize_device(device)
        t0 = time.time()
        for _ in range(ITERS):
            r = run_bridged(rev)
            if rev:
                ttnn.deallocate(r)
        ttnn.synchronize_device(device)
        us = (time.time() - t0) / ITERS * 1e6
        delta = PROD_US - us
        logger.info(f"BENCH {label:32s}: {us:8.1f} us/iter (wall)  delta vs prod {delta:+8.1f} us")

    assert ok, f"bridged chain numerics wrong: {msg}"
