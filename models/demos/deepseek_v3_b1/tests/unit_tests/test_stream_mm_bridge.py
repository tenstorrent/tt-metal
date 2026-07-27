# SPDX-License-Identifier: Apache-2.0
"""Can a tt_transformers decode activation feed dram_streaming_matmul on-device?

The streaming matmul reaches 458 GB/s on bfp4 at m=1 versus 308 GB/s for the
production dram_sharded kernel at m=32, worth ~+23% end-to-end on gemma2-9B. But
it wants in0 as a [1,32]-tiled row REPLICATED one copy per compute core and
HEIGHT_SHARDED, while tt_transformers decode carries [1,1,32,dim] in standard
[32,32] tiles (batch-1 padded to a full tile).

The microbenchmark built that in0 on the host, which proves nothing about
integration. This builds it from a realistic decode activation using only device
ops, then runs the matmul and PCC-checks it. If this passes, wiring it into
MLP.forward is mechanical. If it fails, the failure is the precise blocker.
"""
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

K, N = 3584, 14336  # gemma2-9B FF1/FF3


def test_bridge(device):
    tile_w = 32
    cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(cores)
    num_banks = device.dram_grid_size().x
    n_padded = pad_to_dram_banks(N, tile_w, tile_w * num_banks)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])

    # --- what tt_transformers decode actually hands us: batch-1 in a padded tile
    act = torch.randn(1, 1, 32, K)
    act[:, :, 1:, :] = 0  # rows 1..31 are padding
    x = ttnn.from_torch(
        act,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # --- weights, prepared once at load time (no runtime cost)
    w = torch.randn(1, 1, K, n_padded)
    w_t = ttnn.from_torch(
        shuffle_tensor_tiles(w, tile_w, num_banks),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
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
                [K, n_padded // num_banks],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    # === THE BRIDGE: [1,1,32,K] standard tiles -> [1,1,1,K] tiny tiles, replicated ===
    steps = {}

    # 1. pull row 0 out of the padded tile
    row = ttnn.untilize_with_unpadding(x, [0, 0, 0, K - 1])
    steps["untilize_with_unpadding"] = (tuple(row.shape), str(row.layout), str(row.dtype))
    logger.info(f"BRIDGE after unpad: shape={row.shape} layout={row.layout}")

    # 2. replicate one copy per compute core
    try:
        rep = ttnn.repeat(row, ttnn.Shape([1, 1, num_cores, 1]))
        steps["repeat"] = (tuple(rep.shape), str(rep.layout))
        logger.info(f"BRIDGE after repeat: shape={rep.shape} layout={rep.layout}")
    except Exception as e:
        pytest.fail(f"BLOCKER at repeat: {type(e).__name__}: {e}")

    # 3. height-shard it one row per core, carrying the [1,32] tile spec.
    #    This is the step most likely to fail: device ops generally do not let you
    #    choose an output tile shape.
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [1, K], ttnn.ShardOrientation.ROW_MAJOR),
    )
    try:
        in0 = ttnn.to_memory_config(rep, mc)
        logger.info(
            f"BRIDGE after reshard: shape={in0.shape} layout={in0.layout} tile={list(in0.get_tile().tile_shape)}"
        )
    except Exception as e:
        pytest.fail(f"BLOCKER at reshard-to-height-sharded: {type(e).__name__}: {e}")

    # 4. the op needs TILE layout with a [1,32] tile. Row-major -> tiny tile.
    if in0.layout != ttnn.TILE_LAYOUT:
        try:
            in0 = ttnn.tilize(in0, use_multicore=True)
            logger.info(f"BRIDGE after tilize: tile={list(in0.get_tile().tile_shape)} layout={in0.layout}")
        except Exception as e:
            pytest.fail(f"BLOCKER at tilize to [1,32]: {type(e).__name__}: {e}")

    got_tile = list(in0.get_tile().tile_shape)
    logger.info(f"BRIDGE final in0 tile={got_tile} (need [1,32])")

    # --- output buffer the op writes into
    out_t = ttnn.from_torch(
        torch.zeros(1, 1, 1, n_padded),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, (1, n_padded // num_banks), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=ttnn.Tile([1, tile_w]),
    )

    subblock_k = K // tile_w // 4
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

    res = DRAMStreamingMatmul.op(
        in0,
        w_t,
        out_t,
        fp32_dest_acc_en=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        subblock_k=subblock_k,
        fused_activation=None,
        num_loop_iters=1,
        working_buf_tensor=working,
    )
    ttnn.synchronize_device(device)

    golden = DRAMStreamingMatmul.golden(act[:, :, :1, :], w, None)
    ok, msg = comp_pcc(golden, ttnn.to_torch(res), 0.98)
    logger.info(f"BRIDGE end-to-end PCC: {msg}")
    assert ok, f"bridge produced wrong numerics: {msg}"
