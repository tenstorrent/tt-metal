# SPDX-License-Identifier: Apache-2.0
"""Feed dram_streaming_matmul a row-major activation via CB tile aliasing.

Earlier attempts (test_stream_mm_bridge.py / _bridge2.py) concluded this was
blocked because no ttnn op can produce a [1,32]-tiled tensor and every copy path
enforces tile-shape equality. That conclusion was wrong about the remedy: the op
itself already demonstrates the trick, where CB4/6/7 view a [1,32]-tiled
mm_out_tensor's memory as [16,16] simply by overwriting
`format_descriptors[0].tile` on a CB descriptor built from that tensor.

So no Tensor-level retile is needed. We hand the op a ROW_MAJOR activation and
tell CB0 to view it as [1,32] tiles. That is legal because a [1,32] tile is 2
faces of [1,16] = 32 contiguous values, byte-identical to a row-major row
(measured in test_tile_alias_equiv.py).

Every step here runs on device, which is the property the bridge needed and
could not previously get.
"""
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


def test_cb_alias_bridge(device):
    tile_w = 32
    tiny_tile = ttnn.Tile([1, tile_w])
    cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(cores)
    num_banks = device.dram_grid_size().x
    n_padded = pad_to_dram_banks(N, tile_w, tile_w * num_banks)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])

    # A realistic tt_transformers decode activation: batch-1 padded into a 32-row tile.
    act = torch.randn(1, 1, 32, K)
    act[:, :, 1:, :] = 0
    x = ttnn.from_torch(
        act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # === bridge, all on device ===
    row = ttnn.untilize_with_unpadding(x, [0, 0, 0, K - 1])  # [1,1,1,K] row-major
    rep = ttnn.repeat(row, ttnn.Shape([1, 1, num_cores, 1]))  # [1,1,8,K] row-major
    in0 = ttnn.to_memory_config(
        rep,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, [1, K], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    logger.info(f"ALIAS in0 layout={in0.layout} declared_tile={list(in0.get_tile().tile_shape)} -> CB0 views [1,32]")

    # weights, prepared once at load time
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
        tile=tiny_tile,
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
        in0_tile=tiny_tile,
    )
    ttnn.synchronize_device(device)

    golden = DRAMStreamingMatmul.golden(act[:, :, :1, :], w, None)
    ok, msg = comp_pcc(golden, ttnn.to_torch(res), 0.98)
    logger.info(f"ALIAS end-to-end PCC (row-major in0 viewed as [1,32]): {msg}")
    assert ok, f"CB-aliased in0 gave wrong numerics: {msg}"
