# SPDX-License-Identifier: Apache-2.0
"""Can the streaming matmul write straight into a standard [32,32]-tiled tensor?

If it can, the reverse bridge disappears entirely and no new micro-op is needed.

The idea: the DST register is physically 32x32 regardless of the logical tile
size, and with m=1 the matmul result lands in row 0. If the output CB is declared
with a [32,32] tile, pack_tile should write a full 1024-element tile whose row 0
is the answer and whose rows 1..31 are junk. That is exactly the batch-1 padding
contract tt_transformers already uses, so stock eltwise ops downstream would be
happy.

Costs 32x the output write bandwidth (~1.8 us at N=14336), which is cheap against
the ~31 us the streaming kernel saves.
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

K, N = 3584, 14336


def test_standard_output_tensor(device):
    tile_w = 32
    tiny = ttnn.Tile([1, tile_w])
    cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(cores)
    num_banks = device.dram_grid_size().x
    n_padded = pad_to_dram_banks(N, tile_w, tile_w * num_banks)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])

    act = torch.randn(1, 1, 32, K)
    act[:, :, 1:, :] = 0
    x = ttnn.from_torch(
        act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
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

    # forward bridge, all on device
    row = ttnn.untilize_with_unpadding(x, [0, 0, 0, K - 1])
    rep = ttnn.repeat(row, ttnn.Shape([1, 1, num_cores, 1]))
    in0 = ttnn.to_memory_config(
        rep,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, [1, K], ttnn.ShardOrientation.ROW_MAJOR),
        ),
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

    # THE TEST: a bog-standard [32,32]-tiled output tensor, exactly what stock
    # eltwise ops downstream expect.
    out_std = ttnn.from_torch(
        torch.zeros(1, 1, 32, n_padded),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, (32, n_padded // num_cores), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    logger.info(f"STDOUT out tensor tile={list(out_std.get_tile().tile_shape)} shape={out_std.shape}")

    try:
        res = DRAMStreamingMatmul.op(
            in0,
            w_t,
            out_std,
            fp32_dest_acc_en=False,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            subblock_k=subblock_k,
            fused_activation=None,
            num_loop_iters=1,
            working_buf_tensor=working,
            in0_tile=tiny,
        )
        ttnn.synchronize_device(device)
    except Exception as e:
        pytest.fail(f"STDOUT standard output tensor rejected: {type(e).__name__}: {str(e)[:300]}")

    golden = DRAMStreamingMatmul.golden(act[:, :, :1, :], w, None)
    got = ttnn.to_torch(res)
    ok, msg = comp_pcc(golden[..., :N], got[:, :, :1, :N], 0.97)
    logger.info(f"STDOUT row-0 PCC with standard [32,32] output: {msg}  ({'OK' if ok else 'BAD'})")
    assert ok, f"row 0 of the standard output tensor is wrong: {msg}"
