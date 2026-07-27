# SPDX-License-Identifier: Apache-2.0
"""Second attempt at the on-device bridge, via a persistent tiny-tile buffer.

Attempt 1 died at `ttnn.tilize`: "Physical shard shape (1, 3584) must be tile
{32,32} sized". No device op lets you choose a [1,32] output tile -- the only way
to get one is host-side `from_torch(tile=...)`.

Way around it: allocate the [1,32]-tiled in0 ONCE at model init (host-side, where
the custom tile is legal), then each decode step copy the current activation row
into that existing buffer. The destination already carries the right spec, so no
op has to invent it. The streaming matmul already uses this pattern for its output.

A [1,32] tile is 2 faces of [1,16] = 32 contiguous elements, i.e. byte-identical
to a row-major row, so the copy should be a straight memcpy if ttnn permits it.
Tries several ways in, and reports which (if any) works.
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


def _grid(device):
    cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    return cores, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])


def test_persistent_buffer_bridge(device):
    tile_w = 32
    cores, core_grid = _grid(device)
    num_cores = len(cores)

    # Persistent tiny-tile in0, allocated once. In the real model this lives on the
    # MLP module and is reused every token.
    in0_persist = ttnn.from_torch(
        torch.zeros(1, 1, num_cores, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, [1, K], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=ttnn.Tile([1, tile_w]),
    )
    logger.info(f"PERSIST in0 tile={list(in0_persist.get_tile().tile_shape)} shape={in0_persist.shape}")

    # A realistic decode activation: batch-1 in a 32-row padded tile.
    act = torch.randn(1, 1, 32, K)
    act[:, :, 1:, :] = 0
    x = ttnn.from_torch(
        act,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    row = ttnn.untilize_with_unpadding(x, [0, 0, 0, K - 1])  # [1,1,1,K] row-major
    rep = ttnn.repeat(row, ttnn.Shape([1, 1, num_cores, 1]))  # [1,1,8,K] row-major
    logger.info(f"SRC rep shape={rep.shape} layout={rep.layout} tile={list(rep.get_tile().tile_shape)}")

    worked = None

    # A) copy row-major source straight into the tiny-tile destination
    try:
        ttnn.copy(rep, in0_persist)
        ttnn.synchronize_device(device)
        worked = "ttnn.copy(rowmajor -> tinytile)"
        logger.info(f"OK   {worked}")
    except Exception as e:
        logger.info(f"FAIL ttnn.copy(rowmajor -> tinytile): {type(e).__name__}: {str(e)[:160]}")

    # B) reshard the source to match in0's memory config first, then copy
    if worked is None:
        try:
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core_grid, [1, K], ttnn.ShardOrientation.ROW_MAJOR),
            )
            rep_sh = ttnn.to_memory_config(rep, mc)
            ttnn.copy(rep_sh, in0_persist)
            ttnn.synchronize_device(device)
            worked = "reshard + ttnn.copy"
            logger.info(f"OK   {worked}")
        except Exception as e:
            logger.info(f"FAIL reshard + ttnn.copy: {type(e).__name__}: {str(e)[:160]}")

    # C) assign / clone style
    if worked is None:
        for name, fn in (
            ("ttnn.assign", lambda: ttnn.assign(rep, in0_persist)),
            ("ttnn.experimental.typecast", lambda: ttnn.copy(ttnn.to_layout(rep, ttnn.TILE_LAYOUT), in0_persist)),
        ):
            try:
                fn()
                ttnn.synchronize_device(device)
                worked = name
                logger.info(f"OK   {worked}")
                break
            except Exception as e:
                logger.info(f"FAIL {name}: {type(e).__name__}: {str(e)[:160]}")

    if worked is None:
        pytest.fail("BLOCKER: no device op could write into a pre-allocated [1,32]-tiled buffer")

    # Did the bytes land correctly? Compare row 0 against the source activation.
    got = ttnn.to_torch(in0_persist)
    ok, msg = comp_pcc(act[:, :, :1, :], got[:, :, :1, :], 0.99)
    logger.info(f"BRIDGE data PCC (row 0): {msg}   via {worked}")
    assert ok, f"bridge moved bytes but they are wrong: {msg}"

    # Now run the real matmul off that buffer.
    num_banks = device.dram_grid_size().x
    n_padded = pad_to_dram_banks(N, tile_w, tile_w * num_banks)
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
        in0_persist,
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
    logger.info(f"BRIDGE end-to-end matmul PCC: {msg}")
    assert ok, f"matmul off bridged buffer is wrong: {msg}"
