# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 decode matmuls via DeepSeek's dram_streaming_matmul.

tt_transformers carries decode activations as [1, 1, 32, dim]: one real row and
31 rows of padding, because the standard tile is [32,32]. The production
dram_sharded matmul does the MAC work for all 32. That waste is invisible while
DRAM time dominates (bfp8 reaches 89% MBU) and becomes the bottleneck once bfp4
halves the DRAM time (55% MBU).

dram_streaming_matmul works on [1,32] tiny tiles and so does 1 row of work
instead of 32. Measured at gemma2-9B FF1/FF3 shapes with bfp4 weights:

    production dram_sharded            93.9 us
    bridge + streaming matmul          68.8 us   (-25.1 us, PCC 0.993)

The bridge below (extract row 0, replicate per core, reshard) costs 5.7 us of
that. Two details make it work without any new kernels:

  * in0 is handed over ROW_MAJOR and CB0 is told to view it as [1,32] tiles. That
    is sound because a [1,32] tile is 2 faces of [1,16], i.e. 32 contiguous
    values, byte-identical to a row-major row.
  * the output is a bog-standard [32,32]-tiled tensor. DST is physically 32x32
    regardless, so with m=1 the result lands in row 0 and pack writes a full tile
    whose other rows are junk -- exactly the padding contract already in use, so
    stock eltwise ops downstream need no changes.

Enable with TT_STREAM_MM=1. Single device, no prefetcher.
"""

import os

import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul

TILE_W = 32
TINY_TILE = ttnn.Tile([1, TILE_W])


def enabled(args, prefetcher=None):
    """Streaming decode matmuls are opt-in and single-device only for now."""
    return os.environ.get("TT_STREAM_MM", "0") == "1" and getattr(args, "num_devices", 1) == 1 and prefetcher is None


def shuffle_tensor_tiles(tensor, num_banks, tile_size=TILE_W):
    """Reorder tiles within each DRAM bank's shard from row-major to column-major.

    TTNN stores tiles row-major, but the streaming kernel wants the K tiles of a
    given N column contiguous so it can stream them as one burst. Applied once at
    weight-load time.

    Kept identical to the reference in
    deepseek_v3_b1/tests/unit_tests/test_dram_streaming_matmul.py, which is the
    implementation the kernel's own tests validate against.
    """
    orig_shape = tensor.shape
    K = orig_shape[-2]
    N = orig_shape[-1]

    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    needs_padding = n_padded != N

    tensor = tensor.reshape(-1, K, N)
    batch_size = tensor.shape[0]
    if needs_padding:
        tensor = torch.nn.functional.pad(tensor, (0, n_padded - N))

    K_tiles = K // tile_size
    per_N = n_padded // num_banks
    per_N_tiles = per_N // tile_size
    num_tiles_per_shard = K_tiles * per_N_tiles

    tensor = tensor.reshape(batch_size, K, num_banks, per_N)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    shards = tensor.reshape(-1, K, per_N)

    tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
    tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    i = torch.arange(num_tiles_per_shard, device=tensor.device)
    source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)
    shuffled_tiles = tiles[:, source_idx, :, :]

    shuffled_tiles = shuffled_tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size)
    shuffled_tiles = shuffled_tiles.permute(0, 1, 3, 2, 4).contiguous()
    shuffled_shards = shuffled_tiles.reshape(-1, K, per_N)

    shuffled = shuffled_shards.reshape(batch_size, num_banks, K, per_N)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch_size, K, n_padded)

    if needs_padding:
        shuffled = shuffled[:, :, :N]
    return shuffled.reshape(*orig_shape)


def _compute_cores(device):
    cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    return len(cores), grid


class StreamMMContext:
    """Device buffers shared by every decoder layer.

    The streaming op writes into a caller-supplied output tensor, so these are
    allocated once and reused. Sharing across layers keeps peak L1 at what a
    single layer already uses -- the same buffers ttnn.linear would allocate and
    free each layer today.
    """

    def __init__(self, device):
        self.device = device
        self.num_cores, self.core_grid = _compute_cores(device)
        self._out = {}
        self._working = {}
        self._in0 = {}

    def out_buffer(self, n_padded, slot):
        """Standard [32,32]-tiled output, one per concurrently-live result."""
        key = (n_padded, slot)
        if key not in self._out:
            self._out[key] = ttnn.from_torch(
                torch.zeros(1, 1, 32, n_padded),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(self.core_grid, (32, n_padded // self.num_cores), ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )
        return self._out[key]

    def working_buffer(self, subblock_k):
        key = subblock_k
        if key not in self._working:
            width = subblock_k * 3 * TILE_W
            self._working[key] = ttnn.from_torch(
                torch.zeros(1, 1, TILE_W, width * self.num_cores),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(self.core_grid, (TILE_W, width), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                tile=ttnn.Tile([TILE_W, TILE_W]),
            )
        return self._working[key]

    def in0_mem_config(self, k):
        if k not in self._in0:
            self._in0[k] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.core_grid, [1, k], ttnn.ShardOrientation.ROW_MAJOR),
            )
        return self._in0[k]


def bridge_activation(ctx, x, k):
    """[1,1,32,k] standard tiles -> row 0 replicated per core, ROW_MAJOR.

    Left ROW_MAJOR on purpose; the matmul's CB0 views it as [1,32] tiles. Shared
    by every matmul that consumes the same activation, so FF1 and FF3 pay for it
    once.
    """
    row = ttnn.untilize_with_unpadding(x, [0, 0, 0, k - 1])
    rep = ttnn.repeat(row, ttnn.Shape([1, 1, ctx.num_cores, 1]))
    in0 = ttnn.to_memory_config(rep, ctx.in0_mem_config(k))
    ttnn.deallocate(row)
    ttnn.deallocate(rep)
    return in0


def stream_linear(ctx, in0, weight, k, n_padded, slot, math_fidelity=ttnn.MathFidelity.LoFi):
    """One bridged decode matmul. Returns a standard [32,32]-tiled tensor."""
    subblock_k = k // TILE_W // 4
    return DRAMStreamingMatmul.op(
        in0,
        weight,
        ctx.out_buffer(n_padded, slot),
        fp32_dest_acc_en=False,
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        subblock_k=subblock_k,
        fused_activation=None,
        num_loop_iters=1,
        working_buf_tensor=ctx.working_buffer(subblock_k),
        in0_tile=TINY_TILE,
    )
