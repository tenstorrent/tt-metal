# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The served KV cache of one layer and the chunk geometry of the SP reshuffle.

Per chip (SP row r = mesh axis 0, TP column c = axis 1): key ``[slots, local_kv_heads, 7040, 192]`` and value
``[slots, local_kv_heads, 7040, 128]`` bfloat8_b, TILE, DRAM ND-sharded ``[1, 1, 32, width]`` ROUND_ROBIN_1D over all
banks (one 32-token record of one head contiguous in one bank, serving_contract.md "KV cache"), allocated on the device
and zero-filled. The sequence is block-cyclic over SP: global position g is on SP row ``(g // 640) % 8`` at local row
``(g // 5120) * 640 + g % 640`` (chunk 5120, 640 rows per SP row per chunk). Heads are TP-local (column c holds global
KV heads ``c * local_kv_heads ..``).
"""

from __future__ import annotations

import torch

import ttnn

TILE = ttnn.TILE_SIZE


def served_memory_config(mesh, width: int) -> ttnn.MemoryConfig:
    """One 32-token record per round-robin DRAM shard."""
    banks = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(mesh.dram_grid_size().x)]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, TILE, width],
            grid=banks,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def allocate_served(mesh, shape, width: int, dtype=ttnn.bfloat8_b):
    """A zero-filled per-chip ``shape`` cache on the device (no host staging)."""
    from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

    cache = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mesh, served_memory_config(mesh, width)
    )
    DRAMZeroFill.op(cache)
    dist = ttnn.MeshShape(*tuple(mesh.shape))
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
    cache.update_tensor_topology(
        ttnn.TensorTopology(dist, [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], coords)
    )
    return cache


class LayerKV:
    """The served key / value caches of one layer (``kv_tensors`` of the serving contract)."""

    def __init__(
        self, mesh, slots: int, local_heads: int, local_seq: int, k_dim: int, v_dim: int, attn_v_dim: int | None = None
    ):
        self.slots, self.local_heads, self.local_seq = slots, local_heads, local_seq
        self.key = allocate_served(mesh, [slots, local_heads, local_seq, k_dim], k_dim)
        self.value = allocate_served(mesh, [slots, local_heads, local_seq, v_dim], v_dim)
        # sliding layers: value zero-padded to attn_v_dim, read only by the SDPA (its sliding path needs V head dim ==
        # QK head dim); same layout as the served caches, never served (not in tensors())
        self.attn_value = (
            allocate_served(mesh, [slots, local_heads, local_seq, attn_v_dim], attn_v_dim) if attn_v_dim else None
        )

    def tensors(self) -> dict:
        return {"key": self.key, "value": self.value}

    def zero(self) -> None:
        """Zero every cache of the layer in place on the device (a new sequence)."""
        from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

        for t in (self.key, self.value, self.attn_value):
            if t is not None:
                DRAMZeroFill.op(t)


def chunk_positions(start: int, sp: int, chunk: int) -> torch.Tensor:
    """[sp, chunk // sp] int64: the absolute position of every local row of a chunk at ``start`` (any 32-aligned
    start), as the server's block-cyclic reshuffle lays it out: SP row r holds the chunk's positions g with
    ``(g // W) % sp == r`` (W = chunk // sp) in increasing order, its real rows first."""
    w = chunk // sp
    g = torch.arange(start, start + chunk)
    rows = (g // w) % sp
    return torch.stack([g[rows == r] for r in range(sp)])
