# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The served KV cache of the attention layers of Qwen3.8-27B on the 8 x 4 mesh.

Per chip (SP row r, TP column h): KV head h, the sequence block-cyclic over the 8 SP rows (cache local row lr holds
position (lr // 640) * 5120 + r * 640 + lr % 640, the update_padded_kv_cache writer's layout), max_seq / 8 rows.
bfloat8_b TILE, ND-sharded DRAM with [1, 1, 32, head_dim] shards ROUND_ROBIN_1D over the DRAM banks (one 32-token
record in one bank), batch dim slot * num_layers + layer. K is stored after k_norm and RoPE in HF (rotate-half) order.
"""

import torch

import ttnn

RECORD = 32  # tokens per DRAM-bank record


def blockcyclic_positions(sp: int, chunk: int, max_seq: int) -> torch.Tensor:
    """Global position held by each row of the SP-contiguous split of the cache's sequence dim (chip-major)."""
    local, chunk_local = max_seq // sp, chunk // sp
    c = torch.arange(sp).repeat_interleave(local)
    lr = torch.arange(local).repeat(sp)
    return (lr // chunk_local) * chunk + c * chunk_local + lr % chunk_local


class TtKVCache:
    """K and V caches [num_slots * num_layers, 1, max_seq / sp, head_dim] per chip."""

    def __init__(
        self,
        mesh,
        n_kv: int,
        head_dim: int,
        max_seq: int,
        chunk: int,
        num_slots: int = 1,
        num_layers: int = 1,
        dtype=ttnn.bfloat8_b,
    ):
        self.mesh, self.n_kv, self.head_dim = mesh, n_kv, head_dim
        self.sp = mesh.shape[0]
        assert max_seq % (RECORD * self.sp) == 0 and max_seq % chunk == 0, (max_seq, chunk)
        self.max_seq, self.chunk, self.num_slots, self.num_layers, self.dtype = (
            max_seq,
            chunk,
            num_slots,
            num_layers,
            dtype,
        )
        banks = mesh.dram_grid_size().x
        nd = ttnn.NdShardSpec(
            shard_shape=[1, 1, RECORD, head_dim],
            grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)]),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        self.memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd)
        zeros = torch.zeros(num_slots * num_layers, n_kv, max_seq, head_dim)
        self.k, self.v = self._upload(zeros), self._upload(zeros)

    def _upload(self, t: torch.Tensor) -> ttnn.Tensor:
        """Host [batch, n_kv, max_seq, head_dim] in block-cyclic row order -> the cache (head h on TP column h)."""
        return ttnn.from_torch(
            t,
            device=self.mesh,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 1)),
        )

    def load_prefix(self, key: torch.Tensor, value: torch.Tensor, length: int) -> None:
        """Harness boundary only (a whole-cache upload): every slot / layer holds key / value [n_kv, seq, head_dim]
        (natural order) at positions [0, length), zeros after."""
        pos = blockcyclic_positions(self.sp, self.chunk, self.max_seq)
        keep = (pos < length).float()[:, None]
        idx = pos.clamp_max(max(length - 1, 0))
        B = self.num_slots * self.num_layers
        for name, src in (("k", key), ("v", value)):
            host = torch.zeros(self.n_kv, self.max_seq, self.head_dim)
            if length:
                host = src[:, :length].float()[:, idx] * keep
            old = getattr(self, name)
            setattr(self, name, self._upload(host[None].expand(B, -1, -1, -1).contiguous()))
            ttnn.deallocate(old)

    def batch(self, slot: int, layer: int) -> int:
        return slot * self.num_layers + layer

    def _read(self, t: ttnn.Tensor) -> torch.Tensor:
        """The whole cache [batch, n_kv, max_seq, head_dim] in natural position order (harness boundary only)."""
        flat = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        cat = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 1))
        host = ttnn.to_torch(flat, mesh_composer=cat).float()
        ttnn.deallocate(flat)
        out = torch.empty_like(host)
        out[:, :, blockcyclic_positions(self.sp, self.chunk, self.max_seq)] = host
        return out

    def to_torch(self, slot: int, layer: int, length: int) -> dict:
        """key / value [n_kv, length, head_dim] of one (slot, layer), natural order (harness boundary only)."""
        b = self.batch(slot, layer)
        return {name: self._read(t)[b, :, :length].clone() for name, t in (("key", self.k), ("value", self.v))}

    def load_layer_prefix(self, slot: int, layer: int, key: torch.Tensor, value: torch.Tensor, length: int) -> None:
        """Harness boundary only: (slot, layer) holds key / value [n_kv, seq, head_dim] (natural order) at positions
        [0, length), zeros after; the other slots / layers keep their contents."""
        b = self.batch(slot, layer)
        pos = blockcyclic_positions(self.sp, self.chunk, self.max_seq)
        for name, src in (("k", key), ("v", value)):
            old = getattr(self, name)
            nat = self._read(old)
            nat[b] = 0
            nat[b, :, :length] = src[:, :length].float()
            setattr(self, name, self._upload(nat[:, :, pos].contiguous()))
            ttnn.deallocate(old)

    def cache_tensors(self) -> dict:
        return {"key": self.k, "value": self.v}

    def free(self) -> None:
        for t in (self.k, self.v):
            ttnn.deallocate(t)
