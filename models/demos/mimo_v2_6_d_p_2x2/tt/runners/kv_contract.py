# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV-cache contract for MiMo-V2.6-Flash-RL on a 2x2 mesh (TP = 4 over the flattened mesh, no SP).

Same per-chip layout as the 1x4 prior (models/demos/mimo_v2_6_d_p/tt/runners/kv_contract.py), with TP rank d on
chip d = 2*row + col (tt/attention.py):
  * full layers:    KV head d on chip d            -> slab columns [0:192] = head d, [192:384] = zero
  * sliding layers: KV heads 2d, 2d+1 on chip d    -> slab columns [0:192] = head 2d, [192:384] = head 2d+1
  * V: each head's first 128 of 192 columns (last 64 zero, padded at the contract write), x attention_value_scale
  * K post-RoPE in rotate-half (HF) order, as the golden
  * per chip K and V [num_users * num_layers, 1, max_seq, 384], bfloat8_b, TILE, DRAM NdShard [1, 1, 32, 384]
    ROUND_ROBIN_1D over the DRAM banks; batch index = slot * num_layers + layer (the gpt_oss_d_p GQA substrate)
  * address table: configs 0..3 = K chip 0..3, 4..7 = V chip 0..3, each a one-chip device group at
    MeshCoordinate(d // 2, d % 2); 32-token entries of 12 bf8 tiles (13056 B); the whole sequence on every chip

What changes against 1x4: the sequence is not sharded over either mesh axis (every chip computes K/V for the whole
chunk), and a 2x2 mesh has no size-1 axis. The gpt_oss_d_p allocator and table builder take the sequence as sharded
over the sp rows, so this module allocates the full-length cache itself and walks its own table.
update_padded_kv_cache always derives a block-cyclic offset along cluster_axis (extent 2 here): with a chunk-aligned
start and kv_actual_global = 2 * start, every chip on the axis writes its whole chunk at local row `start`
(boundary chip 0 at offset 0, the other chip at the same slab base), which is the natural contiguous layout.

The read-back (host, through the table) is the prior's: the config ids and slab columns are identical.
"""

from __future__ import annotations

import socket

import torch

import ttnn

NUM_CHIPS = 4
MESH = (2, 2)
HEAD_DIM = 192
V_HEAD_DIM = 128
SLAB = 2 * HEAD_DIM
BLOCK = 32  # tokens per DRAM bank shard / table entry
WRITE_AXIS = 0  # update_padded_kv_cache's cluster_axis (see module docstring)


def chip_coord(d: int) -> tuple[int, int]:
    """TP rank d -> mesh coordinate (row-major device order of ShardTensorToMesh on the 2x2 mesh)."""
    return d // MESH[1], d % MESH[1]


class MiMoContractKV2x2:
    def __init__(self, mesh, num_layers: int, max_seq: int, num_users: int = 1, dtype=ttnn.bfloat8_b):
        from models.demos.common.prefill.runners.migration import get_num_dram_banks

        assert tuple(mesh.shape) == MESH, f"MiMo 2x2 contract cache is built for a 2x2 mesh, got {mesh.shape}"
        assert max_seq % BLOCK == 0, max_seq
        self.mesh, self.num_layers, self.max_seq, self.num_users, self.dtype = (
            mesh,
            num_layers,
            max_seq,
            num_users,
            dtype,
        )
        self.num_banks = get_num_dram_banks(mesh)
        banks = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(self.num_banks)]
        spec = ttnn.NdShardSpec(
            shard_shape=[1, 1, BLOCK, SLAB],
            grid=ttnn.CoreRangeSet(banks),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)

        def alloc():
            # Same (zeroed) buffer on every chip; what chip d holds is decided by the head-sharded write.
            return ttnn.from_torch(
                torch.zeros(num_users * num_layers, 1, max_seq, SLAB),
                dtype=dtype,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.k, self.v = alloc(), alloc()

    def sink(self, layer: int, start: int, slot: int):
        """kv_sink(k, v) for one (cache layer, chunk, slot): k per chip [1, h, S, 192] (h in {1, 2}), v [1, h, S, 128]
        (or 192 under MIMO_V_PAD=1). Writes the chunk at rows [start, start + S) of batch slot * L + layer."""
        from models.demos.mimo_v2_6_d_p.tt.runners.kv_contract import _slab

        assert 0 <= slot < self.num_users and 0 <= layer < self.num_layers, (slot, layer)

        def write(k, v):
            seq = k.shape[-2]
            assert start % seq == 0, f"contract write needs a chunk-aligned start ({start} % {seq})"
            assert start + seq <= self.max_seq, (start, seq, self.max_seq)
            for cache, t in ((self.k, k), (self.v, v)):
                s = _slab(t)
                src = s if s.dtype == cache.dtype else ttnn.typecast(s, cache.dtype)
                if src is not s:
                    ttnn.deallocate(s)
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache,
                    src,
                    slot_idx=slot,
                    layer_idx=layer,
                    num_layers=self.num_layers,
                    kv_actual_global=MESH[WRITE_AXIS] * start,
                    cluster_axis=WRITE_AXIS,
                )
                ttnn.deallocate(src)

        return write

    def chunk_bytes(self) -> int:
        tile = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[self.dtype]
        return (SLAB // 32) * tile

    def address_table(self, seq_len: int, chunk_size: int):
        """Configs 0..3 K chip 0..3, 4..7 V chip 0..3; one entry per (layer, 32-token position, slot)."""
        from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import _make_config, _stable_config_name

        D = ttnn.experimental.disaggregation
        assert seq_len <= self.max_seq and seq_len % chunk_size == 0 and chunk_size % BLOCK == 0
        specs = [("k", self.k, d) for d in range(NUM_CHIPS)] + [("v", self.v, d) for d in range(NUM_CHIPS)]
        n = len(specs)
        nbytes = self.chunk_bytes()
        table = D.KvChunkAddressTable(
            {
                _stable_config_name(i, n): _make_config(
                    num_layers=self.num_layers, max_seq_len=seq_len, num_users=self.num_users, chunk_size_bytes=nbytes
                )
                for i in range(n)
            }
        )
        for i in range(n):
            assert table.config_name(i) == _stable_config_name(i, n)
        host = socket.gethostname()
        seen = set()
        blocks_per_slot = self.max_seq // BLOCK
        for cid, (_, tensor, d) in enumerate(specs):
            base = tensor.buffer_address()
            fid = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*chip_coord(d)))
            group = table.add_device_group([fid])
            key = (int(fid.mesh_id), int(fid.chip_id))
            if key not in seen:
                table.set_fabric_node_host(fid, host_name=host)
                seen.add(key)
            # ROUND_ROBIN_1D: shard j (row-major over [batch, max_seq / 32]) lives in bank j % B at offset (j // B).
            for slot in range(self.num_users):
                for layer in range(self.num_layers):
                    b = slot * self.num_layers + layer
                    for pos in range(0, seq_len, BLOCK):
                        j = b * blocks_per_slot + pos // BLOCK
                        loc = D.KvCacheLocation()
                        loc.noc_addr = ((j % self.num_banks) << 32) | (base + (j // self.num_banks) * nbytes)
                        loc.size_bytes = nbytes
                        loc.device_group_index = group
                        table.set(layer, pos, slot, loc, cid)
        return table
