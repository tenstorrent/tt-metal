# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV-cache contract for ERNIE-4.5 (GQA, 4 KV heads, head_dim 128) on a 1x4 mesh.

Same physical contract as the gpt_oss_d_p GQA template, reused directly:
  * K and V separate, per chip [num_users * num_layers, 1, max_seq, 128], bfloat8_b, TILE,
    DRAM NdShard [1,1,32,128] ROUND_ROBIN_1D over the 8 DRAM banks; batch index = slot * num_layers + layer
  * chip (TP column) c holds KV head c; sp = 1 (mesh rows), so the sequence is contiguous per chip
  * K is post-RoPE in ERNIE's native interleaved order (= Meta order): NO HF->Meta permutation on readback
  * KV chunk address table: configs 0..3 = K heads 0..3, 4..7 = V heads 0..3, 32-token entries
"""

from __future__ import annotations

import ttnn
from models.demos.gpt_oss_d_p.tt.attention.kv_cache import allocate_kv_cache, write_kv_chunk
from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table

SP_AXIS = 0


class ErnieContractKV:
    def __init__(
        self, mesh, num_layers: int, max_seq: int, num_users: int = 1, head_dim: int = 128, dtype=ttnn.bfloat8_b
    ):
        self.mesh, self.num_layers, self.max_seq, self.num_users, self.head_dim = (
            mesh,
            num_layers,
            max_seq,
            num_users,
            head_dim,
        )
        self.cache = allocate_kv_cache(
            mesh,
            num_layers=num_layers,
            max_seq_len=max_seq,
            sp_axis=SP_AXIS,
            num_users=num_users,
            head_dim=head_dim,
            cache_dtype=dtype,
        )

    def write(self, layer: int, k, v, start: int, slot: int = 0) -> None:
        """k, v: per-chip [1, 1, S, head_dim] (chip c = KV head c), absolute offset `start` (32-aligned)."""
        write_kv_chunk(self.cache, k, v, slot_idx=slot, layer_idx=layer, kv_actual=start, sp_axis=SP_AXIS)

    def address_table(self, seq_len: int, chunk_size: int, num_kv_heads: int = 4):
        return build_kv_chunk_address_table(
            mesh_device=self.mesh,
            kv_cache=self.cache,
            seq_len=seq_len,
            num_layers=self.num_layers,
            mesh_shape=list(self.mesh.shape),
            sp_axis=SP_AXIS,
            num_users=self.num_users,
            chunk_size=chunk_size,
            num_kv_heads=num_kv_heads,
            head_dim=self.head_dim,
        )
