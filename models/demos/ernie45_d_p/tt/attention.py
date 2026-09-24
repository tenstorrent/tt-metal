# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill GQA attention, TP4: chip c owns Q heads 5c..5c+4 and KV head c.

KV cache (dev layout, P2.6-P2.14): per layer a paged-shaped tensor [max_seq/B, 1, B, D] per chip.
With one KV head per chip this is bit-identical to a contiguous [1, 1, max_seq, D] cache, so the
identity page table [0, 1, ..., max_seq/B - 1] turns it into a plain contiguous cache while letting us use
paged_fill_cache (write at a chunk offset) and chunked_scaled_dot_product_attention (read the prefix).
P2.15 converts to the prefill-server contract layout.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig, LayerWeights
from models.demos.ernie45_d_p.tt.common import COMPUTE_HIFI2, COMPUTE_HIFI4, cache_name, shard
from models.demos.ernie45_d_p.tt.ops import TtRope, all_reduce

NUM_CHIPS = 4
KV_BLOCK = 64


class TtKVCache:
    def __init__(self, mesh, cfg: ErnieConfig, max_seq: int, layers: list[int], dtype=ttnn.bfloat16):
        assert max_seq % KV_BLOCK == 0
        self.mesh, self.max_seq, self.dtype = mesh, max_seq, dtype
        nb = max_seq // KV_BLOCK
        shape = [nb, cfg.num_key_value_heads // NUM_CHIPS, KV_BLOCK, cfg.head_dim]
        self.k = {i: ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh) for i in layers}
        self.v = {i: ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh) for i in layers}
        self.page_table_host = torch.arange(nb, dtype=torch.int32)[None]
        self.page_table = self._pt(self.page_table_host)
        self._chunk_pt = {}

    def _pt(self, t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def chunk_page_table(self, start: int, seq: int):
        """Page table covering exactly the blocks of positions [start, start+seq)."""
        key = (start, seq)
        if key not in self._chunk_pt:
            assert start % KV_BLOCK == 0 and seq % KV_BLOCK == 0
            self._chunk_pt = {
                key: self._pt(self.page_table_host[:, start // KV_BLOCK : (start + seq) // KV_BLOCK].contiguous())
            }
        return self._chunk_pt[key]

    def load_prefix(self, i: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Overwrite layer i with host K/V [n_kv_total, L, D] at positions [0, L) (rest zero); chip c gets head c."""
        n_kv, L, D = k.shape
        nb = self.max_seq // KV_BLOCK

        def dev(t):
            full = torch.zeros(n_kv, self.max_seq, D, dtype=torch.bfloat16)
            full[:, :L] = t.to(torch.bfloat16)
            paged = full.reshape(n_kv, nb, KV_BLOCK, D).transpose(0, 1).contiguous()  # [nb, n_kv, B, D]
            return ttnn.from_torch(
                paged,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
            )

        ttnn.deallocate(self.k[i])
        ttnn.deallocate(self.v[i])
        self.k[i], self.v[i] = dev(k), dev(v)

    def to_torch(self, i: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather layer i as [n_kv_total, length, D] host tensors (KV head c from chip c)."""

        def flat(t):
            parts = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]  # each [nb, 1, B, D]
            return torch.cat([p.transpose(0, 1).reshape(p.shape[1], -1, p.shape[-1]) for p in parts], dim=0)[:, :length]

        return flat(self.k[i]), flat(self.v[i])


class TtAttention:
    def __init__(self, mesh, cfg: ErnieConfig, layer: int, w: LayerWeights, rope: TtRope):
        self.mesh, self.cfg, self.layer, self.rope = mesh, cfg, layer, rope
        D, n = cfg.head_dim, NUM_CHIPS
        self.nq, self.nkv = cfg.num_attention_heads // n, cfg.num_key_value_heads // n
        # Fused per-chip QKV weight: [q heads of chip c | k head c | v head c]  -> [n, H, (nq+2nkv)*D]
        wq, wk, wv = w.wq.T, w.wk.T, w.wv.T  # [H, out]
        fused = []
        for c in range(n):
            fused.append(
                torch.cat(
                    [
                        wq[:, c * self.nq * D : (c + 1) * self.nq * D],
                        wk[:, c * self.nkv * D : (c + 1) * self.nkv * D],
                        wv[:, c * self.nkv * D : (c + 1) * self.nkv * D],
                    ],
                    dim=-1,
                )
            )
        self.wqkv = shard(mesh, torch.stack(fused)[:, None], dim=0, cache=cache_name(f"L{layer}", "wqkv"))
        self.wo = shard(mesh, w.wo.T.contiguous()[None, None], dim=-2, cache=cache_name(f"L{layer}", "wo"))
        self.scale = D**-0.5

    def _sdpa_cfg(self, seq: int, start: int):
        q = 256 if seq >= 2048 else 64
        k = 256 if seq >= 2048 else 64
        if start:
            lowbit = start & -start
            q, k = min(q, lowbit), min(k, lowbit)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=False,
        )

    def __call__(self, x, start: int, cache: TtKVCache, debug: dict | None = None):
        """x: [1,1,S,H] replicated (already normed). Returns [1,1,S,H] replicated (all-reduced)."""
        seq = x.shape[-2]
        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=COMPUTE_HIFI2)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.nq,
            num_kv_heads=self.nkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q = self.rope(q, start)
        k = self.rope(k, start)
        if debug is not None:
            debug.update(q=q, k=k, v=v)

        pt = cache.chunk_page_table(start, seq)
        ttnn.experimental.paged_fill_cache(
            cache.k[self.layer], ttnn.typecast(k, cache.dtype) if cache.dtype != k.dtype else k, pt, batch_idx=0
        )
        ttnn.experimental.paged_fill_cache(
            cache.v[self.layer], ttnn.typecast(v, cache.dtype) if cache.dtype != v.dtype else v, pt, batch_idx=0
        )

        prog = self._sdpa_cfg(seq, start)
        if start == 0:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale, program_config=prog, compute_kernel_config=COMPUTE_HIFI4
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q,
                input_tensor_k=cache.k[self.layer],
                input_tensor_v=cache.v[self.layer],
                page_table_tensor=cache.page_table,
                chunk_start_idx=start,
                program_config=prog,
                compute_kernel_config=COMPUTE_HIFI4,
            )
        if debug is None:
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        else:
            debug["sdpa"] = attn
        a = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if debug is None:
            ttnn.deallocate(attn)
        o = ttnn.linear(a, self.wo, compute_kernel_config=COMPUTE_HIFI2)
        ttnn.deallocate(a)
        out = all_reduce(o)
        ttnn.deallocate(o)
        return out
