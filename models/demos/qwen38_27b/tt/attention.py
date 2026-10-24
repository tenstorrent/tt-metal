# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gated full attention of Qwen3.8-27B (layers 3, 7, ...), chunked prefill on the 8x4 mesh (SP 8 on axis 0, TP 4 on
axis 1).

Per chip: the chunk's 640 local rows (mesh row r holds positions with (p // 640) % 8 == r) and one TP column's heads:
6 query heads (each with its output gate) and 1 kv head (24 q / 4 kv heads, head_dim 256).

    x [1, 1, 640, 5120] bf16
      -> q|k|v|gate linear (column-parallel; per column q 6x256 | k 256 | v 256 | gate 6x256 = 3584 outputs)
      -> nlp_create_qkv_heads: q [1, 6, 640, 256], k, v [1, 1, 640, 256]
      -> q_norm / k_norm: rms_norm per head with the (1 + w) gamma
      -> partial RoPE (rotate-half on the first 64 dims, rotary_embedding_hf) from block-cyclic cos/sin tables built at
         load time: every row's table holds the positions its cache rows hold, so one slice at start / 8 serves all rows
      -> K, V (bfp8) written into the slot's served cache at the chunk's block-cyclic rows (update_padded_kv_cache)
      -> ring_joint_scaled_dot_product_attention over axis 0: causal over [0, start + 5120) with absolute positions,
         the ring gathers the other rows' cache rows itself
      -> concat heads * sigmoid(gate) -> o_proj (row-parallel) -> all_reduce axis 1 (fp32) -> bf16 [1, 1, 640, 5120].

The KV cache is the served one (serving contract): per chip [slots * layers, 1, max_seq / 8, 256] bfloat8_b, ND-sharded
[1, 1, 32, 256] ROUND_ROBIN_1D over the DRAM banks, user-major (slot s, layer l at batch s * layers + l). Local row lr
of mesh row r holds position (lr // 640) * 5120 + r * 640 + lr % 640.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .norm import compute_config
from .settings import settings

TILE = 32
KV_RECORD = 32  # tokens of one served KV record, contiguous in one DRAM bank


def blockcyclic_positions(sp: int, chunk: int, capacity: int) -> torch.Tensor:
    """The absolute position each cache row holds, device-major: [sp * capacity / sp] (the update_padded_kv_cache
    writer's layout for chunk-aligned starts)."""
    local, part = capacity // sp, chunk // sp
    c = torch.arange(sp).repeat_interleave(local)
    lr = torch.arange(local).repeat(sp)
    return (lr // part) * chunk + c * part + lr % part


def kv_memory_config(mesh, head_dim: int):
    """The served cache layout: one 32-token record per ND shard, round-robin over the DRAM banks."""
    banks = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(mesh.dram_grid_size().x)]
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, KV_RECORD, head_dim],
            grid=banks,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


@dataclass
class KvCache:
    k: ttnn.Tensor  # [slots * layers, 1, capacity / sp, head_dim] bfloat8_b per chip, the served layout
    v: ttnn.Tensor
    num_layers: int
    capacity: int  # tokens per slot (global)


class _Ring:
    """The ring SDPA's persistent CCL state: three global semaphores on the CCL column and the gathered K/V scratch,
    one per (mesh, capacity), shared by every attention layer."""

    _by_key: dict = {}

    @classmethod
    def shared(cls, mesh, capacity: int, head_dim: int):
        key = (mesh.id(), capacity, head_dim)
        if key not in cls._by_key:
            cls._by_key[key] = cls(mesh, capacity, head_dim)
        return cls._by_key[key]

    def __init__(self, mesh, capacity: int, head_dim: int):
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        self.ccl_offset = ttnn.CoreCoord(grid.x - 1, 0)  # the CCL workers take the last column ...
        self.sdpa_grid = ttnn.CoreCoord(grid.x - 1, grid.y)  # ... the SDPA the others
        self.gathered = [
            ttnn.from_torch(
                torch.zeros(1, 1, capacity, head_dim),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for _ in range(2)
        ]


class TtGatedAttention:
    """One full-attention layer. ``w``: the checkpoint's ``self_attn.*`` tensors (fp32 host), ``cfg``: the reference
    Config, ``chunk``: the served chunk (5120), ``capacity``: tokens per cache slot (a multiple of the chunk)."""

    sp_axis, tp_axis = 0, 1

    def __init__(self, mesh, w: dict, cfg, chunk: int, capacity: int):
        self.mesh = mesh
        rows, cols = tuple(mesh.shape)
        self.sp, self.tp = rows, cols
        assert cfg.n_heads % cols == 0 and cfg.n_kv == cols, "one kv head per TP column"
        assert chunk % (rows * TILE) == 0 and capacity % chunk == 0
        self.nq, self.nkv, self.hd = cfg.n_heads // cols, cfg.n_kv // cols, cfg.head_dim
        self.rot = cfg.rotary_dim
        self.chunk, self.capacity, self.local_rows = chunk, capacity, chunk // rows
        self.eps = cfg.eps
        self.cfg = cfg
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.mm_cfg = compute_config(settings.get("ATTN_FIDELITY"), settings.get("ATTN_FP32_ACC"))
        self.sdpa_cfg = compute_config(settings.get("ATTN_SDPA_FIDELITY"), settings.get("ATTN_SDPA_FP32_ACC"))
        rep = ttnn.ReplicateTensorToMesh(mesh)
        col_mapper = lambda dim: ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, dim))  # noqa: E731

        def up(t, dtype, mapper):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=dram,
                mesh_mapper=mapper,
            )

        nq, hd = self.nq, self.hd
        wq = w["self_attn.q_proj.weight"].float().reshape(cfg.n_heads, 2, hd, -1)  # head h: [q | gate] rows
        wk = w["self_attn.k_proj.weight"].float().reshape(cfg.n_kv, hd, -1)
        wv = w["self_attn.v_proj.weight"].float().reshape(cfg.n_kv, hd, -1)
        H = wq.shape[-1]
        per_col = []
        for c in range(cols):
            heads = wq[c * nq : (c + 1) * nq]
            per_col += [heads[:, 0].reshape(-1, H), wk[c], wv[c], heads[:, 1].reshape(-1, H)]
        # [K, N] (bfp8 blocks along the outputs), as the GDN in-projections: the layer input carries a few huge
        # channels whose weights keep their bits this way
        self.w_qkvg = up(torch.cat(per_col).T, ttnn.bfloat8_b, col_mapper(-1))  # [H, cols * 3584]
        self.qkv_width = (nq + 2 * self.nkv) * hd
        # o_proj [H, n_heads * hd] stays [N, K] and is read with transpose_b (blocks along K), as out_proj and down
        self.w_o = up(w["self_attn.o_proj.weight"].float(), ttnn.bfloat8_b, col_mapper(-1))
        self.q_norm = up((1.0 + w["self_attn.q_norm.weight"].float()).reshape(1, 1, 1, hd), ttnn.float32, rep)
        self.k_norm = up((1.0 + w["self_attn.k_norm.weight"].float()).reshape(1, 1, 1, hd), ttnn.float32, rep)

        # RoPE tables in each row's cache order: row r, local row lr -> position blockcyclic_positions(...)
        pos = blockcyclic_positions(rows, chunk, capacity).float()
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.rot, 2, dtype=torch.int64).float() / self.rot))
        freqs = pos[:, None] * inv[None]
        emb = torch.cat([freqs, freqs], -1).reshape(1, 1, -1, self.rot)
        row_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(2, None))
        self.cos = up(emb.cos(), ttnn.bfloat16, row_mapper)  # [1, 1, capacity / sp, rot] per chip
        self.sin = up(emb.sin(), ttnn.bfloat16, row_mapper)

        grid = mesh.compute_with_storage_grid_size()
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # the ring's CCL takes the last column
            q_chunk_size=settings.get("ATTN_SDPA_Q_CHUNK"),
            k_chunk_size=settings.get("ATTN_SDPA_K_CHUNK"),
            exp_approx_mode=False,
        )
        self.links = settings.get("ATTN_RING_LINKS")

    # ---- cache (host <-> device: load time and the harness boundary only)
    def new_cache(self, num_layers: int = 1, slots: int = 1, capacity: int | None = None) -> KvCache:
        return self.cache_from_host(None, None, 0, num_layers, slots, capacity=capacity)

    def cache_from_host(
        self, key, value, length: int, num_layers: int = 1, slots: int = 1, layer: int = 0, capacity: int | None = None
    ) -> KvCache:
        """key, value [n_kv, >= length, hd] in natural order (the reference's layout), positions [0, length) valid; the
        rest of the cache is zero. Written into slot 0, layer ``layer``. ``capacity``: tokens per slot (a multiple of
        the chunk, at most the module's; default the module's)."""
        rows, cols = self.sp, self.tp
        capacity = capacity or self.capacity
        assert capacity % self.chunk == 0 and capacity <= self.capacity, capacity
        local = capacity // rows
        idx = blockcyclic_positions(rows, self.chunk, capacity)
        mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(rows, cols), dims=(2, 1))
        mem = kv_memory_config(self.mesh, self.hd)

        def one(t):
            full = torch.zeros(slots * num_layers, self.cfg.n_kv, capacity, self.hd)
            if t is not None and length:
                full[layer, :, :length] = t[:, :length].float()
            dev = full[:, :, idx].reshape(slots * num_layers, cols, rows * local, self.hd)
            return ttnn.from_torch(
                dev,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        _Ring.shared(self.mesh, capacity, self.hd)  # the ring's scratch for this capacity, at load time
        return KvCache(k=one(key), v=one(value), num_layers=num_layers, capacity=capacity)

    # ---- forward
    def __call__(self, x, cache: KvCache, start: int, slot: int = 0, layer: int = 0, end: int | None = None):
        """x: this chunk's rows [1, 1, 640, H]; ``start``: the chunk's first absolute position (a multiple of the
        chunk); K/V go to (slot, layer) of ``cache``. ``end``: the chunk's actual_end (rows at or past it are pad):
        only K/V rows [start, end) are written and the pad rows of the last 32-token record are zeroed."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        T, nq, nkv, hd, rot = x.shape[-2], self.nq, self.nkv, self.hd, self.rot
        assert T == self.local_rows and start % self.chunk == 0, f"{T} rows at {start}"
        assert start + self.chunk <= cache.capacity <= self.capacity
        end = start + self.chunk if end is None else end
        assert start < end <= start + self.chunk, f"actual_end {end} outside the chunk at {start}"
        ring = _Ring.shared(self.mesh, cache.capacity, hd)

        qkvg = ttnn.linear(x, self.w_qkvg, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg, memory_config=dram)
        qkv = ttnn.slice(qkvg, (0, 0, 0, 0), (1, 1, T, self.qkv_width), memory_config=dram)
        gate = ttnn.slice(qkvg, (0, 0, 0, self.qkv_width), (1, 1, T, self.qkv_width + nq * hd), memory_config=dram)
        ttnn.deallocate(qkvg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nq, num_kv_heads=nkv, transpose_k_heads=False, memory_config=dram
        )  # [1, nq, T, hd], [1, nkv, T, hd] x2
        ttnn.deallocate(qkv)

        lr = start // self.sp
        cos = ttnn.slice(self.cos, (0, 0, lr, 0), (1, 1, lr + T, rot), memory_config=dram)
        sin = ttnn.slice(self.sin, (0, 0, lr, 0), (1, 1, lr + T, rot), memory_config=dram)
        q = self._norm_rope(q, self.q_norm, cos, sin)
        k = self._norm_rope(k, self.k_norm, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        partial = end < start + self.chunk
        for c, t in ((cache.k, k), (cache.v, v)):
            t8 = ttnn.typecast(t, ttnn.bfloat8_b, memory_config=dram)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                c,
                t8,
                slot_idx=slot,
                layer_idx=layer,
                num_layers=cache.num_layers,
                kv_actual_global=start,
                cluster_axis=self.sp_axis,
                **({"valid_global": end} if partial else {}),
            )
            if end % KV_RECORD:
                ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                    c,
                    slot_idx=slot,
                    layer_idx=layer,
                    num_layers=cache.num_layers,
                    valid_global=end,
                    chunk_size_global=self.chunk,
                    cluster_axis=self.sp_axis,
                    pad_align=KV_RECORD,
                )
            ttnn.deallocate(t8)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        o, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            cache.k,
            cache.v,
            None,
            None,
            None,
            persistent_output_buffer_k=ring.gathered[0],
            persistent_output_buffer_v=ring.gathered[1],
            joint_strategy="rear",
            logical_n=start + self.chunk,
            program_config=self.sdpa_pc,
            scale=hd**-0.5,
            compute_kernel_config=self.sdpa_cfg,
            dim=2,
            multi_device_global_semaphore=ring.semaphores,
            num_links=self.links,
            cluster_axis=self.sp_axis,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=ring.ccl_offset,
            use_column_major_ccl=True,
            is_causal=True,
            is_balanced=False,
            kv_cache_batch_idx=slot,
            kv_cache_num_layers=cache.num_layers,
            kv_cache_layer_idx=layer,
        )  # [1, nq, T, hd]
        ttnn.deallocate(q)

        o = ttnn.experimental.nlp_concat_heads(o, memory_config=dram)  # [1, 1, T, nq * hd]
        gated = ttnn.multiply(o, ttnn.sigmoid(gate, memory_config=dram), dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(o)
        ttnn.deallocate(gate)
        partial = ttnn.linear(
            gated, self.w_o, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.mm_cfg, memory_config=dram
        )
        ttnn.deallocate(gated)
        out = ttnn.all_reduce(partial, cluster_axis=self.tp_axis, memory_config=dram)
        ttnn.deallocate(partial)
        res = ttnn.typecast(out, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(out)
        return res

    def _norm_rope(self, t, gamma, cos, sin):
        """Per-head rms_norm with the (1 + w) gamma, then rotate-half RoPE on the first ``rot`` dims."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        heads, T, hd, rot = t.shape[1], t.shape[2], self.hd, self.rot
        n = ttnn.rms_norm(t, epsilon=self.eps, weight=gamma, memory_config=dram, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(t)
        head = ttnn.slice(n, (0, 0, 0, 0), (1, heads, T, rot), memory_config=dram)
        tail = ttnn.slice(n, (0, 0, 0, rot), (1, heads, T, hd), memory_config=dram)
        ttnn.deallocate(n)
        roped = ttnn.experimental.rotary_embedding_hf(
            head, cos, sin, is_decode_mode=False, memory_config=dram, compute_kernel_config=self.mm_cfg
        )
        ttnn.deallocate(head)
        out = ttnn.concat([roped, tail], dim=-1, memory_config=dram)
        ttnn.deallocate(roped)
        ttnn.deallocate(tail)
        return out
