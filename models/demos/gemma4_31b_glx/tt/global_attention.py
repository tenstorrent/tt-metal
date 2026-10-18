# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 global (full) attention on the SP8 x TP4 Galaxy, one layer, over the served packed ring KV cache.

Adapted from models/demos/gemma4_d_p/tt/attention (__init__.py, global_kv_cache.py, ring_prefill.py, weights.py).

Gemma-4 global layers: 32 Q heads and 4 KV heads x 512, no v_proj (attention_k_eq_v): one tied projection k_raw,
V = rms_norm(k_raw) without weight, K = RoPE(k_norm(k_raw)). RoPE is "proportional", theta 1e6: only the first 64
of the 256 NeoX frequencies are non-zero, so 128 lanes (NeoX 0..63 and 256..319) rotate and the other 384 are the
identity. Scale 1.0, causal over the whole prefix.

The served cache row (serving_contract.md) is 640 wide, [K rotary 128 (adjacent pairs) | V non-rotary 384 | V rotary
128] (gemma4_d_p global_kv_cache.global_kv_indices). Its first 512 columns are the K view: the K non-rotary lanes are
V's times k_norm's gamma, so that gamma moves to Q (folded into the q_norm weight) and the score is unchanged. Its last
512 columns are the V view, in "value order"; o_proj's rows are permuted to that order at load.

Per chunk (C = 5120 tokens, L = C / SP = 640 rows per SP row, 160 per chip between blocks), TP column t holds Q heads
8t..8t+7 and KV head t:
    all_gather rows over TP -> [1, 1, 640, 5376]
    tied QK linear (column-parallel, [q 8 heads (packed-Q order) | k 1 head (value order)] x 512)
    nlp_create_qkv_heads(kv_tied) -> q [1, 8, 640, 512], k_raw [1, 1, 640, 512]
    q: fp32 RMS norm with the folded weight, fp32 RoPE on its first 128 lanes (cos 1 / sin 0 beyond) -> bf16
    V = fp32 RMS norm of k_raw (no weight); K rotary = RoPE(gamma * (V's last 128 lanes, NeoX halves -> adjacent pairs))
    packed = [K rotary | V] fp32 -> bfp8 -> update_padded_kv_cache into the slot's rows
    ring_mla over the packed cache (CP ring along mesh axis 0, head_dim_v 512, causal, kv_actual_isl, scale 1.0)
    nlp_concat_heads, o_proj (row-parallel), reduce_scatter rows over TP -> [1, 1, 160, 5376]
"""

import torch

import ttnn

from .attention import (
    SP_AXIS,
    TILE,
    TP_AXIS,
    _allocate_ring_cache,
    _compute_config,
    matmul_grid,
    rot_transformation_mat,
)
from .settings import settings

ROTARY_DIM = 128  # rotating lanes of a 512-wide global head (partial_rotary_factor 0.25 of 512)


def global_kv_indices(head_dim: int, rotary_dim: int = ROTARY_DIM):
    """(rotary, nonrotary, value) NeoX source lanes of the packed row (gemma4_d_p global_kv_cache):
    rotary = [0, h, 1, h + 1, ...] for the first rotary_dim / 2 lanes of each half (h = head_dim / 2), adjacent pairs;
    nonrotary = the rest in HF order; value = [nonrotary | rotary first halves | rotary second halves]."""
    half, rh = head_dim // 2, rotary_dim // 2
    first, second = torch.arange(rh), torch.arange(half, half + rh)
    rotary = torch.stack((first, second), dim=1).reshape(-1)
    nonrotary = torch.cat((torch.arange(rh, half), torch.arange(half + rh, head_dim)))
    return rotary, nonrotary, torch.cat((nonrotary, first, second))


def pack_global_kv(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Host packing of the reference's K (after k_norm and RoPE, NeoX lanes) and V into the served 640-wide row."""
    rotary, _, value_order = global_kv_indices(key.shape[-1])
    return torch.cat((key[..., rotary], value[..., value_order]), dim=-1)


class GlobalAttention:
    def __init__(self, mesh, w: dict, cfg: dict, max_seq: int, chunk: int, chunk_tables: bool = True):
        """w: checkpoint tensors q_proj, k_proj, o_proj, q_norm, k_norm (HF names without the prefix).
        cfg: the text config. max_seq / chunk: the served geometry (cache capacity and chunk size).
        chunk_tables: build this layer's own per-chunk RoPE tables, offsets and causal mask (chunk-aligned starts, the
        component test). The device model passes False and hands every call a ChunkMeta (tt/model.py)."""
        self.mesh = mesh
        sp, tp = tuple(mesh.shape)
        self.sp, self.tp = sp, tp
        self.d = d = cfg["global_head_dim"]
        self.n_q, self.n_kv = cfg["num_attention_heads"], cfg["num_global_key_value_heads"]
        assert cfg["attention_k_eq_v"] and self.n_kv == tp, (cfg["attention_k_eq_v"], self.n_kv, tp)
        self.n_q_local = self.n_q // tp
        self.eps = cfg["rms_norm_eps"]
        rope = cfg["rope_parameters"]["full_attention"]
        assert rope["rope_type"] == "proportional", rope
        assert int(rope["partial_rotary_factor"] * d) == ROTARY_DIM, rope
        self.chunk, self.max_seq = chunk, max_seq
        assert max_seq % chunk == 0 and chunk % (sp * tp * TILE) == 0, (max_seq, chunk)
        self.slab = chunk // sp

        replicate = ttnn.ReplicateTensorToMesh(mesh)
        rotary, nonrotary, value_order = global_kv_indices(d)
        query_order = torch.cat((rotary, nonrotary))

        # Tied QK, column-parallel: TP column t holds [q heads 8t..8t+7 (packed-Q order) | k head t (value order)].
        hidden = w["q_proj"].shape[-1]
        q = w["q_proj"].reshape(self.n_q, d, hidden)[:, query_order]
        k = w["k_proj"].reshape(self.n_kv, d, hidden)[:, value_order]
        cols = []
        for t in range(tp):
            cols += [q[t * self.n_q_local : (t + 1) * self.n_q_local].reshape(-1, hidden), k[t]]
        # out_in: [tp * (8 + 1) * d, hidden] (bfp8 blocks along the input, matmul transpose_b); in_out: its transpose.
        self.w_transposed = settings.get("ATTN_GLOBAL_W_LAYOUT") == "out_in"
        wqk = torch.cat(cols, dim=0)
        if not self.w_transposed:
            wqk = wqk.transpose(0, 1)
        self.wqk = ttnn.from_torch(
            wqk.contiguous(),
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(None, -2 if self.w_transposed else -1)),
        )
        # o_proj, input lanes of each head in value order (the SDPA output's lane order), row-parallel. bfp8 shares one
        # exponent per 16 values of a stored row: "out_in" stores the checkpoint's [hidden, n_q * d] (blocks along the
        # input, matmul with transpose_b); "in_out" stores o_proj^T, whose blocks run along the output channels, where
        # a large channel flushes its 15 neighbours (layer 5 column 2294: rel 0.12 against a limit of 0.038).
        wo = w["o_proj"].reshape(hidden, self.n_q, d)[:, :, value_order].reshape(hidden, self.n_q * d)
        if not self.w_transposed:
            wo = wo.transpose(0, 1)
        self.wo = ttnn.from_torch(
            wo.contiguous(),
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(None, -1 if self.w_transposed else -2)),
        )

        def fp32_const(x):
            return ttnn.from_torch(
                x.float(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        # q_norm weight in packed-Q order, its non-rotary lanes times k_norm's gamma (moved from K to Q).
        k_gamma = w["k_norm"].float()
        self.k_gamma = k_gamma  # host copy: state.to_torch rebuilds K's non-rotary lanes as V's times gamma
        q_scale = torch.cat((torch.ones(ROTARY_DIM), k_gamma[nonrotary]))
        self.q_norm_w = fp32_const((w["q_norm"].float()[query_order] * q_scale).reshape(1, 1, 1, d))
        # K rotary gamma, adjacent-pair order (applied after the NeoX-halves -> adjacent permutation).
        self.k_rot_gamma = fp32_const(k_gamma[rotary].reshape(1, 1, 1, ROTARY_DIM))
        # V's last 128 lanes are [NeoX 0..63 | NeoX 256..319]; this permutation puts them in adjacent pairs.
        rh = ROTARY_DIM // 2
        adj = torch.stack((torch.arange(rh), torch.arange(rh, ROTARY_DIM)), dim=1).reshape(-1)
        perm = torch.zeros(1, 1, ROTARY_DIM, ROTARY_DIM)
        perm[..., adj, torch.arange(ROTARY_DIM)] = 1
        self.k_rot_perm = fp32_const(perm)
        # Adjacent-pair rotation x -> (-x1, x0, ...): 512-wide for q (zero beyond lane 128), 128-wide for K rotary.
        rot_q = torch.zeros(1, 1, d, d)
        rot_q[..., :ROTARY_DIM, :ROTARY_DIM] = rot_transformation_mat(ROTARY_DIM)
        self.rot_q, self.rot_k = fp32_const(rot_q), fp32_const(rot_transformation_mat(ROTARY_DIM))

        # Per chunk index: fp32 RoPE tables [1, 1, L, d] on SP row r for positions k*C + r*L + [0, L), lanes in packed-Q
        # order (cos 1 / sin 0 beyond the 128 rotary lanes), and the cache-write offset k*C (replicated uint32).
        inv_freq = 1.0 / (rope["rope_theta"] ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.int64).float() / d))
        inv_freq = torch.cat((inv_freq, torch.zeros(d // 2 - rh)))
        self.inv_freq = inv_freq  # float32 [d / 2]; tt/model.py builds the any-position tables from it
        n_chunks = max_seq // chunk if chunk_tables else 0

        def table(c, fn):
            pos = torch.arange(c * chunk, (c + 1) * chunk, dtype=torch.float64)
            emb = (pos[:, None] * inv_freq[None, :].double()).repeat(1, 2)[:, query_order]
            return ttnn.from_torch(
                fn(emb).float().reshape(sp, 1, self.slab, d),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(0, None)),
            )

        self.cos = [table(c, torch.cos) for c in range(n_chunks)]
        self.sin = [table(c, torch.sin) for c in range(n_chunks)]

        def scalar(value):
            return ttnn.from_torch(
                torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        self.kv_actual = [scalar(c * chunk) for c in range(n_chunks)]
        self.slot = scalar(0)

        # The served packed cache, one layer: [slots, 1 head, max_seq / SP, d + 128] bfp8.
        self.packed = d + ROTARY_DIM
        self.slots = settings.get("KV_SLOTS")
        self.cache = _allocate_ring_cache(mesh, [self.slots, 1, max_seq // sp, self.packed], self.packed)
        # ring_mla gathers every chip's whole shard: the receive buffer spans the full capacity, DRAM interleaved
        # (the fused gather's bank-owned schedule needs an interleaved output).
        # Only the ring_mla mode reads it (38 MB per layer per chip).
        self.sdpa_mode = settings.get("ATTN_GLOBAL_SDPA")
        if self.sdpa_mode == "ring_mla":
            self.gather = ttnn.zeros(
                [1, 1, max_seq, self.packed],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Ring SDPA: CCL workers on the last compute column, SDPA on the rest.
        grid = mesh.compute_with_storage_grid_size()
        all_cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
        self.ring_sems = [ttnn.create_global_semaphore(mesh, all_cores, 0) for _ in range(3)]
        self.ccl_offset = ttnn.CoreCoord(grid.x - 1, 0)
        q_chunk, k_chunk = settings.get("ATTN_GLOBAL_Q_CHUNK"), settings.get("ATTN_GLOBAL_K_CHUNK")
        sdpa_cores = (grid.x - 1) * grid.y
        # Segmented accumulation needs one Q chunk per core (gemma4_d_p ring_sdpa_chunk_sizes).
        segmented = self.n_q_local * -(-self.slab // q_chunk) <= sdpa_cores
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
            max_k_splits=1,
            segmented_accumulation=segmented and settings.get("ATTN_GLOBAL_SEGMENTED"),
        )
        self.num_links = settings.get("ATTN_RING_NUM_LINKS")
        self.partial_dtype = ttnn.float32 if settings.get("ATTN_GLOBAL_FP32_REDUCE") else ttnn.bfloat16
        self.qk_dtype = ttnn.float32 if settings.get("ATTN_GLOBAL_FP32_QK") else ttnn.bfloat16

        # Composed SDPA: the current chunk's keys arrive SP-gathered (key row r' * L + i = position start + r' * L + i);
        # SP row r's query j (position start + r * L + j) sees key row i' iff i' <= r * L + j. Prefix keys: all visible.
        # (The device model's ChunkMeta carries this mask for any start, and a key mask for a partial prefix slab.)
        if chunk_tables:
            qpos = (torch.arange(sp)[:, None] * self.slab + torch.arange(self.slab)[None, :])[:, :, None]
            mask = torch.where(torch.arange(chunk)[None, None, :] <= qpos, 0.0, -1e9).reshape(sp, 1, self.slab, chunk)
            self.chunk_mask = ttnn.from_torch(
                mask,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(0, None)),
            )

        self.proj_cfg = _compute_config(mesh, settings.get("ATTN_PROJ_FIDELITY"), settings.get("ATTN_PROJ_FP32_ACC"))
        self.grid = matmul_grid()
        self.sdpa_cfg = _compute_config(mesh, settings.get("ATTN_SDPA_FIDELITY"), settings.get("ATTN_SDPA_FP32_ACC"))
        self.norm_cfg = _compute_config(mesh, "HiFi4", True)

    def chunk_index(self, start: int) -> int:
        assert start % self.chunk == 0 and 0 <= start < self.max_seq, start
        return start // self.chunk

    def write_kv(self, packed, start: int, meta=None):
        """Append one chunk's packed rows ([1, 1, L, 640] per chip, SP row r holding its slab) to slot 0. fp32 input:
        the device's bf16 -> bfp8 typecast rounds row norms up on average, fp32 -> bfp8 does not (except on bf16-exact
        fp32 values, e.g. a loaded bf16 state: +0.28 %, so pass those as host-rounded bfp8). meta (ChunkMeta): its
        on-device slot, start and actual_end; rows from ceil32(actual_end) on are not written."""
        xb = ttnn.typecast(packed, ttnn.bfloat8_b) if packed.dtype != ttnn.bfloat8_b else packed
        if meta is None:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache=self.cache,
                input=xb,
                slot_idx=self.slot,
                kv_actual_global=self.kv_actual[self.chunk_index(start)],
                layer_idx=0,
                num_layers=1,
                cluster_axis=SP_AXIS,
            )
        else:
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache=self.cache,
                input=xb,
                slot_idx=meta.slot,
                kv_actual_global=meta.kv_actual,
                layer_idx=0,
                num_layers=1,
                cluster_axis=SP_AXIS,
                valid_global=meta.valid,
            )
        if xb is not packed:
            ttnn.deallocate(xb)

    def load_rows(self, packed, start: int, end: int):
        """Harness / state loading (not the forward): chunk-aligned start, rows [start, ceil32(end)) of slot 0 from
        packed laid out as write_kv's (host scalars)."""
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache=self.cache,
            input=packed,
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            kv_actual_global=start,
            cluster_axis=SP_AXIS,
            valid_global=end,
        )

    def _norm(self, x, weight):
        """fp32 per-head RMS norm over d: x * rsqrt(mean(x^2) + eps) (* weight), x [1, h, L, d] -> fp32."""
        x32 = ttnn.typecast(x, ttnn.float32)
        sq = ttnn.multiply(x32, x32, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ms = ttnn.mean(
            sq, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.norm_cfg
        )
        ttnn.deallocate(sq)
        r = ttnn.rsqrt(ttnn.add(ms, self.eps, dtype=ttnn.float32), fast_and_approximate_mode=False)
        ttnn.deallocate(ms)
        y = ttnn.multiply(x32, r, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in (x32, r):
            ttnn.deallocate(t)
        if weight is not None:
            yw = ttnn.multiply(y, weight, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(y)
            y = yw
        return y

    def _matmul(self, x, m):
        return ttnn.matmul(
            x,
            m,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_cfg,
            core_grid=self.grid,
        )

    def _rope(self, x, rot, cos, sin, dtype):
        """x * cos + (x @ rot) * sin in fp32 (rot the exact +-1 adjacent-pair rotation) -> dtype."""
        r = self._matmul(x, rot)
        xc = ttnn.multiply(x, cos, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rs = ttnn.multiply(r, sin, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(r)
        y = ttnn.add(xc, rs, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xc)
        ttnn.deallocate(rs)
        return y

    def __call__(self, x, start: int, meta=None):
        """x: the residual chunk [1, 1, L / TP, hidden] bf16 per chip. Returns the same layout.
        meta: the device model's ChunkMeta (any tile-aligned start, pad rows); None: chunk-aligned start, own tables."""
        if meta is None:
            c = self.chunk_index(start)
            cos, sin = self.cos[c], self.sin[c]
        else:
            assert self.sdpa_mode == "composed", "the device model runs the composed global attention"
            cos, sin = meta.global_rope
        xg = ttnn.all_gather(x, dim=2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qk = ttnn.linear(
            xg,
            self.wqk,
            transpose_b=self.w_transposed,
            dtype=self.qk_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.proj_cfg,
            core_grid=self.grid,
        )
        ttnn.deallocate(xg)
        q, k_raw, v_raw = ttnn.experimental.nlp_create_qkv_heads(
            qk,
            num_heads=self.n_q_local,
            num_kv_heads=1,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            kv_tied=True,
        )
        ttnn.deallocate(qk)
        ttnn.deallocate(k_raw)

        # Q: norm (folded weight), RoPE on the 128 rotary lanes, bf16 for the SDPA.
        q_n = self._norm(q, self.q_norm_w)
        ttnn.deallocate(q)
        q_r = self._rope(q_n, self.rot_q, cos, sin, ttnn.float32 if self.sdpa_mode == "composed" else ttnn.bfloat16)
        ttnn.deallocate(q_n)

        # V = norm(k_raw) (value order); K rotary from V's last 128 lanes: adjacent pairs, gamma, RoPE.
        v = self._norm(v_raw, None)
        ttnn.deallocate(v_raw)
        _, h, s, d = v.shape
        v_rot = ttnn.slice(v, (0, 0, 0, d - ROTARY_DIM), (1, h, s, d), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_adj = self._matmul(v_rot, self.k_rot_perm)
        ttnn.deallocate(v_rot)
        k_g = ttnn.multiply(k_adj, self.k_rot_gamma, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_adj)
        cos_k = ttnn.slice(cos, (0, 0, 0, 0), (1, 1, s, ROTARY_DIM), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin_k = ttnn.slice(sin, (0, 0, 0, 0), (1, 1, s, ROTARY_DIM), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_rot = self._rope(k_g, self.rot_k, cos_k, sin_k, ttnn.float32)
        for t in (k_g, cos_k, sin_k):
            ttnn.deallocate(t)
        packed = ttnn.concat((k_rot, v), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_rot)
        ttnn.deallocate(v)
        if meta is not None and meta.keep is not None:
            # Pad rows (positions >= actual_end) reach the cache as exact zeros: the last 32-token record ships whole.
            kept = ttnn.multiply(packed, meta.keep, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(packed)
            packed = kept
        if self.sdpa_mode == "composed":
            # The cache keeps bfp8; the chunk attends its own fresh K / V in bf16 (as the reference, whose state only
            # rounds what later chunks read).
            self.write_kv(packed, start, meta)
            fresh = ttnn.typecast(packed, ttnn.bfloat16)
            ttnn.deallocate(packed)
            if meta is None:
                attn = self._composed_sdpa(q_r, fresh, c * self.slab, self.chunk_mask, None)
            else:
                attn = self._composed_sdpa(q_r, fresh, meta.prefix_rows, meta.chunk_mask, meta.prefix_mask)
            ttnn.deallocate(fresh)
        else:
            self.write_kv(packed, start)
            ttnn.deallocate(packed)
            attn = self._ring_mla(q_r, c)
        ttnn.deallocate(q_r)
        heads = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        partial = ttnn.linear(
            heads,
            self.wo,
            transpose_b=self.w_transposed,
            dtype=self.partial_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.proj_cfg,
            core_grid=self.grid,
        )
        ttnn.deallocate(heads)
        out = ttnn.reduce_scatter(partial, dim=2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        if out.dtype != ttnn.bfloat16:
            out32 = out
            out = ttnn.typecast(out32, ttnn.bfloat16)
            ttnn.deallocate(out32)
        return out

    def _gather_rows(self, rows):
        """[1, 1, n, 640] bfp8 or bf16 per chip -> its K / V views over every SP row's n rows, bf16 [1, 1, SP * n, 512]
        each (bfp8 values exactly; key order SP row major, which the softmax does not see)."""
        rb = ttnn.typecast(rows, ttnn.bfloat16) if rows.dtype != ttnn.bfloat16 else rows
        g = ttnn.all_gather(rb, dim=2, cluster_axis=SP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if rb is not rows:
            ttnn.deallocate(rb)
        _, _, n, w = g.shape
        k = ttnn.slice(g, (0, 0, 0, 0), (1, 1, n, self.d), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.slice(g, (0, 0, 0, w - self.d), (1, 1, n, w), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(g)
        return k, v

    def _composed_sdpa(self, q, chunk_rows, prefix_rows, chunk_mask, prefix_mask):
        """Causal attention of q [1, 8, L, 512] over slot 0's prefix and this chunk (prefix: the bfp8 cache's local rows
        [0, prefix_rows) on every SP row, prefix_mask [1, 1, 1, SP * prefix_rows] masking keys at or past the start, None:
        all visible; chunk: chunk_rows, this chunk's packed rows in bf16, chunk_mask the causal mask of q against them):
        QK^T in fp32, mask, fp32 softmax, P V; scale 1.0. -> [1, 8, L, 512]."""
        _, h, s, d = q.shape
        q2 = ttnn.reshape(q, (1, 1, h * s, d))
        k_cur, v_cur = self._gather_rows(chunk_rows)
        scores = ttnn.matmul(
            q2,
            k_cur,
            transpose_b=True,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_cfg,
            core_grid=self.grid,
        )
        ttnn.deallocate(k_cur)
        scores4 = ttnn.reshape(scores, (1, h, s, self.chunk))
        masked = ttnn.add(scores4, chunk_mask, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scores)
        masked = ttnn.reshape(masked, (1, 1, h * s, self.chunk))
        vs = [v_cur]
        if prefix_rows:
            prefix = ttnn.slice(
                self.cache, (0, 0, 0, 0), (1, 1, prefix_rows, self.packed), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            k_pre, v_pre = self._gather_rows(prefix)
            ttnn.deallocate(prefix)
            pre = ttnn.matmul(
                q2,
                k_pre,
                transpose_b=True,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.norm_cfg,
                core_grid=self.grid,
            )
            ttnn.deallocate(k_pre)
            if prefix_mask is not None:
                pm = ttnn.add(pre, prefix_mask, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(pre)
                pre = pm
            allsc = ttnn.concat((pre, masked), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pre)
            ttnn.deallocate(masked)
            masked = allsc
            vs = [v_pre, v_cur]
        # Softmax as exp(x - max) V / sum, each an exact fp32 op (ttnn.softmax: rel 0.002 off on these scores).
        m = ttnn.max(masked, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xm = ttnn.subtract(masked, m, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(masked)
        ttnn.deallocate(m)
        e = ttnn.exp(xm, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xm)
        total = ttnn.sum(
            e, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.norm_cfg
        )
        v = ttnn.concat(vs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(vs) > 1 else vs[0]
        ev = ttnn.matmul(
            e,
            v,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_cfg,
            core_grid=self.grid,
        )
        ttnn.deallocate(e)
        for t in {id(x): x for x in vs + [v]}.values():
            ttnn.deallocate(t)
        inv = ttnn.reciprocal(total, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(total)
        out = ttnn.multiply(ev, inv, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ev)
        ttnn.deallocate(inv)
        return ttnn.reshape(out, (1, h, s, d))

    def _ring_mla(self, q_r, c):
        attn, _ = ttnn.transformer.ring_mla(
            q_r,
            self.cache,
            persistent_output_buffer_kv=self.gather,
            head_dim_v=self.d,
            logical_n=self.max_seq,
            program_config=self.sdpa_program_config,
            scale=1.0,
            compute_kernel_config=self.sdpa_cfg,
            dim=2,
            multi_device_global_semaphore=self.ring_sems,
            num_links=self.num_links,
            cluster_axis=SP_AXIS,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=self.ccl_offset,
            use_column_major_ccl=True,
            is_balanced=False,
            slot_id=self.slot,
            kv_actual_isl_tensor=self.kv_actual[c],
            kv_cache_num_layers=1,
            kv_cache_layer_idx=0,
        )
        return attn
