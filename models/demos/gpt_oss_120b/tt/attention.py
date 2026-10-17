# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS prefill attention on the [8, 4] mesh (plan.md, sliding_moe / attn_moe step 2), after
gpt_oss_d_p/tt/attention/ re-planned for SP 8 x TP 4.

Per chip (row r, column c): x [1, 1, 640, 2880] bf16, the chunk's tokens p with (p // 640) % 8 == r.
    fused QKV ttnn.linear + bias (weight ATTN_QKV_WEIGHT_DTYPE, bf16) -> [640, 1280] = 16 q heads (16c..16c+15) | 2 k heads (2c, 2c+1) | 2 v heads,
        fp32 out (QKV_FP32): the device's bf16 -> bfp8 typecast scales values ~0.3 % up, fp32 -> bfp8 does not
    nlp_create_qkv_heads -> q [1, 16, 640, 64], k / v [1, 2, 640, 64]
    RoPE on q and k: YaRN cos / sin in meta (interleaved-pair) order, one whole-cache block-cyclic table built at
        load, rows picked by rotary_embedding_indexed (ROPE_FP32: the rotation itself composed in fp32, TtRope); the
        q / k weight rows are permuted to meta order at load, so K lands in the cache in meta order (serving_answers
        k_rope_layout meta) and q.k is unchanged
    update_padded_kv_cache: post-RoPE K and V (bfp8) into the served cache at kv_actual = chunk start
    full layers (ATTN_CHUNK_KV_BF16): SDPA reads a per-call bf16 copy of the slot (slice + typecast, then the chunk's
        own K / V written in bf16), not the bfp8 cache; K block SDPA_FULL_K_CHUNK (512)
    sliding layers (ATTN_SLIDING_KV_SPLIT; the op takes only bfp8 K / V there): SDPA reads a per-call slot copy of
        head dim 128, [hi | lo] with hi = bfp8(x) and lo = bfp8(x - hi) (the prefix: [cache | 0]), q is [q | q], and
        the output halves are summed: q.k and P.v then see the chunk's K / V to ~2^-14 instead of bfp8's 2^-7
    ring_joint_scaled_dot_product_attention over SP (axis 0): GQA 16:2, attention_sink = the column's 16 sinks / scale,
        scale 2^-3 (exact in bf16, so the sink fold is exact), sliding_window_size 128 (None on full layers)
    nlp_concat_heads -> [640, 1024]; o_proj in-features 1024c..1024c+1023 (row-parallel, [out, in] bfp8 + transpose_b) -> fp32 partial [640, 2880]
    ttnn.all_reduce over TP (axis 1), then + o bias once, bf16 out [1, 1, 640, 2880] replicated over the columns.
"""

import math

import torch

import ttnn

from .ccl import SP_AXIS, TP_AXIS
from .kv_cache import block_cyclic_reorder
from .settings import settings

HEAD_DIM = 64


def meta_permute_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Rows of a q / k projection (weight [n*D, in] or bias [n*D]) from half-split to meta interleaved-pair order
    within each head (tt_transformers reverse_permute): new row 2j = old j, new row 2j + 1 = old D/2 + j."""
    shape = w.shape
    w2 = w.reshape(n_heads, 2, HEAD_DIM // 2, -1).transpose(1, 2)
    return w2.reshape(shape)


def meta_permute_cols(t: torch.Tensor) -> torch.Tensor:
    """The last dim (head_dim) of a post-RoPE K from half-split to meta order (as meta_permute_rows)."""
    shape = t.shape
    return t.reshape(*shape[:-1], 2, HEAD_DIM // 2).transpose(-1, -2).reshape(shape)


def _compute_config(mesh, fidelity: str, fp32_acc: bool):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


def rot_transformation_mat() -> torch.Tensor:
    """The one-tile RoPE rotation matrix of rotary_embedding_llama / rotary_embedding_indexed (meta order)."""
    m = torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE)
    m[..., torch.arange(0, ttnn.TILE_SIZE, 2), torch.arange(1, ttnn.TILE_SIZE, 2)] = 1
    m[..., torch.arange(1, ttnn.TILE_SIZE, 2), torch.arange(0, ttnn.TILE_SIZE, 2)] = -1
    return m


class TtRope:
    """Whole-cache YaRN cos / sin for rotary_embedding_indexed, built once: meta order [c0, c0, c1, c1, ...] (the
    attention factor folded in), block-cyclic over the 640-row chip slabs, SP-sharded over axis 0, replicated over
    TP. The op picks this chunk's rows on device from kv_actual_global and the chip's SP coordinate.

    ROPE_FP32: the op only fetches the rows, as 1 * table with a zero rotation (exact for bf16 tables), from hi / lo
    bf16 tables (lo = table - bf16(table)); cos / sin = hi + lo in fp32. Then x * cos + rot(x) * sin in fp32, rot(x) as
    matmuls of x's bf16 hi / lo halves with the 64 x 64 meta rotation (+-1 entries, exact with fp32 accumulation)."""

    def __init__(self, mesh, inv_freq: torch.Tensor, attention_factor: float, max_seq: int, chunk: int):
        sp = tuple(mesh.shape)[SP_AXIS]
        pos = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq.float())
        cos = (freqs.cos() * attention_factor).repeat_interleave(2, dim=-1)[None, None]
        sin = (freqs.sin() * attention_factor).repeat_interleave(2, dim=-1)[None, None]
        mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))

        def dev(t):
            return ttnn.from_torch(
                block_cyclic_reorder(t, chunk // sp, sp),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def rep(t, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                t,
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.fp32 = settings.get("ROPE_FP32")
        if self.fp32:
            bf = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
            self.tables = [dev(h) for t in (cos, sin) for h in (bf(t), t - bf(t))]  # cos hi, lo, sin hi, lo
            self.ones = rep(torch.ones(1, 1, chunk // sp, HEAD_DIM), ttnn.float32)
            self.zero_rot = rep(torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE))
            rot = torch.zeros(HEAD_DIM, HEAD_DIM)
            for b in range(0, HEAD_DIM, ttnn.TILE_SIZE):
                rot[b : b + ttnn.TILE_SIZE, b : b + ttnn.TILE_SIZE] = rot_transformation_mat()[0, 0]
            self.rot = rep(rot[None, None])
            self.mm_config = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        else:
            self.cos, self.sin = dev(cos), dev(sin)
        self.trans = rep(rot_transformation_mat())

    def tables_at(self, start: int):
        """This chunk's cos / sin rows [1, 1, S_local, 64] fp32 (ROPE_FP32)."""
        rows = [
            ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                self.ones, t, t, self.zero_rot, kv_actual_global=start, cluster_axis=SP_AXIS
            )
            for t in self.tables
        ]
        out = []
        for hi, lo in (rows[:2], rows[2:]):
            out.append(ttnn.add(hi, lo, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            hi.deallocate(True)
            lo.deallocate(True)
        return out

    def _rotate(self, t: ttnn.Tensor) -> ttnn.Tensor:
        hi = ttnn.typecast(t, ttnn.bfloat16)
        hi_f = ttnn.typecast(hi, ttnn.float32)
        diff = ttnn.subtract(t, hi_f)
        hi_f.deallocate(True)
        lo = ttnn.typecast(diff, ttnn.bfloat16)
        diff.deallocate(True)
        parts = [ttnn.matmul(h, self.rot, dtype=ttnn.float32, compute_kernel_config=self.mm_config) for h in (hi, lo)]
        hi.deallocate(True)
        lo.deallocate(True)
        out = ttnn.add(parts[0], parts[1], dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for p in parts:
            p.deallocate(True)
        return out

    def __call__(self, t: ttnn.Tensor, start: int, tables=None) -> ttnn.Tensor:
        if not self.fp32:
            return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                t, self.cos, self.sin, self.trans, kv_actual_global=start, cluster_axis=SP_AXIS
            )
        cos, sin = tables
        if t.dtype != ttnn.float32:
            t = ttnn.typecast(t, ttnn.float32)
        rot = self._rotate(t)
        a = ttnn.multiply(t, cos, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.multiply(rot, sin, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rot.deallocate(True)
        out = ttnn.add(a, b, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a.deallocate(True)
        b.deallocate(True)
        return out


class TtAttention:
    """One layer's attention. ``w``: the checkpoint tensors (q/k/v/o weight and bias, sinks) in HF layout;
    ``window``: 128 on sliding layers, None on full ones."""

    def __init__(self, mesh, ccl, w: dict, cfg: dict, rope: TtRope, window):
        self.mesh, self.ccl, self.rope, self.window = mesh, ccl, rope, window
        rows, cols = tuple(mesh.shape)
        self.sp, self.tp = rows, cols
        self.nq, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        assert cfg["head_dim"] == HEAD_DIM
        self.nq_local, self.nkv_local = self.nq // cols, self.nkv // cols
        self.scale = HEAD_DIM**-0.5
        assert math.log2(self.scale).is_integer(), "the sink fold is exact only for a power-of-two scale"

        wq = meta_permute_rows(w["wq"].float(), self.nq)
        bq = meta_permute_rows(w["bq"].float(), self.nq)
        wk = meta_permute_rows(w["wk"].float(), self.nkv)
        bk = meta_permute_rows(w["bk"].float(), self.nkv)
        wv, bv = w["wv"].float(), w["bv"].float()
        q_rows, kv_rows = self.nq_local * HEAD_DIM, self.nkv_local * HEAD_DIM
        wcols, bcols = [], []
        for c in range(cols):  # column c: its q heads | its k heads | its v heads
            sl_q, sl_kv = slice(c * q_rows, (c + 1) * q_rows), slice(c * kv_rows, (c + 1) * kv_rows)
            wcols.append(torch.cat([wq[sl_q], wk[sl_kv], wv[sl_kv]], dim=0).T)
            bcols.append(torch.cat([bq[sl_q], bk[sl_kv], bv[sl_kv]], dim=0))
        col_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, 3))
        self.wqkv = ttnn.from_torch(
            torch.cat(wcols, dim=-1)[None, None],
            device=mesh,
            dtype=getattr(ttnn, settings.get("ATTN_QKV_WEIGHT_DTYPE")),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_mapper,
        )
        self.bqkv = ttnn.from_torch(
            torch.cat(bcols, dim=-1).reshape(1, 1, 1, -1),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_mapper,
        )
        # o_proj row-parallel: column c holds the in-features of its q heads (columns 1024c..1024c+1023 of wo). Kept
        # [out, in] and applied with transpose_b, so the bfp8 blocks of 16 run along the in-features, as the checkpoint
        # rows are; blocks along the out-features share one exponent across 16 output columns and lose small columns
        self.wo = ttnn.from_torch(
            w["wo"].float()[None, None],
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, 3)),
        )
        self.reduce_fp32 = settings.get("O_PROJ_REDUCE_FP32")
        self.bo = ttnn.from_torch(
            w["bo"].float().reshape(1, 1, 1, -1),
            device=mesh,
            dtype=ttnn.float32 if self.reduce_fp32 else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # the kernel scales the sink logit too; HF does not: store sink / scale (exact: scale is 2^-3)
        self.sinks = ttnn.from_torch(
            (w["sinks"].float() / self.scale).reshape(1, self.nq, 1, 1),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, 1)),
        )
        self.qkv_dtype = ttnn.float32 if settings.get("QKV_FP32") else ttnn.bfloat16
        self.chunk_kv_bf16 = settings.get("ATTN_CHUNK_KV_BF16")
        self.kv_split = window is not None and settings.get("ATTN_SLIDING_KV_SPLIT")
        self.mm_config = _compute_config(mesh, settings.get("ATTN_MM_FIDELITY"), settings.get("ATTN_MM_FP32_ACC"))
        self.sdpa_config = _compute_config(mesh, settings.get("SDPA_FIDELITY"), settings.get("SDPA_FP32_ACC"))
        self.sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ccl.ring_sdpa_grid,
            q_chunk_size=settings.get("SDPA_Q_CHUNK"),
            k_chunk_size=settings.get("SDPA_K_CHUNK" if window else "SDPA_FULL_K_CHUNK"),
            exp_approx_mode=False,
        )

    def _gather_buffers(self, cache_rows_local: int, dtype=ttnn.bfloat8_b, head_dim: int = HEAD_DIM):
        if self.window is None:  # full layers gather every chip's whole shard
            seq = cache_rows_local * self.sp
        else:  # sliding: two compact predecessor-halo slots (one, or two when Q wraps a ring group)
            k = settings.get("SDPA_K_CHUNK")
            seq = 2 * max(math.ceil((self.window - 1) / k) * k, ttnn.TILE_SIZE)
        tag = "full" if self.window is None else "slide"
        return (
            self.ccl.ring_gather_buffer(f"k_{tag}", self.nkv, seq, head_dim, dtype),
            self.ccl.ring_gather_buffer(f"v_{tag}", self.nkv, seq, head_dim, dtype),
        )

    def _split_slot(self, cache: ttnn.Tensor, t: ttnn.Tensor, batch_idx: int, start: int, hi: ttnn.Tensor):
        """A 128-wide copy of this slot's cache for SDPA: [cache | 0] on every row, then the chunk's rows [hi | lo]
        with lo = bfp8(t - hi) (``t`` the fp32 K or V heads, ``hi`` their bfp8 cast). bfp8 throughout."""
        b = cache.shape
        slot = ttnn.slice(
            cache, [batch_idx, 0, 0, 0], [batch_idx + 1, b[1], b[2], b[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        zeros = ttnn.zeros_like(slot, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wide = ttnn.concat([slot, zeros], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        slot.deallocate(True)
        zeros.deallocate(True)
        hi_f = ttnn.typecast(hi, t.dtype)
        lo_f = ttnn.subtract(t, hi_f)
        hi_f.deallocate(True)
        lo = ttnn.typecast(lo_f, ttnn.bfloat8_b)
        lo_f.deallocate(True)
        src = ttnn.concat([hi, lo], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lo.deallocate(True)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            wide, src, slot_idx=0, layer_idx=0, num_layers=1, kv_actual_global=start, cluster_axis=SP_AXIS
        )
        src.deallocate(True)
        return wide

    def __call__(self, x: ttnn.Tensor, start: int, cache_k: ttnn.Tensor, cache_v: ttnn.Tensor, batch_idx: int = 0):
        """x [1, 1, S_local, H] bf16; the chunk's first global position ``start`` (tile-aligned); the layer's caches
        [n_batch, 2, rows_local, 64] bfp8 and its batch slot. Returns [1, 1, S_local, H] bf16."""
        s_local = x.shape[-2]
        qkv = ttnn.linear(
            x,
            self.wqkv,
            bias=self.bqkv,
            dtype=self.qkv_dtype,
            compute_kernel_config=self.mm_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.nq_local,
            num_kv_heads=self.nkv_local,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv.deallocate(True)
        tables = self.rope.tables_at(start) if self.rope.fp32 else None
        q_rot, k_rot = self.rope(q, start, tables), self.rope(k, start, tables)
        for t in tables or ():
            t.deallocate(True)
        q.deallocate(True)
        if q_rot.dtype != ttnn.bfloat16:  # ring SDPA takes a bf16 Q
            q_bf = ttnn.typecast(q_rot, ttnn.bfloat16)
            q_rot.deallocate(True)
            q_rot = q_bf
        k.deallocate(True)
        # full layers: SDPA reads a bf16 copy of this slot's cache holding the chunk's own K / V in bf16 (the served
        # cache still gets bfp8). Full causal attention over the prefix: bfp8 on the chunk's own keys alone is rel 0.06
        sdpa_kv, sdpa_batch = None, batch_idx
        if self.window is None and self.chunk_kv_bf16:
            sdpa_kv, sdpa_batch = [], 0
            for cache, t in ((cache_k, k_rot), (cache_v, v)):
                b = cache.shape
                slot = ttnn.slice(
                    cache,
                    [batch_idx, 0, 0, 0],
                    [batch_idx + 1, b[1], b[2], b[3]],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                shadow = ttnn.typecast(slot, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                slot.deallocate(True)
                src = ttnn.typecast(t, ttnn.bfloat16)
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    shadow, src, slot_idx=0, layer_idx=0, num_layers=1, kv_actual_global=start, cluster_axis=SP_AXIS
                )
                src.deallocate(True)
                sdpa_kv.append(shadow)
        if self.kv_split:
            sdpa_kv, sdpa_batch = [], 0
        for cache, t in ((cache_k, k_rot), (cache_v, v)):
            src = ttnn.typecast(t, ttnn.bfloat8_b)
            if self.kv_split:  # before the served write: the slot copy's prefix must not see the chunk's hi
                sdpa_kv.append(self._split_slot(cache, t, batch_idx, start, src))
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                src,
                slot_idx=batch_idx,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=start,
                cluster_axis=SP_AXIS,
            )
            src.deallocate(True)
        k_rot.deallocate(True)
        v.deallocate(True)
        if self.kv_split:
            q_wide = ttnn.concat([q_rot, q_rot], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q_rot.deallocate(True)
            q_rot = q_wide

        sdpa_k, sdpa_v = sdpa_kv if sdpa_kv else (cache_k, cache_v)
        buf_k, buf_v = self._gather_buffers(cache_k.shape[-2], sdpa_k.dtype, sdpa_k.shape[-1])
        attn, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q_rot,
            sdpa_k,
            sdpa_v,
            None,
            None,
            None,
            persistent_output_buffer_k=buf_k,
            persistent_output_buffer_v=buf_v,
            joint_strategy="rear",
            logical_n=start + s_local * self.sp,
            program_config=self.sdpa_program,
            compute_kernel_config=self.sdpa_config,
            dim=2,
            multi_device_global_semaphore=self.ccl.ring_semaphores,
            num_links=self.ccl.num_links,
            cluster_axis=SP_AXIS,
            mesh_device=self.mesh,
            topology=self.ccl.topology,
            ccl_core_grid_offset=self.ccl.ring_ccl_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=self.scale,
            is_balanced=False,
            kv_cache_batch_idx=sdpa_batch,
            kv_actual_isl=start,
            attention_sink=self.sinks,
            sliding_window_size=self.window,
        )
        q_rot.deallocate(True)
        for t in sdpa_kv or ():
            t.deallocate(True)
        if self.kv_split:  # P.[v_hi | v_lo]: the two halves sum to P.v
            a = attn.shape
            hi = ttnn.slice(attn, [0, 0, 0, 0], [a[0], a[1], a[2], HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            lo = ttnn.slice(attn, [0, 0, 0, HEAD_DIM], [a[0], a[1], a[2], 2 * HEAD_DIM])
            attn.deallocate(True)
            attn = ttnn.add(hi, lo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hi.deallocate(True)
            lo.deallocate(True)
        heads = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        part = ttnn.linear(
            heads,
            self.wo,
            transpose_b=True,
            dtype=ttnn.float32 if self.reduce_fp32 else ttnn.bfloat16,
            compute_kernel_config=self.mm_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        heads.deallocate(True)
        red = ttnn.all_reduce(part, cluster_axis=TP_AXIS, num_links=self.ccl.num_links, topology=self.ccl.topology)
        part.deallocate(True)
        out = ttnn.add(red, self.bo, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        red.deallocate(True)
        return out
