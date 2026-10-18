# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 sliding-window attention for prefill on the 8x4 Galaxy mesh.

Layout: the chunk is replicated on every mesh row; along the 4 columns the heads are split (TP=4). Column c holds
Q heads 4c..4c+3 and KV heads 2c, 2c+1 (GQA 2:1), so attention is local to each chip. o_proj is row-parallel and is
followed by an all_reduce over the columns (cluster_axis=1). Every row computes the same result.

Per chunk at absolute positions [start, start+S):
  q = rope(q_norm(x Wq)), k = rope(k_norm(x Wk)), v = rms(x Wv)   (Gemma RMSNorm x * w, eps 1e-6; V has no weight)
  K/V are written into the bfp8 cache at [start, start+S), then SDPA (causal, window 1024, scale 1.0) runs over the
  square [cache tail (start-hist, start) | chunk]. Q is front-padded with hist dummy rows whose outputs are dropped.
Serving chunks (tt-d-gen): ``start`` is any multiple of 32 and ``end`` (actual_end) may fall inside the chunk; K/V rows
at and past ``end`` are zeroed before the cache write (the record holding actual_end ships whole), so the rows past it
stay zero until a later chunk writes them. The pad rows' own outputs are dropped by the caller.
"""

import math

import torch

import ttnn

from .settings import settings

_FIDELITY = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}

TP_AXIS = 1
TILE = 32


def _kernel_config(mesh, fidelity: str, fp32_acc: bool):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=_FIDELITY[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


def _col_mapper(mesh, dim):
    """Shard tensor dim `dim` over the mesh columns (TP), replicate over the rows."""
    return ttnn.ShardTensor2dMesh(mesh, tuple(mesh.shape), dims=(None, dim))


def nd_kv_memory_config(mesh, width: int):
    """The served KV layout: DRAM ND-sharded, shard [1, 1, 32, width] ROUND_ROBIN_1D over every DRAM bank, so one
    32-token record of one head is contiguous in one bank (one KV table entry is an address and a size)."""
    banks = mesh.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, TILE, width],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def zero_fill(t):
    from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

    DRAMZeroFill.op(t)


def slot_rows(t, slot: int, lo: int, hi: int):
    """Rows [lo, hi) of KV slot `slot` of a [slots, h, cap, D] cache as a new DRAM-interleaved tensor (the caller
    frees it). A slice covering the whole tensor returns the tensor itself, so that case is a layout copy."""
    slots, h, cap, d = (int(x) for x in t.shape)
    if slots == 1 and lo == 0 and hi == cap:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.slice(t, [slot, 0, lo, 0], [slot + 1, h, hi, d], memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TtKVCache:
    """Per-layer K/V cache [slots, Hkv/TP, max_seq, D] bfp8 per chip (column c holds its own KV heads, every mesh row
    the same), in the served layout (nd_kv_memory_config), zero when allocated. K is stored post k_norm + RoPE, V post
    the unscaled RMS (the reference state's layout). ``slot`` is the slot attention reads and writes."""

    def __init__(self, mesh, num_kv_heads: int, head_dim: int, max_seq: int, slots: int = 1):
        self.mesh, self.max_seq, self.slots, self.d = mesh, max_seq, slots, head_dim
        tp = mesh.shape[TP_AXIS]
        assert num_kv_heads % tp == 0, f"{num_kv_heads} kv heads over TP={tp}"
        assert max_seq % TILE == 0, f"cache length {max_seq} not a multiple of {TILE}"
        self.nkv = num_kv_heads // tp
        self.slot = 0
        mc = nd_kv_memory_config(mesh, head_dim)
        shape = ttnn.Shape([slots, self.nkv, max_seq, head_dim])
        self.k = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh, mc)
        self.v = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh, mc)
        self.zero()

    def zero(self):
        zero_fill(self.k)
        zero_fill(self.v)

    def tensors(self) -> dict:
        return {"key": self.k, "value": self.v}

    def load(self, slot: int, k_prefix=None, v_prefix=None):
        """Write host prefixes, head-major [Hkv, n, D] (n may be 0), at rows [0, n) of `slot`; n is padded to whole
        records with zeros. Load time only (host upload)."""
        for t, prefix in ((self.k, k_prefix), (self.v, v_prefix)):
            if prefix is None or prefix.shape[1] == 0:
                continue
            n = prefix.shape[1]
            n_pad = -(-n // TILE) * TILE
            full = torch.zeros(1, prefix.shape[0], n_pad, self.d, dtype=torch.float32)
            full[0, :, :n] = prefix.float()
            up = ttnn.from_torch(
                full,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_col_mapper(self.mesh, 1),
            )
            ttnn.fill_cache(t, up, batch_idx=slot, update_idx=0)
            ttnn.deallocate(up)

    def deallocate(self):
        ttnn.deallocate(self.k)
        ttnn.deallocate(self.v)


class TtSlidingAttention:
    kv_repeat = 1  # chips per KV head along the TP columns

    def __init__(
        self,
        mesh,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        *,
        num_heads,
        num_kv_heads,
        head_dim,
        window,
        rope_inv_freq,
        eps=1e-6,
        scale=1.0,
    ):
        self.mesh = mesh
        tp = mesh.shape[TP_AXIS]
        assert num_heads % tp == 0 and num_kv_heads % tp == 0
        self.nh, self.nkv, self.d = num_heads // tp, num_kv_heads // tp, head_dim
        self.window, self.eps, self.scale = window, eps, scale
        self.inv_freq = rope_inv_freq.float()

        # Fused per-chip QKV weight [in, (4 Q | 2 K | 2 V) heads * D], chip blocks stacked along the output dim.
        qd, kd = self.nh * head_dim, self.nkv * head_dim
        blocks = []
        for c in range(tp):
            blocks += [wq[c * qd : (c + 1) * qd], wk[c * kd : (c + 1) * kd], wv[c * kd : (c + 1) * kd]]
        wqkv = torch.cat(blocks, dim=0).t().contiguous()  # [hidden, tp * (qd + 2 kd)]
        self.wqkv = self._weight(wqkv, dim=-1)
        self.wo = self._weight(wo.t().contiguous(), dim=-2)  # [nh*D, hidden], rows split by Q head
        self.q_norm = self._replicated(q_norm.float().reshape(1, 1, 1, -1))
        self.k_norm = self._replicated(k_norm.float().reshape(1, 1, 1, -1))

        self.mm_cfg = _kernel_config(mesh, settings.get("ATTN_MATMUL_FIDELITY"), settings.get("ATTN_MATMUL_FP32_ACC"))
        self.norm_cfg = _kernel_config(mesh, "HiFi4", True)
        self.sdpa_cfg = _kernel_config(mesh, settings.get("SDPA_FIDELITY"), settings.get("SDPA_FP32_ACC"))
        self.q_chunk, self.k_chunk = settings.get("SDPA_Q_CHUNK"), settings.get("SDPA_K_CHUNK")
        self.cos = self.sin = self.rows = None
        self.rope_impl = settings.get("ROPE_IMPL")
        self.rope_len = 0

    def _weight(self, w, dim):
        return ttnn.from_torch(
            w.float().reshape(1, 1, *w.shape),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_col_mapper(self.mesh, dim),
        )

    def _replicated(self, t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def ensure_rope(self, max_seq: int):
        """Build the rotate-half cos/sin tables [1, 1, max_seq, D] once (load time); forward slices them on device."""
        if max_seq <= self.rope_len:
            return
        pos = torch.arange(max_seq, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        if self.cos is not None:
            ttnn.deallocate(self.cos)
            ttnn.deallocate(self.sin)
        sin = emb.sin()
        if self.rope_impl == "composed":  # rotate_half(x) * sin == swap_halves(x) * [-sin_lo | sin_hi]
            sin = torch.cat([-sin[:, : sin.shape[-1] // 2], sin[:, sin.shape[-1] // 2 :]], dim=-1)
        self.cos = self._replicated(emb.cos().reshape(1, 1, max_seq, -1), ttnn.float32)
        self.sin = self._replicated(sin.reshape(1, 1, max_seq, -1), ttnn.float32)
        if self.rows is not None:
            ttnn.deallocate(self.rows)
        self.rows = self._replicated(pos.reshape(1, 1, max_seq, 1), ttnn.float32)  # local row index (exact in fp32)
        self.rope_len = max_seq

    def _rope_rows(self, start: int, end: int, d: int):
        """cos/sin rows [start, end) of the tables; the full range returns the tables themselves (never free them)."""
        if end - start == self.rope_len:
            return self.cos, self.sin
        return (
            ttnn.slice(self.cos, [0, 0, start, 0], [1, 1, end, d]),
            ttnn.slice(self.sin, [0, 0, start, 0], [1, 1, end, d]),
        )

    def _pad_mask(self, s: int, n: int):
        """[1, 1, s, 1] fp32: 1 for the chunk's rows below n (actual_end - start), 0 for the pad rows; None when every
        row is real. A compare against the load-time row index table, no host work."""
        if n >= s:
            return None
        rows = self.rows if s == self.rope_len else ttnn.slice(self.rows, [0, 0, 0, 0], [1, 1, s, 1])
        mask = ttnn.lt(rows, float(n))
        if rows is not self.rows:
            ttnn.deallocate(rows)
        return mask

    @staticmethod
    def _masked(t, mask):
        """where(mask, t, 0), t freed; t itself when mask is None. A select, not a multiply: the FPU multiply feeds
        fp32 operands as tf32 and would change the real rows' bits, so a pulled-back chunk's rewrite of rows an
        unmasked chunk wrote would differ."""
        if mask is None:
            return t
        out = ttnn.where(mask, t, 0.0)
        ttnn.deallocate(t)
        return out

    def _kv(self, k, v, cos, sin, mask):
        """fp32 K/V heads -> bfp8 (K = rope(k_norm(k)), V = rms(v)), the pad rows zeroed."""
        k8 = self._norm_rope(k, self.k_norm, cos, sin, ttnn.bfloat8_b, mask)
        vn = self._masked(self._head_norm(v, None), mask)
        ttnn.deallocate(v)
        v8 = ttnn.typecast(vn, ttnn.bfloat8_b)
        ttnn.deallocate(vn)
        return k8, v8

    def new_cache(self, max_seq: int, slots: int = 1) -> TtKVCache:
        """An empty (zero) served-layout cache for this layer: `slots` KV slots of max_seq rows."""
        return TtKVCache(self.mesh, self.nkv * self.mesh.shape[TP_AXIS], self.d, max_seq, slots)

    def load_prefix(self, cache: TtKVCache, slot: int, k_prefix=None, v_prefix=None):
        """Write host prefixes in the reference's state exchange layout, position-major [n, Hkv * D] (row = token,
        heads concatenated), into rows [0, n) of `slot`. Each KV head goes on kv_repeat adjacent columns."""
        r = self.kv_repeat

        def head_major(p):
            if p is None:
                return None
            p = p.reshape(p.shape[0], p.shape[1] // self.d, self.d).transpose(0, 1)  # [Hkv, n, D]; n may be 0
            return p.repeat_interleave(r, dim=0) if r > 1 else p

        cache.load(slot, head_major(k_prefix), head_major(v_prefix))

    def make_cache(self, max_seq: int, k_prefix=None, v_prefix=None) -> TtKVCache:
        """One-slot device cache for this layer holding the host prefixes (see load_prefix)."""
        cache = self.new_cache(max_seq)
        self.load_prefix(cache, 0, k_prefix, v_prefix)
        return cache

    def _head_norm(self, t, weight):
        shape = t.shape
        flat = ttnn.reshape(t, (1, 1, shape[1] * shape[2], shape[3]))
        out = ttnn.rms_norm(flat, weight=weight, epsilon=self.eps, compute_kernel_config=self.norm_cfg)
        return ttnn.reshape(out, shape)

    @staticmethod
    def _fp32(t):
        out = ttnn.typecast(t, ttnn.float32)
        ttnn.deallocate(t)
        return out

    def _rope(self, t, cos, sin):
        """rotate-half RoPE of fp32 [1, h, S, D]. composed: t*cos + swap_halves(t)*sin (sin table pre-signed), plain
        eltwise ops whose bits for a row do not depend on the row's place in the chunk (rotary_embedding's do)."""
        if self.rope_impl == "fused":
            return ttnn.experimental.rotary_embedding(t, cos, sin, compute_kernel_config=self.norm_cfg)
        h, s_, d = t.shape[1], t.shape[2], t.shape[3]
        lo = ttnn.slice(t, [0, 0, 0, 0], [1, h, s_, d // 2])
        hi = ttnn.slice(t, [0, 0, 0, d // 2], [1, h, s_, d])
        swapped = ttnn.concat([hi, lo], dim=-1)
        ttnn.deallocate(lo)
        ttnn.deallocate(hi)
        rs = ttnn.multiply(swapped, sin)
        ttnn.deallocate(swapped)
        out = ttnn.multiply(t, cos)
        ttnn.add_(out, rs)
        ttnn.deallocate(rs)
        return out

    def _norm_rope(self, t, weight, cos, sin, dtype, mask=None):
        """fp32 [1, h, S, D] -> rope(rms_norm(t) * w) in `dtype` (rows where mask is 0 zeroed)."""
        tn = self._head_norm(t, weight)
        ttnn.deallocate(t)
        tr = self._rope(tn, cos, sin)
        ttnn.deallocate(tn)
        tr = self._masked(tr, mask)
        out = ttnn.typecast(tr, dtype)
        ttnn.deallocate(tr)
        return out

    def __call__(self, x, cache: TtKVCache, start: int, actual_end: int | None = None):
        """x: [1, 1, S, hidden] bf16 replicated; positions [start, start+S), rows at and past actual_end are pad (their
        cache rows are written as zeros). Returns [1, 1, S, hidden] replicated."""
        s = x.shape[-2]
        end = start + s
        assert start % TILE == 0 and s % TILE == 0, f"start {start} and S {s} must be tile aligned"
        assert end <= cache.max_seq and end <= self.rope_len, f"chunk end {end} past cache/rope ({cache.max_seq})"
        n = s if actual_end is None else actual_end - start
        assert 0 < n <= s, f"actual_end {actual_end} not in ({start}, {end}]"
        nh, nkv, d = self.nh, self.nkv, self.d

        xqkv = ttnn.linear(x, self.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(xqkv)

        # Norms and RoPE in fp32 (HiFi4 + fp32 dest). K/V must reach bfp8 from real fp32 values: the device rounds
        # bf16 -> bfp8 ties away from zero (bf16 values sit exactly on bfp8 ties), which grew K norms by ~0.3 % and
        # the attention output by ~0.4 %; fp32 -> bfp8 is unbiased. rotary_embedding's default config adds ~0.2 %.
        q, k, v = (self._fp32(t) for t in (q, k, v))
        cos, sin = self._rope_rows(start, end, d)
        q = self._norm_rope(q, self.q_norm, cos, sin, ttnn.bfloat16)
        mask = self._pad_mask(s, n)
        k8, v8 = self._kv(k, v, cos, sin, mask)
        if mask is not None:
            ttnn.deallocate(mask)
        if end - start < self.rope_len:  # a full-range slice is the table itself: keep it
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)

        # Write the chunk into the cache slot (bfp8), then read back the window tail + chunk as K/V for SDPA.
        ttnn.fill_cache(cache.k, k8, batch_idx=cache.slot, update_idx=start)
        ttnn.fill_cache(cache.v, v8, batch_idx=cache.slot, update_idx=start)
        ttnn.deallocate(k8)
        ttnn.deallocate(v8)
        hist = min(start, ((self.window + TILE - 1) // TILE) * TILE)
        k_win = slot_rows(cache.k, cache.slot, start - hist, end)
        v_win = slot_rows(cache.v, cache.slot, start - hist, end)
        if hist:
            q_pad = ttnn.slice(q, [0, 0, 0, 0], [1, nh, hist, d])  # dummy queries, outputs dropped
            q_sq = ttnn.concat([q_pad, q], dim=2)
            ttnn.deallocate(q_pad)
            ttnn.deallocate(q)
        else:
            q_sq = q
        n = hist + s
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=min(self.q_chunk, n),
            k_chunk_size=min(self.k_chunk, n),
            exp_approx_mode=False,
        )
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_sq,
            k_win,
            v_win,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.window,
            program_config=program_config,
            compute_kernel_config=self.sdpa_cfg,
        )
        ttnn.deallocate(q_sq)
        ttnn.deallocate(k_win)
        ttnn.deallocate(v_win)
        if hist:
            full = attn
            attn = ttnn.slice(full, [0, 0, hist, 0], [1, nh, n, d])
            ttnn.deallocate(full)

        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        partial = ttnn.linear(concat, self.wo, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(concat)
        out = ttnn.all_reduce(partial, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        return out


class TtGlobalAttention(TtSlidingAttention):
    """Gemma-4 global (full causal) attention: 16 Q heads x 512, 2 KV heads x 512, no v_proj.

    Column c holds Q heads 4c..4c+3 and KV head c // 2 (each KV head lives on 2 adjacent columns), so attention stays
    local to each chip (GQA 4:1). The per-chip fused weight is [Wq_c | Wk_{c//2}] and `nlp_create_qkv_heads(kv_tied)`
    reads raw K twice: K = rope(k_norm(k_raw)), V = rms(k_raw) without a weight. RoPE is the rotate-half op with the
    proportional inv_freq (zeros past 25 %: cos 1, sin 0 there). SDPA is chunked causal SDPA over the cache slot (copied to an
    interleaved [1, 1, max_seq, D] tensor) seen as pages of KV_BLOCK rows with an identity page table (one head per chip,
    so the paged view is a zero-cost reshape of the copy).
    """

    kv_repeat = 2
    KV_BLOCK = 64

    def __init__(
        self, mesh, wq, wk, wo, q_norm, k_norm, *, num_heads, num_kv_heads, head_dim, rope_inv_freq, eps=1e-6, scale=1.0
    ):
        self.mesh = mesh
        tp = mesh.shape[TP_AXIS]
        assert num_heads % tp == 0 and num_kv_heads * self.kv_repeat == tp, (num_heads, num_kv_heads, tp)
        self.nh, self.nkv, self.d = num_heads // tp, 1, head_dim
        self.window, self.eps, self.scale = None, eps, scale
        self.inv_freq = rope_inv_freq.float()

        qd = self.nh * head_dim
        blocks = []
        for c in range(tp):
            h = c // self.kv_repeat
            blocks += [wq[c * qd : (c + 1) * qd], wk[h * head_dim : (h + 1) * head_dim]]
        self.wqkv = self._weight(torch.cat(blocks, dim=0).t().contiguous(), dim=-1)  # [hidden, tp * (qd + D)]
        self.wo = self._weight(wo.t().contiguous(), dim=-2)  # [nh*D, hidden], rows split by Q head
        self.q_norm = self._replicated(q_norm.float().reshape(1, 1, 1, -1))
        self.k_norm = self._replicated(k_norm.float().reshape(1, 1, 1, -1))

        self.mm_cfg = _kernel_config(mesh, settings.get("ATTN_MATMUL_FIDELITY"), settings.get("ATTN_MATMUL_FP32_ACC"))
        self.norm_cfg = _kernel_config(mesh, "HiFi4", True)
        self.sdpa_cfg = _kernel_config(mesh, settings.get("SDPA_FIDELITY"), settings.get("SDPA_FP32_ACC"))
        self.q_chunk, self.k_chunk = settings.get("GLOBAL_SDPA_Q_CHUNK"), settings.get("GLOBAL_SDPA_K_CHUNK")
        self.cos = self.sin = self.rows = None
        self.rope_impl = settings.get("ROPE_IMPL")
        self.rope_len = 0
        self.page_table = None

    def ensure_rope(self, max_seq: int):
        """RoPE tables and the identity page table for max_seq, built once (load time)."""
        grow = max_seq > self.rope_len
        super().ensure_rope(max_seq)
        if grow or self.page_table is None:
            assert max_seq % self.KV_BLOCK == 0, f"max_seq {max_seq} not a multiple of {self.KV_BLOCK}"
            if self.page_table is not None:
                ttnn.deallocate(self.page_table)
            self.page_table = ttnn.from_torch(
                torch.arange(max_seq // self.KV_BLOCK, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

    def __call__(self, x, cache: TtKVCache, start: int, actual_end: int | None = None):
        """x: [1, 1, S, hidden] bf16 replicated; positions [start, start+S), rows at and past actual_end are pad (their
        cache rows are written as zeros). Returns [1, 1, S, hidden] replicated."""
        s = x.shape[-2]
        end = start + s
        assert start % TILE == 0 and s % TILE == 0, f"start {start} and S {s} must be tile aligned"
        assert end <= cache.max_seq <= self.rope_len, f"chunk end {end} / cache {cache.max_seq} / rope {self.rope_len}"
        n = s if actual_end is None else actual_end - start
        assert 0 < n <= s, f"actual_end {actual_end} not in ({start}, {end}]"
        nh, d = self.nh, self.d

        xqkv = ttnn.linear(x, self.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=nh,
            num_kv_heads=1,
            transpose_k_heads=False,
            kv_tied=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # fp32 norms + RoPE; K/V reach bfp8 from fp32 (see TtSlidingAttention.__call__).
        q, k, v = (self._fp32(t) for t in (q, k, v))
        cos, sin = self._rope_rows(start, end, d)
        q = self._norm_rope(q, self.q_norm, cos, sin, ttnn.bfloat16)
        mask = self._pad_mask(s, n)
        k8, v8 = self._kv(k, v, cos, sin, mask)
        if mask is not None:
            ttnn.deallocate(mask)
        if end - start < self.rope_len:  # a full-range slice is the table itself: keep it
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)

        ttnn.fill_cache(cache.k, k8, batch_idx=cache.slot, update_idx=start)
        ttnn.fill_cache(cache.v, v8, batch_idx=cache.slot, update_idx=start)
        ttnn.deallocate(k8)
        ttnn.deallocate(v8)

        # Chunked SDPA wants chunk_start_idx on a q/k chunk boundary; a start that is only tile aligned (a follow-up
        # turn) runs from the boundary below it, Q front-padded with `pre` dummy rows whose outputs are dropped.
        q_chunk, k_chunk = min(self.q_chunk, s), min(self.k_chunk, s)
        pre = start % math.lcm(q_chunk, k_chunk)
        if pre:
            q_pad = ttnn.slice(q, [0, 0, 0, 0], [1, nh, pre, d])
            q_sq = ttnn.concat([q_pad, q], dim=2)
            ttnn.deallocate(q_pad)
            ttnn.deallocate(q)
        else:
            q_sq = q
        # The slot as an interleaved [1, 1, max_seq, D] copy, seen as pages (a zero-cost reshape of the copy).
        pages = (cache.max_seq // self.KV_BLOCK, 1, self.KV_BLOCK, d)
        k_slot = slot_rows(cache.k, cache.slot, 0, cache.max_seq)
        v_slot = slot_rows(cache.v, cache.slot, 0, cache.max_seq)
        k_paged = ttnn.reshape(k_slot, pages)
        v_paged = ttnn.reshape(v_slot, pages)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=q_sq,
            input_tensor_k=k_paged,
            input_tensor_v=v_paged,
            page_table_tensor=self.page_table,
            chunk_start_idx=start - pre,
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=self.sdpa_cfg,
        )
        ttnn.deallocate(q_sq)
        ttnn.deallocate(k_slot)
        ttnn.deallocate(v_slot)
        if pre:
            full = attn
            attn = ttnn.slice(full, [0, 0, pre, 0], [1, nh, pre + s, d])
            ttnn.deallocate(full)

        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        partial = ttnn.linear(concat, self.wo, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(concat)
        out = ttnn.all_reduce(partial, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        return out
