# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compressed Sparse Attention (V4-Flash even layers 2..42) in ttnn -- plan path B (dense mask).

``TtCSACompressor`` (two-series rate-4 compressor, W = 512 for the attention entries and W = 128 for the lightning
indexer's keys) follows ``tt/v4/attention/csa_math.py`` exactly: kv/gate are SP-all-gathered so every chip computes
every entry of the chunk (replicated, no cross-shard window dependency), the 8-slot softmax is done as exp with one
per-channel constant + constant 0/1 window-sum / shift matmuls, window 0 takes the previous chunk's last-window Ca
rows from the state. ``TtCSAIndexer`` scores every (query, entry) pair per head (``relu(q_h . k) * w_h``, TP over
heads, all-reduce), applies the causal cut, and turns the top-512 into an additive 0/-inf selection mask by
THRESHOLD (k-th largest score per row), so no scatter is needed. ``TtCSA`` is ``TtHCA`` with that compressor, the
indexer's mask folded into the compressed columns of the persistent mask, and tile-aligned entry writes (1280
entries per 5120-token chunk -- no tail tile). Attention itself is TtHCA's dense SDPA over
``[carry | chunk | entries(cap) | pad]`` with sinks -- correct at any length, ~4x HCA's cost at 64k (path A,
``sparse_sdpa`` + sink, is plan M8).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import (
    TtHCA,
    TtHCACompressor,
    TtHCAState,
    _rope_table_tokens,
    _TtHCABase,
)
from models.demos.deepseek_v3_d_p.tt.v4.attention import csa_math as C

__all__ = ["TtCSACompressor", "TtCSAIndexer", "TtCSA", "TtCSAState"]


class TtCSAState(TtHCAState):
    """TtHCAState plus the indexer key cache and the two compressors' overlap priors (last 32 real Ca rows of the
    previous chunk: kv and biased gate, fp32)."""

    def __init__(self, *, compressed_kv, index_k, sliding_carry, prior_c, prior_i, max_seq_len):
        super().__init__(compressed_kv=compressed_kv, sliding_carry=sliding_carry, tail=None, max_seq_len=max_seq_len)
        self.index_k = index_k
        self.prior_c = prior_c  # (kv_a_last32 [1,1,32,512], gate_a_last32 [1,1,32,512])
        self.prior_i = prior_i  # (kv_a_last32 [1,1,32,128], gate_a_last32 [1,1,32,128])


class TtCSACompressor(TtHCACompressor):
    """Two-series compressor at width W (= ``head_dim`` argument): ``kv_proj``/``gate_proj`` are ``[2W, hidden]``,
    ``position_bias`` ``[rate, 2W]``. ``forward`` returns entries for the WHOLE chunk on every chip."""

    def __init__(self, device, *, position_bias, **kwargs):
        # the base class reshapes position_bias to [rate, head_dim]; ours is [rate, 2*head_dim] -- upload it below
        rate, two_w = position_bias.shape
        super().__init__(device, position_bias=torch.zeros(rate, two_w // 2), **kwargs)
        assert two_w == 2 * self.head_dim, (tuple(position_bias.shape), self.head_dim)
        self._bias_host = position_bias.detach().float()
        self.position_bias = None
        self.fp32 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._pool_consts = None

    @classmethod
    def from_reference(cls, device, reference, config, *, head_dim=None, **kwargs) -> "TtCSACompressor":
        """``reference`` is a DeepseekV4CSACompressor (W = head_dim) or a DeepseekV4Indexer (W = index_head_dim)."""
        return cls(
            device,
            kv_proj_weight=reference.kv_proj.weight,
            gate_proj_weight=reference.gate_proj.weight,
            position_bias=reference.position_bias,
            kv_norm_weight=reference.kv_norm.weight,
            head_dim=int(head_dim if head_dim is not None else reference.head_dim),
            compress_rate=config.compress_rates["compressed_sparse_attention"],
            rope_head_dim=config.qk_rope_head_dim,
            rotary_emb=reference.rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def _f32(self, x: torch.Tensor):
        return self._from_torch(x.float().contiguous(), dtype=ttnn.float32)

    def alloc_tables(self, max_seq_len: int, chunk_tokens: int, mask_width: int):
        super().alloc_tables(max_seq_len, chunk_tokens, mask_width)
        S, rate, W = int(chunk_tokens), self.compress_rate, self.head_dim
        n = S // rate
        # every chip holds every entry of the chunk: a replicated entry-rope index (base pushed per chunk)
        self._entry_index_full = (
            self._from_torch(
                torch.arange(n, dtype=torch.int32).view(1, n), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            self._scalar_buffer(ttnn.uint32, shape=(1, 1), layout=ttnn.ROW_MAJOR_LAYOUT),
        )
        self._pool_consts = {
            "G": self._f32(C.group_sum_matrix(S, rate).view(1, 1, n, S)),
            "Sh": self._f32(C.shift_matrix(n).view(1, 1, n, n)),
            "Sel": self._f32(C.prior_select_matrix(rate).view(1, 1, C.TILE, C.TILE)),
            "P0": self._f32(C.first_window_matrix(n).view(1, 1, n, C.TILE)),
            "BIAS": self._f32(C.bias_rows(self._bias_host, S).view(1, 1, S, 2 * W)),
            "rows": S,
        }
        self._prior_index = {}

    def empty_prior(self, batch: int = 1):
        pk, pg = C.empty_prior(self.head_dim)
        return (self._f32(pk.view(1, 1, C.TILE, self.head_dim)), self._f32(pg.view(1, 1, C.TILE, self.head_dim)))

    def reset_prior(self, prior: tuple) -> None:
        """Refill a slot's persistent prior pair with the empty prior by device copies (no allocation, no host
        write): the constant empties are built once per compressor."""
        const = self.__dict__.get("_empty_prior_const")
        if const is None:
            const = self._empty_prior_const = self.empty_prior()
        for dst, src in zip(prior, const):
            ttnn.copy(src, dst)

    def _mm(self, a, b):
        return ttnn.matmul(a, b, dtype=ttnn.float32, memory_config=self.memory_config, compute_kernel_config=self.fp32)

    def _sp_gather(self, x):
        if self.sp_factor == 1:
            return x
        return ttnn.experimental.all_gather_async(
            x,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.sp_axis,
        )

    def _tp_all_reduce(self, x):
        if self.tp_factor == 1:
            return x
        x = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.tp_axis,
        )
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.tp_axis,
        )

    def _last32(self, x, real_len: int):
        """Rows [real_len - 32, real_len) of a replicated [1,1,S,W] slab (start as a device tensor; 32-aligned)."""
        S = int(x.shape[2])
        assert real_len % C.TILE == 0 and C.TILE <= real_len <= S, real_len
        pair = self._prior_index.get(int(real_len))
        if pair is None:

            def idx(vals):
                return self._from_torch(
                    torch.tensor(vals, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                )

            pair = self._prior_index[int(real_len)] = (
                idx([0, 0, real_len - C.TILE, 0]),
                idx([1, 1, real_len, x.shape[3]]),
            )
        return ttnn.slice(x, pair[0], pair[1], slice_dim=2, num_devices=S // C.TILE)

    def forward(self, hidden_states, seq_len_actual: int, first_window_position: int, prior: tuple):
        """-> (entries bf16 [1, 1, S/rate, W] replicated on every chip, mask_block [1, 1, S_l, mask_width],
        new_prior). ``seq_len_actual`` is the chunk's real length (a multiple of 32 unless it is the final chunk,
        whose prior nobody reads)."""
        K = self._pool_consts
        assert K is not None, "alloc_tables first"
        W, rate = self.head_dim, self.compress_rate
        S_l = hidden_states.shape[2]
        kv = self._tp_all_reduce(ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config))
        gate = self._tp_all_reduce(ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config))
        kv = ttnn.typecast(self._sp_gather(kv), ttnn.float32)  # [1, 1, S, 2W]
        gate = ttnn.add(ttnn.typecast(self._sp_gather(gate), ttnn.float32), K["BIAS"])
        S = kv.shape[2]
        assert S == K["rows"], (S, K["rows"])
        kv_a, kv_b = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, S, W]), ttnn.slice(kv, [0, 0, 0, W], [1, 1, S, 2 * W])
        g_a, g_b = ttnn.slice(gate, [0, 0, 0, 0], [1, 1, S, W]), ttnn.slice(gate, [0, 0, 0, W], [1, 1, S, 2 * W])
        pk, pg = prior
        # one per-channel constant for the whole chunk (softmax-invariant, keeps exp in range)
        M = ttnn.maximum(
            ttnn.maximum(ttnn.max(g_a, dim=2, keepdim=True), ttnn.max(g_b, dim=2, keepdim=True)),
            ttnn.max(pg, dim=2, keepdim=True),
        )
        E_a, E_b, E_p = ttnn.exp(ttnn.subtract(g_a, M)), ttnn.exp(ttnn.subtract(g_b, M)), ttnn.exp(ttnn.subtract(pg, M))
        den_b, num_b = self._mm(K["G"], E_b), self._mm(K["G"], ttnn.multiply(E_b, kv_b))
        den_a, num_a = self._mm(K["G"], E_a), self._mm(K["G"], ttnn.multiply(E_a, kv_a))
        den_a = ttnn.add(self._mm(K["Sh"], den_a), self._mm(K["P0"], self._mm(K["Sel"], E_p)))
        num_a = ttnn.add(self._mm(K["Sh"], num_a), self._mm(K["P0"], self._mm(K["Sel"], ttnn.multiply(E_p, pk))))
        entries = ttnn.div(ttnn.add(num_a, num_b), ttnn.add(den_a, den_b))  # [1, 1, n, W] fp32
        entries = ttnn.rms_norm(
            ttnn.typecast(entries, self.dtype), weight=self.kv_norm_weight, epsilon=self.rms_norm_eps
        )
        n = entries.shape[2]
        nope_dim = W - self.rope_head_dim
        nope = ttnn.slice(entries, [0, 0, 0, 0], [1, 1, n, nope_dim])
        rope = ttnn.slice(entries, [0, 0, 0, nope_dim], [1, 1, n, W])
        cos, sin = self._rope_gather(
            self._entry_rope, self._rope_index(self._entry_index_full, first_window_position // rate)
        )
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        entries = ttnn.concat([nope, rope], dim=-1)
        # the next chunk's window 0 needs this chunk's last real window's Ca rows (kv and biased gate)
        # Only a chunk that another chunk follows needs to leave a prior, and such a chunk is tile-aligned (asserted by
        # TtCSA.forward on the next call); a ragged FINAL chunk keeps the old prior, which nobody reads.
        aligned = seq_len_actual % C.TILE == 0 and seq_len_actual >= C.TILE
        new_prior = (self._last32(kv_a, seq_len_actual), self._last32(g_a, seq_len_actual)) if aligned else prior
        mask_block = self._mask_block(S_l, first_window_position, seq_len_actual)
        return entries, mask_block, new_prior


class TtCSAIndexer(_TtHCABase):
    """Lightning indexer: keys = a rate-4 compressor at index_head_dim, queries from the shared q_a latent,
    scores = sum_h relu(q_h . k) * w_h (scaled), causal cut, top-k by threshold -> additive 0/-inf mask."""

    def __init__(
        self,
        device,
        *,
        compressor: TtCSACompressor,
        q_b_proj_weight: torch.Tensor,
        weights_proj_weight: torch.Tensor,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        topk: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_cache_path=None,
        cache_name_prefix=None,
    ):
        self.device, self.dtype, self.weights_dtype, self.memory_config = device, dtype, weights_dtype, memory_config
        self.weight_cache_path, self.cache_name_prefix = weight_cache_path, cache_name_prefix
        self.compressor = compressor
        self.n_heads, self.head_dim, self.rope_head_dim, self.topk = (
            int(n_heads),
            int(head_dim),
            int(rope_head_dim),
            int(topk),
        )
        self.is_mesh = hasattr(device, "shape")
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.sp_factor = device.shape[sp_axis] if self.is_mesh else 1
        self.tp_factor = device.shape[tp_axis] if self.is_mesh else 1
        self.ccl_topology = topology
        self.tt_ccl = compressor.tt_ccl
        self.ccl_num_links = compressor.ccl_num_links
        self.fp32 = compressor.fp32
        self.trans_mat = compressor.trans_mat
        self.wq_b = self._to_tt_linear_weight(q_b_proj_weight, tp_shard_dim=3, cache_name="wq_b")  # heads over TP
        self.w_proj = self._to_tt_linear_weight(weights_proj_weight, tp_shard_dim=2, cache_name="w_proj")  # TP partials
        # chip c picks its own heads out of the TP-replicated weights row: a one-hot [H, H] sharded on columns
        self.head_sel = self._from_torch(
            torch.eye(self.n_heads).view(1, 1, self.n_heads, self.n_heads), mesh_mapper=self._mesh_mapper(tp_dim=3)
        )
        self.scale = float(self.head_dim**-0.5 * self.n_heads**-0.5)
        self._gather_bufs = {}

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtCSAIndexer":
        """``reference`` is a DeepseekV4Indexer."""
        ckeys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype", "memory_config")
        comp = TtCSACompressor.from_reference(
            device, reference, config, head_dim=config.index_head_dim, **{k: kwargs[k] for k in ckeys if k in kwargs}
        )
        return cls(
            device,
            compressor=comp,
            q_b_proj_weight=reference.q_b_proj.weight,
            weights_proj_weight=reference.scorer.weights_proj.weight,
            n_heads=config.index_n_heads,
            head_dim=config.index_head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            topk=config.index_topk,
            **kwargs,
        )

    def _tp_all_reduce_via_gather(self, t):
        if self.tp_factor == 1:
            return t
        key = tuple(t.shape)
        buf = self._gather_bufs.get(key)
        if buf is None:
            buf = self._gather_bufs[key] = self._from_torch(
                torch.zeros(1, self.tp_factor, key[2], key[3]), dtype=t.dtype
            )
        g = ttnn.experimental.high_bw_all_gather(
            t, dim=1, output_tensor=buf, num_links=self.ccl_num_links, cluster_axis=self.tp_axis
        )
        return ttnn.experimental.fast_reduce_nc(g, dims=[1], output=None, compute_kernel_config=self.fp32)

    def select(self, q_latent, hidden_states, cos, sin, keys, mask_block):
        """-> additive [1, 1, S_l, cap] mask: 0 on the top-k causally visible entries of each query, -inf else.
        ``keys`` is the index-key cache [1, 1, cap, Dh] (replicated), ``mask_block`` the causal 0/-inf block."""
        S_l = q_latent.shape[2]
        heads_local = self.n_heads // self.tp_factor
        q = ttnn.linear(q_latent, self.wq_b, memory_config=self.memory_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=heads_local, num_kv_heads=0, transpose_k_heads=False, memory_config=self.memory_config
        )
        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(q, [0, 0, 0, 0], [1, heads_local, S_l, nope_dim])
        rope = ttnn.slice(q, [0, 0, 0, nope_dim], [1, heads_local, S_l, self.head_dim])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        q = ttnn.concat([nope, rope], dim=-1)  # [1, H_l, S_l, Dh]
        scores = ttnn.relu(
            ttnn.matmul(q, keys, transpose_b=True, memory_config=self.memory_config)
        )  # [1, H_l, S_l, cap]
        w = ttnn.linear(hidden_states, self.w_proj, memory_config=self.memory_config)  # partial [1, 1, S_l, H]
        w = self.compressor._tp_all_reduce(w)
        w = ttnn.permute(
            ttnn.matmul(w, self.head_sel, memory_config=self.memory_config), (0, 3, 2, 1)
        )  # [1, H_l, S_l, 1]
        scores = ttnn.multiply(ttnn.sum(ttnn.multiply(scores, w), dim=1, keepdim=True), self.scale)  # [1, 1, S_l, cap]
        scores = ttnn.add(self._tp_all_reduce_via_gather(scores), mask_block)
        cap = scores.shape[3]
        k = min(self.topk, cap)
        theta = ttnn.min(ttnn.topk(scores, k=k, dim=-1, largest=True, sorted=True)[0], dim=-1, keepdim=True)
        return ttnn.add(ttnn.log(ttnn.ge(scores, theta)), mask_block)


class TtCSA(TtHCA):
    """CSA block = TtHCA's stems / window carry / sinks / dense SDPA / o-proj with the two-series compressor, the
    indexer's selection mask on the compressed columns, and tile-aligned entry writes (path B)."""

    def __init__(self, device, *, compressor: TtCSACompressor, indexer: TtCSAIndexer, **kwargs):
        kwargs.setdefault("rope_layer_type", "compress")
        super().__init__(device, compressor=compressor, **kwargs)
        self.indexer = indexer

    @staticmethod
    def prepare_input(hidden, sp_factor: int, compress_rate: int = ttnn.TILE_SIZE):
        return TtHCA.prepare_input(hidden, sp_factor, ttnn.TILE_SIZE)

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtCSA":
        """``reference`` is a DeepseekV4Attention of a compressed_sparse_attention layer."""
        ckeys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype", "memory_config")
        sub = {k: kwargs[k] for k in ckeys if k in kwargs}
        compressor = TtCSACompressor.from_reference(device, reference.compressor, config, **sub)
        indexer = TtCSAIndexer.from_reference(device, reference.compressor.indexer, config, **sub)
        return cls(
            device,
            compressor=compressor,
            indexer=indexer,
            q_a_proj_weight=reference.q_a_proj.weight,
            q_a_norm_weight=reference.q_a_norm.weight,
            q_b_proj_weight=reference.q_b_proj.weight,
            kv_proj_weight=reference.kv_proj.weight,
            kv_norm_weight=reference.kv_norm.weight,
            sinks=reference.sinks,
            o_a_proj_weight=reference.o_a_proj.weight,
            o_b_proj_weight=reference.o_b_proj.weight,
            rotary_emb=reference.compressor.rotary_emb,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            sliding_window=config.sliding_window,
            o_groups=config.o_groups,
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None) -> TtCSAState:
        rate = self.compressor.compress_rate
        chunk = chunk_tokens or max_seq_len
        align = ttnn.TILE_SIZE * self.sp_factor
        assert (
            chunk % align == 0 and chunk % rate == 0
        ), f"chunk {chunk} must be a multiple of TILE * sp ({align}) and of {rate}"
        entries = -(-int(max_seq_len) // rate)
        # + one chunk's worth of entries: the final chunk writes its whole padded width (pad-derived entries are -inf-masked)
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE + chunk // rate
        self._build_carry_index(chunk)
        self._build_masks(chunk, capacity)
        self.compressor.alloc_tables(max_seq_len, chunk, capacity)
        self.indexer.compressor.alloc_tables(max_seq_len, chunk, capacity)
        self._slab_rope = self._build_rope_table(_rope_table_tokens(max_seq_len, chunk), 1)
        self._slab_index = self._rope_index_base(chunk // self.sp_factor)
        return TtCSAState(
            compressed_kv=self._from_torch(torch.zeros(batch, 1, capacity, self.head_dim)),
            index_k=self._from_torch(torch.zeros(batch, 1, capacity, self.indexer.head_dim)),
            sliding_carry=self._from_torch(torch.zeros(batch, 1, self.sliding_window, self.head_dim)),
            prior_c=self.compressor.empty_prior(batch),
            prior_i=self.indexer.compressor.empty_prior(batch),
            max_seq_len=max_seq_len,
        )

    def forward(self, hidden_states, seq_len_actual: int | None = None, *, state: TtCSAState, export=None):
        batch = hidden_states.shape[0]
        seq_pad_global = hidden_states.shape[2] * self.sp_factor
        rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else int(seq_len_actual)
        assert batch == 1, f"CSA prefill expects batch 1, got {batch}"
        assert state.kv_actual + real_len <= state.max_seq_len, "context longer than the state was allocated for"
        # Non-final chunks must be tile-aligned (the entry write lands on a tile boundary and the carry / prior
        # slices take 32-aligned starts); only the final chunk may be ragged.
        assert (
            state.kv_actual % ttnn.TILE_SIZE == 0
        ), f"cannot append after a chunk with {state.kv_actual % 32} leftover tokens"
        n_new = real_len // rate
        fwp = state.entry_count * rate

        cos, sin = self._rope_gather(self._slab_rope, self._rope_index(self._slab_index, state.kv_actual))
        q, q_latent = self._q_stem(hidden_states, cos, sin, return_latent=True)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)

        entries, mask_block, new_prior_c = self.compressor(hidden_states, real_len, fwp, state.prior_c)
        keys, _, new_prior_i = self.indexer.compressor(hidden_states, real_len, fwp, state.prior_i)
        for persistent, new in zip(state.prior_c + state.prior_i, new_prior_c + new_prior_i):
            self._update_in_place(persistent, new)
        # the whole padded width is written (pad-derived entries are -inf-masked); tile-aligned, no tail tile
        assert state.entry_count % ttnn.TILE_SIZE == 0 and entries.shape[2] % ttnn.TILE_SIZE == 0
        assert state.entry_count + entries.shape[2] <= state.compressed_kv.shape[2], "compressed cache full"
        ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, entries, 0, update_idx=state.entry_count)
        ttnn.kv_cache.fill_cache_for_user_(state.index_k, keys, 0, update_idx=state.entry_count)

        mask_sel = self.indexer.select(q_latent, hidden_states, cos, sin, state.index_k, mask_block)
        self.debug_last_selection = mask_sel  # tests compare the selected sets with the reference block bias
        attn, next_carry, slab = self._attention(
            q,
            sliding_kv,
            state.compressed_kv,
            mask_sel,
            cos,
            sin,
            carry=state.sliding_carry,
            kv_actual=state.kv_actual,
            real_len=real_len,
        )
        if export is not None:
            self._export_csa(export, entries, keys, slab, state, real_len)
        state.entry_count += n_new
        state.kv_actual += real_len
        self._update_in_place(state.sliding_carry, next_carry)
        return self._o_proj(attn)

    # ---- export into the contract's unified caches (M5b) -------------------------------------------------------
    def _write_rm(self, cache, block_tile, batch_idx: int, row: int):
        """Write ``block_tile`` [1, 1, H, W] (TILE, module dtype) at unified row ``row`` of batch ``batch_idx`` of a
        bf16 ROW_MAJOR ND-sharded cache, identically on every chip. ``update_padded_kv_cache`` derives a per-chip
        offset from ``kv_actual_global`` and the chip's SP coordinate; with ``kv_actual_global = r * sp`` and
        ``r % piece == 0`` every chip lands on local row ``r`` (the writer kernel's slab arithmetic), so the block is
        cut into ``piece``-row pieces, the largest of 128 / 64 / 32 that divides both ``row`` and ``H`` (128 for the
        engine's 5120-token chunks: 1280 entries per chunk)."""
        H, W = int(block_tile.shape[2]), int(block_tile.shape[3])
        assert row % ttnn.TILE_SIZE == 0 and H % ttnn.TILE_SIZE == 0, (row, H)
        assert row + H <= int(cache.shape[2]), f"unified cache too small: rows [{row}, {row + H}) into {cache.shape[2]}"
        piece = 128
        while row % piece or H % piece:
            piece //= 2
        src = block_tile if block_tile.dtype == cache.dtype else ttnn.typecast(block_tile, cache.dtype)
        rm = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT)
        for j in range(H // piece):
            part = rm if H == piece else ttnn.slice(rm, [0, 0, j * piece, 0], [1, 1, (j + 1) * piece, W])
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                part,
                slot_idx=int(batch_idx),
                layer_idx=0,
                num_layers=1,
                kv_actual_global=(row + j * piece) * self.sp_factor,
                cluster_axis=self.sp_axis,
            )

    def _export_csa(self, export, entries, keys, slab, state, real_len):
        """``export = (csa_unified, csa_index_k, batch_idx)``: this chunk's entries -> unified rows
        ``128 + entry_count ..`` (bf16 ROW_MAJOR), its index keys -> the key cache (bfp8 tiles) at ``entry_count``,
        and the window ring -> unified rows [0, 128)."""
        unified, index_k, batch_idx = export
        row = self.sliding_window + int(state.entry_count)
        self._write_rm(unified, entries, batch_idx, row)
        k = keys if keys.dtype == index_k.dtype else ttnn.typecast(keys, index_k.dtype)
        ttnn.kv_cache.fill_cache_for_user_(index_k, k, int(batch_idx), update_idx=int(state.entry_count))
        ring = self._ring_rows(slab, state.sliding_carry, state.kv_actual, real_len)
        self._write_rm(unified, ring, batch_idx, 0)
