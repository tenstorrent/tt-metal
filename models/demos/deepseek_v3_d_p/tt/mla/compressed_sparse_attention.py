# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Compressed Sparse Attention (TTNN prefill).
Mirrors ``DeepseekV4Attention`` on a ``compressed_sparse_attention`` layer.

CSA compresses every 4 tokens instead of every 128 and runs a Lightning Indexer over the compressed
entries, so each query attends to at most ``index_topk`` of them. The attention core is HCA's, in
``v4_attention_base.py``; what is here is the ratio-4 cache append and the block mask, which ANDs the
indexer's per-query picks into the causal compressed columns.

The mask route attends densely over the whole compressed cache and lets the mask drop the unpicked
entries, exactly as the reference does. That costs O(S * S/4) instead of O(S * index_topk), so this is
the correctness path; long context wants the entries gathered instead, which needs a sparse kernel that
supports V4's attention sinks."""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCSACompressor, csa_slab_align, rope_table_tokens
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtCsaIndexer
from models.demos.deepseek_v3_d_p.tt.mla.v4_attention_base import TtV4AttentionBase
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache


def block_mask(causal_block, top_k, *, template, zeros, dump: int):
    """AND a query's indexer picks into its causal compressed columns.

    Both operands are 0 / -inf, so adding them IS the AND: an entry survives only if it is causally
    visible and the indexer selected it. That AND is also what keeps a ragged final chunk honest -- the
    indexer scores the padded slab, but every entry a pad window produced sits past the causal threshold
    of every real query row.

    ``top_k`` arrives as ROW_MAJOR uint32 with ``0xFFFFFFFF`` where the reference has -1. ``template`` is
    a -inf slab whose width is the next power of two above the cache capacity and ``dump`` that width
    minus one, so masking a pick with it maps every real entry to itself and every sentinel to the last
    column -- which is past the capacity, so slicing the mask down drops it.

    The round-trip through TILE is because the bitwise op is an SFPU one, while scatter refuses an index
    this wide in tiled layout."""
    batch, rows, width = causal_block.shape[0], causal_block.shape[2], causal_block.shape[3]

    picks = ttnn.to_layout(top_k, ttnn.TILE_LAYOUT)
    bounded = ttnn.bitwise_and(picks, dump)
    ttnn.deallocate(picks)
    index = ttnn.to_layout(bounded, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(bounded)

    scattered = ttnn.scatter(template, -1, index, zeros)
    ttnn.deallocate(index)
    selected = ttnn.slice(scattered, [0, 0, 0, 0], [batch, 1, rows, width])
    ttnn.deallocate(scattered)
    block = ttnn.add(causal_block, selected)
    ttnn.deallocate(selected)
    return block


def _compute_kernel_config(device, *, fp32: bool):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4 if fp32 else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


class TtCSAState:
    """Chunked-prefill state, owned by the caller and passed to ``TtCSA.forward``.

    The device tensors keep a FIXED shape for the whole prefill and only their contents advance --
    that is what lets one compiled program serve every chunk. The counters say how much is real;
    the attention mask -infs the rest."""

    def __init__(self, compressed_kv, sliding_carry, kv_state, score_state, index_kv_cache, max_seq_len):
        self.compressed_kv = compressed_kv  # [B, 1, compressed_capacity, head_dim]
        self.sliding_carry = sliding_carry  # [B, 1, sliding_window, head_dim]
        # The overlap the next chunk's first window needs: its predecessor window's Ca slice, in the
        # decode-compatible Blaze layout the compressor op emits and consumes.
        self.kv_state = kv_state  # [B, 1, CSA_STATE_ROWS, head_dim]
        self.score_state = score_state  # [B, 1, CSA_STATE_ROWS, head_dim]
        self.index_kv_cache = index_kv_cache  # block-cyclic indexer key cache, written in place
        self.max_seq_len = int(max_seq_len)
        self.entry_count = 0
        self.kv_actual = 0


class TtCSA(TtV4AttentionBase):
    """CSA block: query/kv stems + compressor + indexer + attention core + grouped output projection.

    Block I/O is ``[B, 1, S/sp, hidden/tp]``, so layers chain without a reshard."""

    def __init__(self, device, *, indexer_reference, indexer_settings, **kwargs):
        super().__init__(device, **kwargs)
        # The indexer is sized by max_seq_len and the slab width, which only alloc_state knows, so it is
        # built there; these hold what it needs until then.
        self._indexer_reference = indexer_reference
        self._indexer_settings = indexer_settings
        self._indexer = None
        # Everything below comes from alloc_state, which every caller has to run before forward.
        self._sel_template = None  # persistent -inf slab the picks are scattered into
        self._sel_zeros = None  # the zeros they scatter
        self._sel_dump = None

    @property
    def chunk_align(self) -> int:
        """One compression window per row of a tile. At ratio 4 that is 128 tokens, and it buys three
        things at once: the compressed append offset (entry_count = tokens/4) lands tile-aligned so
        fill_cache can take it directly, the carry slice start (real_len - 128) is tile-aligned, and a
        chunk always holds a whole sliding window for the next one to carry."""
        return self.compressor.compress_rate * ttnn.TILE_SIZE

    @property
    def indexer(self) -> TtCsaIndexer:
        assert self._indexer is not None, "call alloc_state before touching the indexer"
        return self._indexer

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None) -> TtCSAState:
        """Size the state once for the longest context this layer will serve, so its shape is fixed for
        every chunk. ``max_seq_len`` is the longest context to serve, ``chunk_tokens`` the slab width
        forward will be called with (defaults to one chunk).

        Every host tensor this layer will ever need is built here, which is what leaves forward with
        none. So the caller has to own the state: a prefill of one chunk allocates the same way a long
        one does."""
        rate = self.compressor.compress_rate
        chunk = chunk_tokens or max_seq_len
        align = csa_slab_align(rate, self.sp_factor, self.tp_factor)
        assert chunk % align == 0, (
            f"the slab is {chunk} wide, which is not a multiple of {align}; see csa_slab_align for what "
            f"asks for that. prepare_input rounds a raw prompt up to it."
        )

        width = chunk // rate  # entries one call emits, a whole number of tiles by chunk_align
        entries = -(-int(max_seq_len) // rate)
        # Writes are exact-width from a tile-aligned offset, so the last one can reach one slab past the
        # entries themselves.
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE + width

        self._build_carry_index(chunk)
        self._build_masks(chunk, capacity)
        self.compressor.alloc_tables(max_seq_len, chunk, capacity)

        # One rope table per state, so forward only gathers. This one has a row per TOKEN; the compressor
        # builds its own, with a row per ENTRY.
        self._slab_rope = self.ops.build_rope_table(rope_table_tokens(max_seq_len, chunk), 1)
        self._slab_index = self.ops.rope_index_base(chunk // self.sp_factor)

        index_kv_cache = self._build_indexer(max_seq_len, chunk)
        self._build_selection_consts(batch, chunk // self.sp_factor, capacity)

        kv_state, score_state = self.compressor.alloc_overlap_state(batch)
        return TtCSAState(
            compressed_kv=self.ops.from_torch(torch.zeros(batch, 1, capacity, self.head_dim)),
            sliding_carry=self.ops.from_torch(torch.zeros(batch, 1, self.sliding_window, self.head_dim)),
            kv_state=kv_state,
            score_state=score_state,
            index_kv_cache=index_kv_cache,
            max_seq_len=max_seq_len,
        )

    def _build_indexer(self, max_seq_len: int, chunk: int):
        """Build the indexer and its block-cyclic key cache.

        The context the indexer is told about is wider than the real one for two reasons: it writes
        whole padded slabs, so a run of ragged chunks lands its last write up to one slab past
        ``max_seq_len``, and each chip's share of the compressed rows has to be a whole number of tiles.
        Both only ever add rows past ``entry_count``, which the score op's own causal mask drops."""
        rate = self.compressor.compress_rate
        settings = self._indexer_settings
        index_align = self.chunk_align * self.sp_factor
        index_tokens = rope_table_tokens(max_seq_len, chunk)
        index_seq_len = -(-index_tokens // index_align) * index_align

        self._indexer = TtCsaIndexer.from_reference(
            self._indexer_reference,
            config=settings["config"],
            mesh_device=self.device,
            sp_axis=self.sp_axis,
            tp_axis=self.tp_axis,
            default_compute_kernel_config=settings["default_compute_kernel_config"],
            hifi4_fp32_compute_kernel_config=settings["hifi4_fp32_compute_kernel_config"],
            weight_cache_path=settings["weight_cache_path"],
            layer_idx=settings["layer_idx"],
            tt_ccl=self.tt_ccl,
            ccl_num_links=self.ccl_num_links,
            sp_ccl_topology=self.tp_ccl_topology,
            tp_ccl_topology=self.tp_ccl_topology,
            seq_len=index_seq_len,
            active_seq_len=chunk,
        )
        return init_kvpe_cache(
            kvpe_cache_head_dim=settings["config"].index_head_dim,
            mesh_device=self.device,
            seq_len=index_seq_len // rate,
            mesh_shape=list(self.device.shape),
            sp_axis=self.sp_axis,
            num_kvpe_cache_layers=1,
        )

    def _build_selection_consts(self, batch: int, rows: int, capacity: int):
        """The two persistent operands of the block-mask scatter.

        The scratch is the next power of two above the cache capacity, so masking a pick with
        ``width - 1`` maps every real entry to itself and the ``0xFFFFFFFF`` sentinel to the last
        column -- past the capacity, so slicing the mask down drops it."""
        k = self.indexer.index_topk_capacity
        width = 1 << capacity.bit_length()  # strictly greater than capacity, so the last column is spare
        self._sel_template = self.ops.from_torch(torch.full((batch, 1, rows, width), float("-inf")))
        self._sel_zeros = self.ops.from_torch(torch.zeros(batch, 1, rows, k))
        self._sel_dump = width - 1

    @classmethod
    def from_reference(
        cls,
        device,
        reference,
        config,
        *,
        layer_idx: int = 0,
        weight_cache_path=None,
        default_compute_kernel_config=None,
        hifi4_fp32_compute_kernel_config=None,
        **kwargs,
    ) -> "TtCSA":
        assert reference.compressor is not None and hasattr(
            reference.compressor, "indexer"
        ), "TtCSA needs a compressed_sparse_attention layer, whose compressor owns the indexer"
        # Forward the mesh/CCL config so the compressor rides the same SP/TP axes as the block.
        compressor_keys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype", "memory_config")
        compressor = TtCSACompressor.from_reference(
            device, reference.compressor, config, **{k: kwargs[k] for k in compressor_keys if k in kwargs}
        )
        return cls(
            device,
            compressor=compressor,
            indexer_reference=reference.compressor.indexer,
            indexer_settings={
                "config": config,
                "layer_idx": layer_idx,
                "weight_cache_path": weight_cache_path,
                "default_compute_kernel_config": default_compute_kernel_config
                or _compute_kernel_config(device, fp32=False),
                "hifi4_fp32_compute_kernel_config": hifi4_fp32_compute_kernel_config
                or _compute_kernel_config(device, fp32=True),
            },
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

    def _write_compressed(self, state, new_entries):
        """Append this call's entries to the cache at row ``state.entry_count``.

        ``chunk_align`` keeps that offset tile-aligned, which is all ``fill_cache_for_user_`` asks of it
        -- and since it keeps update_idx out of its program hash, the offset rides along as data and one
        program serves every chunk.

        The whole padded width is written, not just the real entries, so the width is the same for every
        chunk; the mask -infs everything past ``total_entries`` anyway."""
        width = new_entries.shape[2]
        assert state.entry_count % ttnn.TILE_SIZE == 0, (
            f"compressed append offset {state.entry_count} is not tile-aligned; a previous chunk was not "
            f"a multiple of {self.chunk_align} tokens"
        )
        assert state.entry_count + width <= state.compressed_kv.shape[2], (
            f"compressed cache full: writing rows [{state.entry_count}, {state.entry_count + width}) "
            f"exceeds capacity {state.compressed_kv.shape[2]}; allocate the state with a larger max_seq_len"
        )
        ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, new_entries, 0, update_idx=state.entry_count)

    def forward(
        self,
        hidden_states,
        seq_len_actual: int | None = None,
        *,
        state: TtCSAState,
    ):
        """One chunk: [B, 1, S_pad/sp, hidden/tp] in and out; the caller keeps the first S_real rows.

        ``seq_len_actual`` is the chunk's real pre-pad length. Where the chunk sits in the sequence comes
        from ``state``, which ``alloc_state`` builds and this advances in place -- a prefill of one chunk
        passes a state too, so there is no second path through here."""
        batch = hidden_states.shape[0]
        seq_local = hidden_states.shape[2]
        seq_pad_global = seq_local * self.sp_factor
        rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else seq_len_actual

        assert batch == 1, f"CSA prefill expects batch 1, got {batch}"
        assert real_len >= rate, (
            f"CSA prefill needs at least one full compression window: got seq_len {real_len} < " f"compress_rate {rate}"
        )
        assert state.kv_actual + real_len <= state.max_seq_len, (
            f"context longer than the state was allocated for: {state.kv_actual + real_len} tokens > "
            f"max_seq_len {state.max_seq_len}"
        )
        # A non-final chunk off the chunk_align grid strands its leftover tokens -- they never join a
        # compression window -- and takes the append offset off its tile boundary for every chunk after.
        # Checked on tokens, where a dropped partial window is still visible.
        assert state.kv_actual % self.chunk_align == 0, (
            f"cannot append after a chunk that left {state.kv_actual % self.chunk_align} tokens past a "
            f"{self.chunk_align}-token boundary; only the final chunk may be ragged"
        )

        n_new = real_len // rate
        total_entries = state.entry_count + n_new

        # One rotation for the whole padded slab, shared by both stems and the output un-rope.
        slab_index = self.ops.rope_index(self._slab_index, state.kv_actual)
        cos, sin = self.ops.rope_gather(self._slab_rope, slab_index)
        q_lora = self._q_lora(hidden_states)  # the reference's q_residual; the indexer scores from it
        q = self._q_heads(q_lora, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)

        new_entries, causal_block, kv_state, score_state = self.compressor(
            hidden_states,
            state.kv_state,
            state.score_state,
            seq_len_actual=real_len,
            first_window_position=state.entry_count * rate,
        )
        # Attention then reads the WHOLE cache every chunk, so its shape stays constant and the mask
        # -infs everything past total_entries.
        self._write_compressed(state, new_entries)

        # The indexer runs its own ratio-4 compressor at index_head_dim over the same windows and writes
        # its own block-cyclic key cache, so it only shares the hidden states and q_lora with us.
        top_k = self.indexer.forward(
            hidden_states,
            q_lora,
            seq_len=seq_local,
            start_pos=state.kv_actual,
            index_kv_cache=state.index_kv_cache,
            seq_len_actual=real_len,
        )
        mask_block = block_mask(
            causal_block,
            top_k,
            template=self._sel_template,
            zeros=self._sel_zeros,
            dump=self._sel_dump,
        )

        attn, next_carry = self._attention(
            q,
            sliding_kv,
            state.compressed_kv,
            mask_block,
            cos,
            sin,
            carry=state.sliding_carry,
            kv_actual=state.kv_actual,
            real_len=real_len,
        )

        state.kv_state = self.compressor.terminal_state(kv_state)
        state.score_state = self.compressor.terminal_state(score_state)
        state.entry_count = total_entries
        state.kv_actual += real_len
        state.sliding_carry = next_carry
        return self._o_proj(attn)
