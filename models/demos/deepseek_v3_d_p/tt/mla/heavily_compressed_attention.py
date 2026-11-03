# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Heavily Compressed Attention (TTNN prefill).
Mirrors ``DeepseekV4Attention`` in ``reference/deepseek_v4/modeling_deepseek_v4.py``.

The attention core it runs on lives in ``v4_attention_base.py``; what is here is HCA's own compressed-cache
append, which has to place entries at offsets that are not tile-aligned."""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtHCACompressor, rope_table_tokens
from models.demos.deepseek_v3_d_p.tt.mla.v4_attention_base import TtV4AttentionBase


def _cache_write_rows(chunk_entries: int) -> int:
    """How many cache rows one write covers: this chunk's new entries plus the 0..31 rows already sitting in
    the last tile. The cache is written a whole tile at a time, so this rounds up to whole tiles -- 64 rows
    for a 4096-token chunk, 96 for 5120."""
    return -(-(ttnn.TILE_SIZE - 1 + chunk_entries) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


class TtHCAState:
    """Chunked-prefill state, owned by the caller and passed to ``TtHCA.forward``.

    The device tensors keep a FIXED shape for the whole prefill and only their contents advance --
    that is what lets one compiled program serve every chunk. The counters say how much is real;
    the attention mask -infs the rest."""

    def __init__(self, compressed_kv, sliding_carry, tail, max_seq_len):
        self.compressed_kv = compressed_kv  # [B, 1, compressed_capacity, head_dim]
        self.sliding_carry = sliding_carry  # [B, 1, sliding_window, head_dim]
        # The cache's last tile is usually only partly filled. Its entries sit here, right-aligned, so the
        # next write can place them and the new entries with a single shift.
        self.tail = tail  # [B, 1, TILE_SIZE, head_dim]
        self.max_seq_len = int(max_seq_len)
        self.entry_count = 0
        self.kv_actual = 0


class TtHCA(TtV4AttentionBase):
    """HCA block: query/kv stems + compressor + attention core + grouped output projection.

    Block I/O is ``[B, 1, S/sp, hidden/tp]``, so layers chain without a reshard."""

    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        # Everything below comes from alloc_state, which every caller has to run before forward.
        self._shift_cache = {}
        self._take_cache = {}

    @property
    def chunk_align(self) -> int:
        """One compression window. The append offset is then a whole number of windows in, which the
        tail-tile write handles at any alignment, and the carry start follows because a window is a
        multiple of TILE_SIZE."""
        return self.compressor.compress_rate

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None) -> TtHCAState:
        """Size the state once for the longest context this layer will serve, so its shape is fixed for
        every chunk. ``max_seq_len`` is the longest context to serve, ``chunk_tokens`` the slab width
        forward will be called with (defaults to one chunk).
        Contents start zeroed and nothing is read before it is written -- the mask -infs everything past
        ``entry_count`` / ``kv_actual``, including chunk 0's empty carry.

        Every host tensor this layer will ever need is built here, which is what leaves forward with none.
        So the caller has to own the state: a prefill of one chunk allocates the same way a long one
        does."""
        entries = -(-int(max_seq_len) // self.compressor.compress_rate)
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE  # cache writes land on tile boundaries
        # A write always rewrites whole tiles, so the last one can reach past the entries themselves.
        # ``chunk_tokens`` sizes that headroom.
        chunk = chunk_tokens or max_seq_len
        align = self.compressor.compress_rate * self.sp_factor
        assert chunk % align == 0, (
            f"the slab is {chunk} wide, which is not a multiple of compress_rate * sp_factor "
            f"({self.compressor.compress_rate} * {self.sp_factor} = {align}); every chip's share of it has "
            f"to end on a compression-window boundary. prepare_input rounds a raw prompt up to that."
        )
        width = -(-int(chunk) // self.compressor.compress_rate)
        capacity += _cache_write_rows(width)
        # Every one-hot the write can ever need: with chunks of differing real length r_e reaches every
        # value, so the whole TILE_SIZE set is built.
        self._build_tail_tile_matrices(width)
        self._build_carry_index(chunk)
        self._build_masks(chunk, capacity)
        self.compressor.alloc_tables(max_seq_len, chunk, capacity)

        # One rope table per state, so forward only gathers. This one has a row per TOKEN; the compressor
        # builds its own, with a row per ENTRY.
        self._slab_rope = self.ops.build_rope_table(rope_table_tokens(max_seq_len, chunk), 1)
        self._slab_index = self.ops.rope_index_base(chunk // self.sp_factor)
        return TtHCAState(
            compressed_kv=self.ops.from_torch(torch.zeros(batch, 1, capacity, self.head_dim)),
            sliding_carry=self.ops.from_torch(torch.zeros(batch, 1, self.sliding_window, self.head_dim)),
            tail=self.ops.from_torch(torch.zeros(batch, 1, ttnn.TILE_SIZE, self.head_dim)),
            max_seq_len=max_seq_len,
        )

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCA":
        # Forward the mesh/CCL config so the compressor rides the same SP/TP axes as the block.
        compressor_keys = ("sp_axis", "tp_axis", "topology", "dtype", "weights_dtype", "memory_config")
        compressor = TtHCACompressor.from_reference(
            device, reference.compressor, config, **{k: kwargs[k] for k in compressor_keys if k in kwargs}
        )
        return cls(
            device,
            compressor=compressor,
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

    def _write_compressed(self, state, new_entries, n_new):
        """Append this call's entries to the cache at row ``state.entry_count``.

        One entry is one row, so the offset advances by ``chunk/compress_rate`` per chunk -- a multiple of
        TILE_SIZE only for chunks of 4096 tokens. ``fill_cache_for_user_`` needs that alignment
        which a 5120-token chunk does not give it, so the write goes through ``_write_tail_tile`` instead.

        The whole padded width is written, not just the real entries, so the width is the same for every
        chunk; the mask -infs everything past ``total_entries`` anyway."""
        width = new_entries.shape[2]
        tile_start = (state.entry_count // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        write_end = tile_start + _cache_write_rows(width)
        assert write_end <= state.compressed_kv.shape[2], (
            f"compressed cache full: writing rows [{tile_start}, {write_end}) exceeds capacity "
            f"{state.compressed_kv.shape[2]}; allocate the state with a larger max_seq_len"
        )
        self._write_tail_tile(state, new_entries, width, n_new)

    def _write_tail_tile(self, state, new_entries, width, n_new):
        """fill_cache leaves update_idx out of its program but needs it tile-aligned, so this writes at the
        tile boundary below entry_count and supplies that whole tile -- the entries already in it come from
        ``state.tail``.

        The tail being right-aligned is what makes one shift enough: it and ``new_entries`` then sit in
        ``src`` under the same rule, so placing them is a single shift by ``TILE_SIZE - r_e``.

        The shift is a matmul against a one-hot matrix and not a slice: a slice's offset becomes part of
        its program, a matmul carries it as data."""
        tile = ttnn.TILE_SIZE
        f, r_e = divmod(state.entry_count, tile)
        shift, take = self._tail_tile_matrices(r_e, width, n_new)

        src = ttnn.concat([state.tail, new_entries], dim=2)  # [B, 1, tile + width, head_dim]
        merged = ttnn.matmul(shift, src, memory_config=self.memory_config)
        ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, merged, 0, update_idx=f * tile)
        state.tail = ttnn.matmul(take, merged, memory_config=self.memory_config)

    def _build_tail_tile_matrices(self, width):
        """Every one-hot pair the write can need, for one slab width. Called from alloc_state and never
        from forward, because a pair costs ~0.9 ms of host time to build. The whole set is 768 KB a chip."""
        tile, buf = ttnn.TILE_SIZE, _cache_write_rows(width)
        for r_e in range(tile):
            # merged row i takes src row i + (tile - r_e); rows past the entries stay zero, so nothing
            # reads past src.
            rows = torch.arange(r_e + width)
            shift = torch.zeros(1, 1, buf, tile + width)
            shift[0, 0, rows, rows + (tile - r_e)] = 1.0
            self._shift_cache[(r_e, width)] = self.ops.from_torch(shift)

        # The take matrix is keyed on r_e + n_new and not r_e + width: the whole padded width is written,
        # but entry_count only advances by the real entries. Rows before the tile's first live entry stay
        # zero, so the next chunk's shift skips them by construction instead of having to mask them.
        for s in range(tile + width):
            rows = torch.arange(tile - s % tile, tile)
            take = torch.zeros(1, 1, tile, buf)
            take[0, 0, rows, rows + s - tile] = 1.0
            self._take_cache[s] = self.ops.from_torch(take)

    def _tail_tile_matrices(self, r_e, width, n_new):
        """Lookup only. A miss means alloc_state was given a different chunk width, and building one here
        would put host work back into forward."""
        shift = self._shift_cache.get((r_e, width))
        take = self._take_cache.get(r_e + n_new)
        assert shift is not None and take is not None, (
            f"no tail-tile one-hot for (r_e={r_e}, width={width}, n_new={n_new}); alloc_state built widths "
            f"{sorted({w for _, w in self._shift_cache})} -- pass chunk_tokens matching the slab width"
        )
        return shift, take

    def forward(
        self,
        hidden_states,
        seq_len_actual: int | None = None,
        *,
        state: TtHCAState,
    ):
        """One chunk: [B, 1, S_pad/sp, hidden/tp] in and out; the caller keeps the first S_real rows.

        ``seq_len_actual`` is the chunk's real pre-pad length. Where the chunk sits in the sequence comes
        from ``state``, which ``alloc_state`` builds and this advances in place -- a prefill of one chunk
        passes a state too, so there is no second path through here."""
        batch = hidden_states.shape[0]
        seq_pad_global = hidden_states.shape[2] * self.sp_factor
        compress_rate = self.compressor.compress_rate
        real_len = seq_pad_global if seq_len_actual is None else seq_len_actual

        assert batch == 1, f"HCA prefill expects batch 1, got {batch}"

        assert real_len >= compress_rate, (
            f"HCA prefill needs at least one full compression window: got seq_len {real_len} < "
            f"compress_rate {compress_rate}"
        )

        n_new = real_len // compress_rate
        total_entries = state.entry_count + n_new
        assert state.kv_actual + real_len <= state.max_seq_len, (
            f"context longer than the state was allocated for: {state.kv_actual + real_len} tokens > "
            f"max_seq_len {state.max_seq_len}"
        )
        # Checked on tokens and not on entry_count, where a dropped partial window is invisible: 4097
        # tokens still gives 32 entries, and the next chunk would start at the wrong position.
        #
        # A non-final chunk ending mid-window would strand those leftover tokens -- they never join a
        # compression window -- and push the next chunk off the 128-token grid the compressor assumes.
        assert state.kv_actual % compress_rate == 0, (
            f"cannot append after a chunk with {state.kv_actual % compress_rate} leftover tokens; only "
            f"the final chunk may be ragged, non-final chunks must be a multiple of {compress_rate}"
        )

        # One rotation for the whole padded slab, shared by both stems and the output un-rope.
        slab_index = self.ops.rope_index(self._slab_index, state.kv_actual)
        cos, sin = self.ops.rope_gather(self._slab_rope, slab_index)
        q = self._q_stem(hidden_states, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)
        new_entries, mask_block = self.compressor(
            hidden_states,
            seq_len_actual=seq_len_actual,
            first_window_position=state.entry_count * compress_rate,
        )
        # Attention then reads the WHOLE cache every chunk, so its shape stays constant and the mask
        # -infs everything past total_entries.
        self._write_compressed(state, new_entries, n_new)

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

        state.entry_count = total_entries
        state.kv_actual += real_len
        state.sliding_carry = next_carry
        return self._o_proj(attn)
