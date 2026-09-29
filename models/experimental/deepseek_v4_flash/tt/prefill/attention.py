# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill attention for DeepSeek-V4-Flash: one block, many tokens per call.

:class:`DeepSeekV4PrefillAttention` is the multi-token counterpart of the decode
:class:`~..decode.attention.DeepSeekV4Attention`. It consumes a *chunk* of ``T`` hidden states
``[1, 1, T, D]`` and returns the attention output ``[1, 1, T, D]``, for all three layer types of
the model:

* ``sliding_attention``           -- shared-KV MQA over the last ``sliding_window`` tokens,
* ``heavily_compressed_attention`` (HCA, rate 128) -- the sliding window plus every closed
  128-token window pooled into one compressed KV entry,
* ``compressed_sparse_attention`` (CSA, rate 4)   -- the sliding window plus 4-token windows
  pooled with the overlapping Ca/Cb scheme.

The math is the reference ``DeepseekV4Attention`` (see ``modular_deepseek_v4.py``): low-rank Q
(``q_a`` -> norm -> ``q_b`` -> per-head unweighted RMSNorm), single shared K=V head
(``kv_proj`` -> norm), partial interleaved RoPE on the trailing ``qk_rope_head_dim`` channels,
a per-head attention sink folded into the softmax denominator, the conjugate RoPE undone on the
attention output, and the grouped low-rank output projection (``o_a`` per group, then ``o_b``).

Chunking. A prompt is fed as consecutive chunks through one :class:`PrefillAttentionState`. The
state carries exactly what a later chunk needs from earlier ones:

* ``kv_tail``        -- the last ``sliding_window`` roped K=V rows (the next chunk's window
  reaches back into this one),
* ``compressed_kv``  -- every compressed entry emitted so far (HCA / CSA),
* ``csa_prev_kv`` / ``csa_prev_gate`` -- the last closed window's Ca projections (CSA's
  overlap: window ``w`` pools window ``w-1``'s Ca slice with window ``w``'s Cb slice).

Supported shapes (v1):

* ``T`` and the chunk's start position must be multiples of ``ALIGNMENT`` (128 = the HCA
  compress rate = the SDPA chunk size). A ragged tail is left to the decode path.
* CSA runs the *dense* degeneracy of the lightning indexer: while the layer has at most
  ``index_topk`` compressed entries (a prompt of at most ``index_topk * 4 = 2048`` tokens) top-k
  selects every causally valid entry, so the indexer's output is the plain causal block mask and
  no indexer is needed. Past that the indexer + sparse SDPA are required and this module raises
  :class:`NotImplementedError` instead of returning something wrong.
* ``tp_size == 1`` only (one device / a 1x1 mesh).

Layouts. Compressed-KV state lives ROW_MAJOR so appending a chunk's entries is a plain concat for
any entry count (a TILE concat needs tile-aligned rows, and a layer has ``E = pos / rate``
entries). The K=V slab handed to SDPA is assembled in ROW_MAJOR too, padded to a multiple of the
SDPA chunk with zero rows (masked ``-inf``), and only then tilized.

Every projection is a plain ``ttnn.linear`` with HiFi4/fp32 accumulation and DRAM-interleaved
weights; this is the functional baseline the per-block optimization work starts from.
"""

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from ..common import _HIFI4, DeepSeekV4Module
from ..layers import Linear
from ..weight_cache import WeightCache, _as_cache, _load_weight, _materialize

SLIDING_ATTENTION = "sliding_attention"
COMPRESSED_SPARSE_ATTENTION = "compressed_sparse_attention"
HEAVILY_COMPRESSED_ATTENTION = "heavily_compressed_attention"

# Chunk lengths and start positions are multiples of this: it is the HCA compress rate and the
# SDPA q/k chunk, and a whole number of CSA windows and tiles.
ALIGNMENT = 128
_SDPA_CHUNK = 128

# Gate logit for a slot that must get softmax weight 0 (window 0 of the first CSA chunk has no
# previous window). Finite so ``exp(gate - max)`` underflows to exactly 0 instead of ``inf - inf``.
_NO_WINDOW_GATE = -1.0e30


@dataclass
class PrefillAttentionState:
    """What one layer's attention carries from one prefill chunk to the next.

    ``seq_len`` is the number of tokens consumed so far, i.e. the next chunk's start position.
    Every tensor is ``None`` until the chunk that produces it has run.
    """

    seq_len: int = 0
    # ROW_MAJOR bf16 ``[1, 1, sliding_window, Dh]``: the last window of roped K=V rows.
    kv_tail: Optional[ttnn.Tensor] = None
    # ROW_MAJOR bf16 ``[1, 1, E, Dh]``: all compressed entries so far (HCA / CSA only).
    compressed_kv: Optional[ttnn.Tensor] = None
    # ROW_MAJOR fp32 ``[1, 1, 1, rate * Dh]`` (CSA only): the last closed window's Ca kv and
    # (position-bias-added) gate, slot-major.
    csa_prev_kv: Optional[ttnn.Tensor] = None
    csa_prev_gate: Optional[ttnn.Tensor] = None

    @property
    def num_entries(self) -> int:
        """Compressed entries emitted so far."""
        return 0 if self.compressed_kv is None else self.compressed_kv.shape[2]


def _rot_transformation_mat() -> torch.Tensor:
    """The ``[1, 1, 32, 32]`` per-tile interleaved ``rotate_half`` ``rotary_embedding_llama`` wants.

    Maps every consecutive pair ``(x_{2p}, x_{2p+1}) -> (-x_{2p+1}, x_{2p})`` as a right-multiply,
    which is V4's interleaved RoPE.
    """
    dhead = ttnn.TILE_SIZE
    mat = torch.zeros(1, 1, dhead, dhead)
    mat[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    mat[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return mat


class DeepSeekV4PrefillAttention(DeepSeekV4Module):
    """ttnn prefill port of ``DeepseekV4Attention`` (see the module docstring).

    ``weights`` is the same HF-named dict :class:`~..decode.attention.DeepSeekV4Attention` takes,
    each value a torch tensor or a zero-arg thunk returning one, ``nn.Linear`` layout ``[out, in]``::

        q_a_proj.weight  q_a_norm.weight  q_b_proj.weight  kv_proj.weight  kv_norm.weight
        o_a_proj.weight  o_b_proj.weight  sinks
        compressor.kv_proj.weight  compressor.gate_proj.weight  compressor.kv_norm.weight
        compressor.position_bias                                  # HCA / CSA layers only

    (a CSA layer's ``compressor.indexer.*`` weights are not read: see the module docstring.)

    ``rope`` is the host rotary bundle the surrounding model already builds for decode::

        rope["main"]     = (cos_half, sin_half)   # sliding layers
        rope["compress"] = (cos_half, sin_half)   # CSA / HCA layers

    each ``[L, qk_rope_head_dim / 2]`` fp32 with one entry per interleaved pair and ``L`` at least
    the longest prompt. ``weight_dtype`` is the dtype of the big projections (``q_b``, ``o_a``,
    ``o_b``, ...); the compressor's kv / gate projections stay bf16 because their softmax gate is
    sensitive to weight error and they are small.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        rope: dict,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        tp_size: int = 1,
    ):
        """Upload the layer's weights and constants to ``device``.

        ``config`` is the HF ``DeepseekV4Config`` (or anything with the same attributes),
        ``layer_idx`` picks the layer type from ``config.layer_types``, ``cache`` is an optional
        :class:`~..weight_cache.WeightCache` namespace for the converted weights (this block's
        entries carry a ``.prefill`` suffix so they never collide with the decode layouts).
        """
        if tp_size != 1:
            raise NotImplementedError(f"prefill attention is single-device for now, got tp_size={tp_size}")
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.sliding_window = config.sliding_window
        self.eps = config.rms_norm_eps
        self.scaling = self.head_dim**-0.5
        if self.num_heads % self.o_groups:
            raise ValueError(f"num_attention_heads {self.num_heads} is not divisible by o_groups {self.o_groups}")
        if self.sliding_window % ALIGNMENT:
            raise ValueError(f"sliding_window {self.sliding_window} must be a multiple of {ALIGNMENT}")

        self.is_sliding = self.layer_type == SLIDING_ATTENTION
        self.is_csa = self.layer_type == COMPRESSED_SPARSE_ATTENTION
        self.rate = None if self.is_sliding else config.compress_rates[self.layer_type]
        if self.rate is not None and ALIGNMENT % self.rate:
            raise ValueError(f"compress rate {self.rate} must divide the chunk alignment {ALIGNMENT}")
        self.index_topk = config.index_topk
        # Sliding layers use the plain theta table, CSA / HCA the YaRN-scaled one they share
        # with their compressor.
        self.rope_kind = "main" if self.is_sliding else "compress"
        self.rope = rope

        cache = _as_cache(cache)

        def linear(name: str, dtype: ttnn.DataType = weight_dtype) -> Linear:
            return Linear(weights[f"{name}.weight"], device, cache.file(f"{name}.prefill"), dtype=dtype)

        def host(name: str) -> torch.Tensor:
            w = weights[name]
            return (w() if callable(w) else w).detach().float()

        def norm_gamma(name: str) -> ttnn.Tensor:
            return self._to_device(host(name).reshape(1, 1, 1, -1))

        self.q_a_proj = linear("q_a_proj")
        self.q_b_proj = linear("q_b_proj")
        self.kv_proj = linear("kv_proj")
        self.o_b_proj = linear("o_b_proj")
        self.q_a_norm_weight = norm_gamma("q_a_norm.weight")
        self.kv_norm_weight = norm_gamma("kv_norm.weight")
        # The per-head Q norm has no weight in the reference; a ones gamma is the unweighted norm.
        self.q_b_norm_weight = self._to_device(torch.ones(1, 1, 1, self.head_dim))

        # Grouped output projection: block-diagonal over o_groups, run as one batched matmul.
        # The HF weight is [g * o_lora, in_per_group]; the batched matmul wants [1, g, K, N].
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        o_a_file = cache.file("o_a_proj.prefill")
        o_a = _materialize(weights["o_a_proj.weight"], o_a_file, weight_dtype)
        if o_a is not None:
            o_a = o_a.detach().reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).unsqueeze(0)
            o_a = o_a.contiguous()
        self.o_a_proj = _load_weight(o_a, device, cache_file_name=o_a_file, dtype=weight_dtype)

        # The sink joins the softmax denominator un-scaled, but SDPA multiplies ``scale`` into the
        # QK logits and the sink alike, so pre-divide by the scale to cancel it.
        sinks = host("sinks").reshape(1, self.num_heads, 1, 1) / self.scaling
        self.sinks = self._to_device(sinks)

        self.trans_mat = self._to_device(_rot_transformation_mat())

        if not self.is_sliding:
            # The compressor's projection width: Dh for HCA, 2*Dh (the Ca | Cb pair) for CSA.
            compressor_width = self.head_dim * (2 if self.is_csa else 1)
            self.c_kv_proj = linear("compressor.kv_proj", ttnn.bfloat16)
            self.c_gate_proj = linear("compressor.gate_proj", ttnn.bfloat16)
            self.c_norm_weight = norm_gamma("compressor.kv_norm.weight")
            bias = host("compressor.position_bias")
            assert tuple(bias.shape) == (
                self.rate,
                compressor_width,
            ), f"compressor.position_bias is {tuple(bias.shape)}, expected {(self.rate, compressor_width)}"
            if self.is_csa:
                # fp32 [1,1,1,2*Dh] per slot: added to the fp32 gate slots of the pooling below.
                self.c_bias_slots = [
                    self._to_device(bias[s].reshape(1, 1, 1, -1), dtype=ttnn.float32) for s in range(self.rate)
                ]
            else:
                # bf16 [1,1,rate,Dh]: added to the window-reshaped gate [1, n_win, rate, Dh].
                self.c_bias = self._to_device(bias.reshape(1, 1, self.rate, self.head_dim))

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    def _to_device(self, t: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
        """A host tensor as a TILE, DRAM-interleaved ``dtype`` tensor on the device."""
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _zeros_rm(self, rows: int, width: Optional[int] = None) -> ttnn.Tensor:
        """A ROW_MAJOR bf16 ``[1, 1, rows, width]`` zero tensor (``width`` defaults to ``Dh``)."""
        return ttnn.from_torch(
            torch.zeros(1, 1, rows, width or self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def new_state(self) -> PrefillAttentionState:
        """An empty state: the start of a prompt."""
        return PrefillAttentionState()

    def _rope_tables(self, positions: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``(cos, sin)`` ``[1, 1, n, rope_dim]`` bf16 tables for the absolute ``positions`` ``[n]``.

        The bundle holds one entry per interleaved pair; ``repeat_interleave(2)`` expands it to
        the pair layout ``rotary_embedding_llama`` reads (``cos[2p] == cos[2p+1]``).
        """
        cos_half, sin_half = self.rope[self.rope_kind]
        if int(positions.max()) >= cos_half.shape[0]:
            raise ValueError(
                f"rope table covers {cos_half.shape[0]} positions but position {int(positions.max())} is needed"
            )
        n = positions.shape[0]

        def table(half: torch.Tensor) -> ttnn.Tensor:
            return self._to_device(half[positions].float().repeat_interleave(2, dim=-1).reshape(1, 1, n, -1))

        return table(cos_half), table(sin_half)

    def _rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Partial RoPE on the trailing ``rope_dim`` channels of each head of ``x`` ``[1, H, n, Dh]``.

        The leading nope channels pass through untouched. ``cos`` / ``sin`` are ``[1, 1, n, rope_dim]``
        (broadcast over heads); pass ``-sin`` to rotate the other way.
        """
        heads, rows = x.shape[1], x.shape[2]
        nope_dim = self.head_dim - self.rope_dim
        nope = ttnn.slice(x, [0, 0, 0, 0], [1, heads, rows, nope_dim])
        rope = ttnn.slice(x, [0, 0, 0, nope_dim], [1, heads, rows, self.head_dim])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    # ------------------------------------------------------------------ #
    # Q / K=V stems
    # ------------------------------------------------------------------ #
    def _q_stem(self, hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, T, D]`` -> roped, per-head-normed queries ``[1, H, T, Dh]``."""
        q = self.q_a_proj(hidden)
        q = ttnn.rms_norm(q, weight=self.q_a_norm_weight, epsilon=self.eps)
        q = self.q_b_proj(q)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q,
            num_heads=self.num_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.rms_norm(q, weight=self.q_b_norm_weight, epsilon=self.eps)
        return self._rope(q, cos, sin)

    def _kv_stem(self, hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, T, D]`` -> the roped single K=V head ``[1, 1, T, Dh]``."""
        kv = self.kv_proj(hidden)
        kv = ttnn.rms_norm(kv, weight=self.kv_norm_weight, epsilon=self.eps)
        return self._rope(kv, cos, sin)

    # ------------------------------------------------------------------ #
    # compressors
    # ------------------------------------------------------------------ #
    def _finish_entries(self, pooled: ttnn.Tensor, first_position: int) -> ttnn.Tensor:
        """Norm + RoPE the pooled ``[1, 1, n_win, Dh]`` entries, returned ROW_MAJOR.

        Entry ``w`` is RoPE'd at the absolute position ``first_position + w * rate``: the position
        of its window's first token, so entries stay valid across chunks.
        """
        n_win = pooled.shape[2]
        pooled = ttnn.rms_norm(pooled, weight=self.c_norm_weight, epsilon=self.eps)
        positions = first_position + torch.arange(n_win) * self.rate
        cos, sin = self._rope_tables(positions)
        pooled = self._rope(pooled, cos, sin)
        return ttnn.to_layout(pooled, ttnn.ROW_MAJOR_LAYOUT)

    def _pool_hca(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Non-overlapping softmax pooling of every closed window: ``[1, 1, T, D]`` -> ``[1, 1, T/rate, Dh]``."""
        n_win = hidden.shape[2] // self.rate
        dh = self.head_dim
        kv = self.c_kv_proj(hidden)
        gate = self.c_gate_proj(hidden)

        gate = ttnn.reshape(gate, [1, n_win, self.rate, dh])
        gate = ttnn.add(gate, self.c_bias)
        weights = ttnn.softmax(gate, dim=2, numeric_stable=True)
        kv = ttnn.reshape(kv, [1, n_win, self.rate, dh])
        pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)
        return ttnn.reshape(pooled, [1, 1, n_win, dh])

    def _window_view(self, x: ttnn.Tensor, n_win: int, width: int) -> ttnn.Tensor:
        """``[1, 1, T, w]`` -> fp32 TILE ``[1, 1, T/rate, rate * w]``: each window's tokens side by side.

        A ROW_MAJOR reshape merges consecutive rows into one wide row, so token ``s`` of window
        ``w`` lands in columns ``[s*w, (s+1)*w)`` of row ``w`` (tile-aligned column slices then
        pull a slot out without touching the 4-row window axis, which is not tile-aligned).
        """
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [1, 1, n_win, width])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return ttnn.typecast(x, ttnn.float32)

    def _shift_windows(self, cur: ttnn.Tensor, prev_row: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Shift per-window rows down by one: row ``w`` becomes window ``w-1``'s, row 0 is ``prev_row``.

        ``cur`` is TILE ``[1, 1, n_win, W]`` and ``prev_row`` ROW_MAJOR ``[1, 1, 1, W]``. Returns the
        shifted tensor (TILE) and ``cur``'s last row (ROW_MAJOR), the next chunk's ``prev_row``.
        """
        n_win, width = cur.shape[2], cur.shape[3]
        cur_rm = ttnn.to_layout(cur, ttnn.ROW_MAJOR_LAYOUT)
        head = ttnn.slice(cur_rm, [0, 0, 0, 0], [1, 1, n_win - 1, width])
        shifted = ttnn.to_layout(ttnn.concat([prev_row, head], dim=2), ttnn.TILE_LAYOUT)
        last = ttnn.slice(cur_rm, [0, 0, n_win - 1, 0], [1, 1, n_win, width])
        return shifted, last

    def _initial_csa_overlap(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The "previous window" of a prompt's first CSA window: zero kv and a gate that weighs 0."""
        width = self.rate * self.head_dim
        kv = ttnn.from_torch(
            torch.zeros(1, 1, 1, width), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        gate = ttnn.from_torch(
            torch.full((1, 1, 1, width), _NO_WINDOW_GATE),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        return kv, gate

    def _pool_csa(self, hidden: ttnn.Tensor, state: PrefillAttentionState) -> ttnn.Tensor:
        """CSA's overlapping pooling: ``[1, 1, T, D]`` -> ``[1, 1, T/rate, Dh]``; updates the overlap state.

        Every token projects to ``2*Dh`` = ``Ca | Cb``. Entry ``w`` is the softmax-gated combination
        over ``2*rate`` slots: window ``w-1``'s Ca slice (``rate`` slots) then window ``w``'s Cb
        slice (``rate`` slots), with ``position_bias[s]`` added to slot ``s``'s gate before the
        split. The 8-way softmax is written out elementwise in fp32 (max, exp, sum, divide) over
        ``[n_win, Dh]`` slot tensors instead of reshaping to a 4-row window axis.
        """
        rate, dh = self.rate, self.head_dim
        wd = 2 * dh
        n_win = hidden.shape[2] // rate

        kv = self._window_view(self.c_kv_proj(hidden), n_win, rate * wd)
        gate = self._window_view(self.c_gate_proj(hidden), n_win, rate * wd)

        def slot(x: ttnn.Tensor, s: int, lo: int, hi: int) -> ttnn.Tensor:
            return ttnn.slice(x, [0, 0, 0, s * wd + lo], [1, 1, n_win, s * wd + hi])

        # Position bias joins the gate before the Ca / Cb split (and before the overlap is saved).
        gate_slots = [ttnn.add(slot(gate, s, 0, wd), self.c_bias_slots[s]) for s in range(rate)]

        # This chunk's Ca slice, all slots side by side [1, 1, n_win, rate * Dh].
        ca_kv = ttnn.concat([slot(kv, s, 0, dh) for s in range(rate)], dim=-1)
        ca_gate = ttnn.concat([ttnn.slice(gate_slots[s], [0, 0, 0, 0], [1, 1, n_win, dh]) for s in range(rate)], dim=-1)

        if state.csa_prev_kv is None:
            state.csa_prev_kv, state.csa_prev_gate = self._initial_csa_overlap()
        prev_kv, state.csa_prev_kv = self._shift_windows(ca_kv, state.csa_prev_kv)
        prev_gate, state.csa_prev_gate = self._shift_windows(ca_gate, state.csa_prev_gate)

        logits, values = [], []
        for j in range(rate):  # window w-1's Ca slots
            logits.append(ttnn.slice(prev_gate, [0, 0, 0, j * dh], [1, 1, n_win, (j + 1) * dh]))
            values.append(ttnn.slice(prev_kv, [0, 0, 0, j * dh], [1, 1, n_win, (j + 1) * dh]))
        for j in range(rate):  # window w's Cb slots
            logits.append(ttnn.slice(gate_slots[j], [0, 0, 0, dh], [1, 1, n_win, wd]))
            values.append(slot(kv, j, dh, wd))

        peak = logits[0]
        for logit in logits[1:]:
            peak = ttnn.maximum(peak, logit)
        numerator = denominator = None
        for logit, value in zip(logits, values):
            weight = ttnn.exp(ttnn.subtract(logit, peak))
            term = ttnn.multiply(weight, value)
            numerator = term if numerator is None else ttnn.add(numerator, term)
            denominator = weight if denominator is None else ttnn.add(denominator, weight)
        pooled = ttnn.multiply(numerator, ttnn.reciprocal(denominator))
        return ttnn.typecast(pooled, ttnn.bfloat16)

    def _compress(self, hidden: ttnn.Tensor, state: PrefillAttentionState) -> Optional[ttnn.Tensor]:
        """Pool this chunk's windows, append them to the state, and return every entry (ROW_MAJOR).

        ``None`` for a sliding layer. The chunk is window-aligned, so no partially filled window
        ever crosses a chunk boundary.
        """
        if self.is_sliding:
            return None
        pooled = self._pool_csa(hidden, state) if self.is_csa else self._pool_hca(hidden)
        entries = self._finish_entries(pooled, state.seq_len)
        state.compressed_kv = (
            entries if state.compressed_kv is None else ttnn.concat([state.compressed_kv, entries], dim=2)
        )
        return state.compressed_kv

    # ------------------------------------------------------------------ #
    # attention
    # ------------------------------------------------------------------ #
    def _mask(self, num_tokens: int, start: int, num_entries: int, padded_keys: int) -> ttnn.Tensor:
        """Additive mask ``[1, 1, T, padded_keys]`` over the key layout ``[tail | chunk | entries | pad]``.

        * Window part (``sliding_window + T`` columns; column ``k`` is the token at absolute position
          ``start - sliding_window + k``): query ``i`` (absolute ``start + i``) sees the ``sliding_window``
          tokens ending at itself, i.e. ``i < k <= i + sliding_window``, and never a position before 0
          (the zero rows the first chunk's empty tail stands in for).
        * Entry part: query ``i`` sees entry ``w`` iff ``w < (start + i + 1) // rate`` (the entry's
          window is fully in the query's past). For CSA this is exactly the indexer's output while
          there are no more than ``index_topk`` entries.
        * Padding columns are masked.
        """
        sw = self.sliding_window
        query = torch.arange(num_tokens).view(-1, 1)
        key = torch.arange(sw + num_tokens).view(1, -1)
        window = (key <= query + sw) & (key > query) & (key + (start - sw) >= 0)
        mask = torch.full((num_tokens, padded_keys), float("-inf"))
        mask[:, : sw + num_tokens].masked_fill_(window, 0.0)
        if num_entries:
            entry = torch.arange(num_entries).view(1, -1)
            visible = ((start + torch.arange(num_tokens) + 1) // self.rate).view(-1, 1)
            mask[:, sw + num_tokens : sw + num_tokens + num_entries].masked_fill_(entry < visible, 0.0)
        return self._to_device(mask.reshape(1, 1, num_tokens, padded_keys))

    def _attend(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        entries: Optional[ttnn.Tensor],
        state: PrefillAttentionState,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """SDPA of ``q`` over ``[tail | chunk | entries | pad]``, then undo V's RoPE. Rolls the KV tail."""
        num_tokens = q.shape[2]
        sw = self.sliding_window

        chunk_rm = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
        tail = state.kv_tail if state.kv_tail is not None else self._zeros_rm(sw)
        window = ttnn.concat([tail, chunk_rm], dim=2)  # [1, 1, sw + T, Dh]
        # The next chunk's window is this one's last ``sw`` rows.
        state.kv_tail = ttnn.slice(window, [0, 0, num_tokens, 0], [1, 1, num_tokens + sw, self.head_dim])

        parts = [window]
        num_entries = 0
        if entries is not None:
            parts.append(entries)
            num_entries = entries.shape[2]
        keys = sw + num_tokens + num_entries
        padded_keys = -(-keys // _SDPA_CHUNK) * _SDPA_CHUNK
        if padded_keys > keys:
            parts.append(self._zeros_rm(padded_keys - keys))
        kv_all = ttnn.concat(parts, dim=2) if len(parts) > 1 else window
        kv_all = ttnn.to_layout(kv_all, ttnn.TILE_LAYOUT)

        mask = self._mask(num_tokens, state.seq_len, num_entries, padded_keys)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            kv_all,
            kv_all,
            attn_mask=mask,
            is_causal=False,
            scale=self.scaling,
            attention_sink=self.sinks,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=_SDPA_CHUNK,
                k_chunk_size=_SDPA_CHUNK,
                exp_approx_mode=False,
            ),
        )
        # K = V, so V carried K's RoPE on its trailing slice: rotate it back by the same angle
        # the other way (sin -> -sin) at the query's position.
        return self._rope(attn, cos, ttnn.neg(sin))

    def _o_proj(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """Grouped output projection: ``[1, H, T, Dh]`` -> ``[1, 1, T, D]``.

        Consecutive heads form a group (``H / g`` heads = ``in_per_group`` channels): ``o_a`` maps each
        group to ``o_lora_rank`` independently (one batched matmul), the results are laid side by
        side and ``o_b`` mixes them to the hidden size.
        """
        num_tokens = attn.shape[2]
        g = self.o_groups
        heads_per_group = self.num_heads // g
        in_per_group = heads_per_group * self.head_dim

        x = ttnn.reshape(attn, [g, heads_per_group, num_tokens, self.head_dim])
        x = ttnn.experimental.nlp_concat_heads(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [g, 1, T, in_per_group]
        x = ttnn.reshape(x, [1, g, num_tokens, in_per_group])

        grouped = ttnn.linear(x, self.o_a_proj, compute_kernel_config=_HIFI4)  # [1, g, T, o_lora_rank]
        grouped = ttnn.concat(
            [ttnn.slice(grouped, [0, gi, 0, 0], [1, gi + 1, num_tokens, self.o_lora_rank]) for gi in range(g)],
            dim=-1,
        )  # [1, 1, T, g * o_lora_rank]
        return self.o_b_proj(grouped)

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #
    def _check_chunk(self, hidden: ttnn.Tensor, state: PrefillAttentionState) -> None:
        """Reject a chunk this version cannot process, before anything is computed."""
        shape = tuple(hidden.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 1 or shape[3] != self.config.hidden_size:
            raise ValueError(f"expected hidden states [1, 1, T, {self.config.hidden_size}], got {shape}")
        num_tokens = shape[2]
        if num_tokens == 0 or num_tokens % ALIGNMENT:
            raise ValueError(f"chunk length {num_tokens} must be a positive multiple of {ALIGNMENT}")
        if state.seq_len % ALIGNMENT:
            raise ValueError(f"chunk start {state.seq_len} must be a multiple of {ALIGNMENT}")
        if self.is_csa:
            entries = (state.seq_len + num_tokens) // self.rate
            if entries > self.index_topk:
                raise NotImplementedError(
                    f"CSA layer {self.layer_idx} would hold {entries} compressed entries after this chunk, "
                    f"more than index_topk={self.index_topk}: top-k selection is no longer the identity, and "
                    "the lightning indexer + sparse SDPA are not implemented in prefill yet "
                    f"(prompts up to {self.index_topk * self.rate} tokens are supported)"
                )

    def forward(self, hidden: ttnn.Tensor, state: Optional[PrefillAttentionState] = None) -> ttnn.Tensor:
        """Attention over one prompt chunk.

        ``hidden`` is ``[1, 1, T, D]`` bf16 TILE on the device; ``state`` is this layer's
        :class:`PrefillAttentionState` (``None`` runs a whole prompt from scratch and discards the
        state). The state is advanced in place. Returns ``[1, 1, T, D]`` bf16 TILE.
        """
        if state is None:
            state = self.new_state()
        self._check_chunk(hidden, state)
        num_tokens = hidden.shape[2]
        start = state.seq_len

        cos, sin = self._rope_tables(start + torch.arange(num_tokens))
        q = self._q_stem(hidden, cos, sin)
        kv = self._kv_stem(hidden, cos, sin)
        entries = self._compress(hidden, state)
        attn = self._attend(q, kv, entries, state, cos, sin)
        out = self._o_proj(attn)

        state.seq_len += num_tokens
        return out
