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
  no indexer is needed. Past that the lightning indexer is required: with ``lightning_indexer=True`` the
  indexer scores every (query, entry) pair (``indexer_score_dsa``), the top ``index_topk`` entries per query
  are kept by threshold and folded into the additive SDPA mask (dense masked SDPA, no sparse kernel).
  Eager :meth:`forward` and traced :meth:`forward_static` both do this; otherwise this module raises
  :class:`NotImplementedError` (or, with ``dense_csa``, attends to every entry).
* ``tp_size`` is 1 (one device / a 1x1 mesh) or the width of a 1xTP mesh (TP=4 in the model).

Tensor parallelism (``tp_size > 1``) follows the decode block's split. The hidden chunk arrives
replicated on every rank, and everything that does not scale with the head count stays replicated:
``q_a`` (+ its norm), ``kv_proj`` (+ its norm), the compressor and therefore the whole KV state
(``kv_tail`` / ``compressed_kv`` / the CSA overlap are identical on every rank). The head-dependent
work is split over ranks:

* ``q_b`` is column-parallel: each rank owns ``H / TP`` contiguous query heads,
* the per-head attention sinks are sharded the same way, so each rank runs SDPA for its heads
  against the shared K=V with no collective,
* ``o_a`` is block-diagonal over ``o_groups``, so each rank owns ``o_groups / TP`` complete groups
  (its heads are exactly those groups' inputs) and runs them locally,
* ``o_b`` is row-parallel over the concatenated group outputs: each rank produces a full-``D``
  partial, and one all-reduce over the TP axis returns the replicated ``[1, 1, T, D]`` output.

Layouts. Compressed-KV state lives ROW_MAJOR so appending a chunk's entries is a plain concat for
any entry count (a TILE concat needs tile-aligned rows, and a layer has ``E = pos / rate``
entries). The K=V slab handed to SDPA is assembled in ROW_MAJOR too, padded to a multiple of the
SDPA chunk with zero rows (masked ``-inf``), and only then tilized.

Every projection is a plain ``ttnn.linear`` with HiFi4/fp32 accumulation and DRAM-interleaved
weights; this is the functional baseline the per-block optimization work starts from.

Mask-free attention (``maskless=True``, see ``MASKLESS_PREFILL.md``). Sliding layers build no additive mask: they
run the dense op's own ``is_causal`` + ``sliding_window_size`` over ``[real tail | chunk]``, the query front-padded
so ``Sq == Sk``. CSA / HCA layers keep the masked SDPA (their entry visibility is not a token-causal relation).
"""

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from ..common import _HIFI4, DeepSeekV4Module
from ..decode.moe import _tp_all_reduce
from ..layers import Linear
from ..weight_cache import WeightCache, _as_cache, _load_weight, _materialize

SLIDING_ATTENTION = "sliding_attention"
COMPRESSED_SPARSE_ATTENTION = "compressed_sparse_attention"
HEAVILY_COMPRESSED_ATTENTION = "heavily_compressed_attention"

# Chunk lengths and start positions are multiples of this: it is the HCA compress rate and the
# SDPA q/k chunk, and a whole number of CSA windows and tiles.
ALIGNMENT = 128
_SDPA_CHUNK = 128

# ``indexer_score_dsa`` keeps every index head resident, so its q and gate circular buffers are
# ``index_n_heads * q_chunk_size`` rows (64 heads x 64 rows: ~1.3 MB of the core's L1). 32 halves them, which
# leaves room for the L1 the decode model keeps on the same cores; the scores are identical.
_INDEXER_PROGRAM_CONFIG = dict(q_chunk_size=32, k_chunk_size=64, head_group_size=0)

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
    # Lightning indexer (CSA with ``lightning_indexer=True``): ROW_MAJOR bf16 ``[1, 1, E, index_head_dim]`` index keys
    # (normed + roped, one per compressed entry) and the indexer compressor's own Ca overlap (fp32, as above).
    idx_keys: Optional[ttnn.Tensor] = None
    idx_prev_kv: Optional[ttnn.Tensor] = None
    idx_prev_gate: Optional[ttnn.Tensor] = None

    @property
    def num_entries(self) -> int:
        """Compressed entries emitted so far."""
        return 0 if self.compressed_kv is None else self.compressed_kv.shape[2]


@dataclass
class PrefillStaticBuffers:
    """One layer's *persistent* attention state for traced prefill (see :class:`~..model.TracedPrefill`).

    A trace replays a fixed op sequence over fixed buffers, so unlike :class:`PrefillAttentionState` (whose
    ``compressed_kv`` grows by a concat per chunk) nothing here ever changes shape or address; every chunk
    updates the buffers in place.

    * ``tail``      -- ROW_MAJOR bf16 ``[1, 1, sliding_window, Dh]``: the last window of roped K=V rows.
    * ``entries``   -- ROW_MAJOR bf16 ``[1, 1, cap, Dh]`` (HCA / CSA): the compressed entries, *front-anchored* as
      in decode's caches: entry ``w`` always sits in row ``w``, so after a chunk the valid rows are the prefix
      ``[0, (start + T) / rate)``. A chunk writes its ``T / rate`` entries at rows ``start / rate + w`` (row ids
      from the packet, ``PrefillStaticStep.entry_rows``) with ``indexed_fill``; rows past the prefix are stale
      or zero and never visible (:meth:`DeepSeekV4PrefillAttention.mask_tables_host`).
    * ``prev_kv`` / ``prev_gate`` -- CSA's overlap window, exactly as in :class:`PrefillAttentionState`.
    * ``idx_keys`` / ``idx_prev_kv`` / ``idx_prev_gate`` -- the lightning indexer's keys (same layout as
      ``entries``, width ``index_head_dim``) and its own overlap window, when the layer runs the indexer.

    Every chunk is ``C`` tokens, a prompt's last one padded at its end, so after it ``tail`` and the overlap
    windows describe the padding. The last chunk's rows they are picked from are kept instead, and the real
    state is sliced out of them afterwards (:meth:`~..model.TracedPrefill._export_states`):

    * ``window``    -- ROW_MAJOR bf16 ``[1, 1, sliding_window + C, Dh]``: ``[tail | chunk K=V]`` of the last chunk.
    * ``ca_kv`` / ``ca_gate`` (CSA) and ``idx_ca_kv`` / ``idx_ca_gate`` (indexer) -- ROW_MAJOR fp32
      ``[1, 1, C / rate, rate * width]``: every window's Ca slice of the last chunk (each row an overlap window).
    """

    tail: ttnn.Tensor
    entries: Optional[ttnn.Tensor] = None
    prev_kv: Optional[ttnn.Tensor] = None
    prev_gate: Optional[ttnn.Tensor] = None
    idx_keys: Optional[ttnn.Tensor] = None
    idx_prev_kv: Optional[ttnn.Tensor] = None
    idx_prev_gate: Optional[ttnn.Tensor] = None
    window: Optional[ttnn.Tensor] = None
    ca_kv: Optional[ttnn.Tensor] = None
    ca_gate: Optional[ttnn.Tensor] = None
    idx_ca_kv: Optional[ttnn.Tensor] = None
    idx_ca_gate: Optional[ttnn.Tensor] = None


@dataclass
class PrefillStaticStep:
    """The per-chunk inputs of a traced attention layer, computed inside the trace from the chunk's packet.

    ``rope[kind]`` = ``(cos, sin)`` ``[1, 1, T, rope_dim]`` for the chunk's tokens, ``entry_rope[rate]`` the same
    for the chunk's new compressed entries (positions ``start + w * rate``), ``masks[layer_type]`` the additive
    mask ``[1, 1, T, sliding_window + T + cap]`` (see :meth:`DeepSeekV4PrefillAttention.mask_tables_host`) and
    ``caps[layer_type]`` the number of entry rows (``cap``) that layer type's SDPA reads (0 for sliding layers). The
    entry buffers are front-anchored, so ``cap`` may be below their size: a trace reads and writes only their first
    ``cap`` rows, enough for every chunk of its position tier (``TracedPrefill``), and ``masks`` / ``index_cuts`` /
    ``index_pads`` are sized to it.
    """

    rope: dict
    entry_rope: dict
    masks: dict
    caps: dict
    # Lightning indexer (CSA only): the causal cut ``[1, 1, T, cap]`` over the entry rows (the mask's entry columns),
    # and a persistent zero key pad ``[1, 1, kv_len, index_head_dim]`` the trace fills with those rows before scoring.
    index_cuts: Optional[dict] = None
    index_pads: Optional[dict] = None
    # rate -> uint32 ROW_MAJOR ``[T / rate]``: the entry rows ``start / rate + w`` this chunk's entries are written to.
    entry_rows: Optional[dict] = None
    # Mask-free sliding layers: this step is the prompt's first chunk (its tail holds no real token). A trace bakes
    # it in, so the traced prefill captures a separate first-chunk trace.
    first_chunk: bool = False


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

    ``tp_size > 1`` expects ``device`` to be a 1xTP mesh with one device per rank, ``num_attention_heads``
    and ``o_groups`` divisible by ``tp_size``, and a replicated ``[1, 1, T, D]`` hidden chunk; the output
    is the all-reduced, replicated ``[1, 1, T, D]`` (see the module docstring). Every state tensor is
    replicated, so read one rank's copy.
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
        dense_csa: bool = False,
        lightning_indexer: bool = False,
        maskless: bool = False,
    ):
        """Upload the layer's weights and constants to ``device``.

        ``maskless`` runs a sliding layer without an additive mask (dense causal sliding-window SDPA, see the module
        docstring); CSA / HCA layers ignore it.

        ``lightning_indexer`` runs the real lightning indexer on CSA layers (needs the ``compressor.indexer.*``
        weights): past ``index_topk`` compressed entries each query attends to the top-``index_topk`` entries by
        indexer score, exactly as the model does, so there is no length limit. Attention itself stays the dense
        masked SDPA (the selection is folded into the additive mask). :meth:`forward` and the traced
        :meth:`forward_static` both run it.

        ``dense_csa`` lets a CSA layer run past ``index_topk`` compressed entries *without* the lightning
        indexer: every causally visible compressed entry is attended to, instead of the indexer's top
        ``index_topk``. That is a superset of what the model selects, so it is NOT numerically the model
        (the indexer's job is to drop the low-scoring entries); it exists to exercise long prompts end to
        end until the indexer + sparse SDPA land. Inside the supported regime (at most ``index_topk``
        entries) it changes nothing.

        ``config`` is the HF ``DeepseekV4Config`` (or anything with the same attributes),
        ``layer_idx`` picks the layer type from ``config.layer_types``, ``cache`` is an optional
        :class:`~..weight_cache.WeightCache` namespace for the converted weights (this block's
        entries carry a ``.prefill`` suffix so they never collide with the decode layouts; the
        rank-sharded ones add ``.tp{tp_size}``).
        """
        if tp_size < 1:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        if tp_size > 1 and device.get_num_devices() != tp_size:
            raise ValueError(
                f"tensor-parallel attention expects one device per TP rank, got tp_size={tp_size} "
                f"on a {device.get_num_devices()}-device mesh"
            )
        self.tp_size = tp_size
        # Replicates a host tensor onto every rank; ``None`` (plain upload) on one device.
        self._replicate = ttnn.ReplicateTensorToMesh(device) if tp_size > 1 else None
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
        if self.num_heads % tp_size:
            raise ValueError(f"num_attention_heads {self.num_heads} is not divisible by tp_size {tp_size}")
        if self.o_groups % tp_size:
            raise ValueError(f"o_groups {self.o_groups} is not divisible by tp_size {tp_size}")
        # Heads and o_a groups are split over the ranks: rank r owns heads [r*H/TP, (r+1)*H/TP), which
        # are exactly the inputs of groups [r*g/TP, (r+1)*g/TP).
        self.local_num_heads = self.num_heads // tp_size
        self.local_o_groups = self.o_groups // tp_size
        if self.sliding_window % ALIGNMENT:
            raise ValueError(f"sliding_window {self.sliding_window} must be a multiple of {ALIGNMENT}")

        self.is_sliding = self.layer_type == SLIDING_ATTENTION
        self.is_csa = self.layer_type == COMPRESSED_SPARSE_ATTENTION
        self.rate = None if self.is_sliding else config.compress_rates[self.layer_type]
        if self.rate is not None and ALIGNMENT % self.rate:
            raise ValueError(f"compress rate {self.rate} must divide the chunk alignment {ALIGNMENT}")
        self.index_topk = config.index_topk
        self.dense_csa = dense_csa
        self.use_indexer = bool(lightning_indexer) and self.is_csa
        # Sliding layers use the plain theta table, CSA / HCA the YaRN-scaled one they share
        # with their compressor.
        self.rope_kind = "main" if self.is_sliding else "compress"
        self.rope = rope

        cache = _as_cache(cache)

        def linear(name: str, dtype: ttnn.DataType = weight_dtype, shard_dim: Optional[int] = None) -> Linear:
            """``Linear`` for ``{name}.weight``: replicated over the ranks, or split on ``shard_dim`` of
            the transposed ``[K, N]`` weight (-1 = column-parallel, -2 = row-parallel) when TP > 1."""
            if shard_dim is not None and tp_size > 1:
                mapper = ttnn.ShardTensorToMesh(device, dim=shard_dim)
                cache_name = f"{name}.prefill.tp{tp_size}"
            else:
                # A replicated weight is cached unsharded, so it shares the single-device file.
                mapper = self._replicate
                cache_name = f"{name}.prefill"
            return Linear(weights[f"{name}.weight"], device, cache.file(cache_name), dtype=dtype, mesh_mapper=mapper)

        def host(name: str) -> torch.Tensor:
            w = weights[name]
            return (w() if callable(w) else w).detach().float()

        def norm_gamma(name: str) -> ttnn.Tensor:
            return self._to_device(host(name).reshape(1, 1, 1, -1))

        self.q_a_proj = linear("q_a_proj")
        self.q_b_proj = linear("q_b_proj", shard_dim=-1)  # column-parallel: this rank's heads
        self.kv_proj = linear("kv_proj")
        self.o_b_proj = linear("o_b_proj", shard_dim=-2)  # row-parallel: this rank's groups, then all-reduce
        self.q_a_norm_weight = norm_gamma("q_a_norm.weight")
        self.kv_norm_weight = norm_gamma("kv_norm.weight")
        # The per-head Q norm has no weight in the reference; a ones gamma is the unweighted norm.
        self.q_b_norm_weight = self._to_device(torch.ones(1, 1, 1, self.head_dim))

        # Grouped output projection: block-diagonal over o_groups, run as one batched matmul.
        # The HF weight is [g * o_lora, in_per_group]; the batched matmul wants [1, g, K, N].
        in_per_group = self.num_heads * self.head_dim // self.o_groups
        # Under TP each rank keeps ``o_groups / TP`` complete groups (the group axis, dim 1, is split).
        o_a_file = cache.file(f"o_a_proj.prefill.tp{tp_size}" if tp_size > 1 else "o_a_proj.prefill")
        o_a = _materialize(weights["o_a_proj.weight"], o_a_file, weight_dtype)
        if o_a is not None:
            o_a = o_a.detach().reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).unsqueeze(0)
            o_a = o_a.contiguous()
        self.o_a_proj = _load_weight(
            o_a,
            device,
            cache_file_name=o_a_file,
            dtype=weight_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1) if tp_size > 1 else None,
        )

        # The sink joins the softmax denominator un-scaled, but SDPA multiplies ``scale`` into the
        # QK logits and the sink alike, so pre-divide by the scale to cancel it. Sharded on the head
        # axis so each rank's sinks line up with its query heads.
        sinks = host("sinks").reshape(1, self.num_heads, 1, 1) / self.scaling
        self.sinks = self._to_device(sinks, shard_dim=1)

        self.maskless = bool(maskless) and self.is_sliding
        self._q_pad = None  # [1, H_local, sliding_window, Dh] zeros, the mask-free sliding query's front pad
        if self.maskless:
            self._q_pad = ttnn.from_torch(
                torch.zeros(1, self.local_num_heads, self.sliding_window, self.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=self._replicate,
            )

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

        if self.use_indexer:
            # Lightning indexer, replicated on every rank (all index heads on each): its selection row is the
            # head-sum, so replicating avoids a collective at the price of redundant (cheap) compute.
            self.index_heads = config.index_n_heads
            self.index_head_dim = config.index_head_dim
            if self.index_head_dim < self.rope_dim or self.index_head_dim % ttnn.TILE_SIZE:
                raise ValueError(f"index_head_dim {self.index_head_dim} unsupported (rope_dim {self.rope_dim})")
            self.i_q_b_proj = linear("compressor.indexer.q_b_proj")
            self.i_weights_proj = linear("compressor.indexer.weights_proj", ttnn.bfloat16)
            self.i_kv_proj = linear("compressor.indexer.kv_proj", ttnn.bfloat16)
            self.i_gate_proj = linear("compressor.indexer.gate_proj", ttnn.bfloat16)
            self.i_norm_weight = norm_gamma("compressor.indexer.kv_norm.weight")
            i_bias = host("compressor.indexer.position_bias")
            assert tuple(i_bias.shape) == (self.rate, 2 * self.index_head_dim), tuple(i_bias.shape)
            self.i_bias_slots = [
                self._to_device(i_bias[s].reshape(1, 1, 1, -1), dtype=ttnn.float32) for s in range(self.rate)
            ]
            # relu(q.k) * softmax_scale, weighted by weights_proj(x) * n_heads^-0.5, summed over heads.
            self.index_scale = float(self.index_head_dim**-0.5 * self.index_heads**-0.5)

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    def _to_device(
        self, t: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16, shard_dim: Optional[int] = None
    ) -> ttnn.Tensor:
        """A host tensor as a TILE, DRAM-interleaved ``dtype`` tensor on the device.

        Replicated on every rank under TP, unless ``shard_dim`` splits that axis over the ranks.
        """
        if shard_dim is not None and self.tp_size > 1:
            mapper = ttnn.ShardTensorToMesh(self.device, dim=shard_dim)
        else:
            mapper = self._replicate
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=mapper)

    def _zeros_rm(self, rows: int, width: Optional[int] = None) -> ttnn.Tensor:
        """A ROW_MAJOR bf16 ``[1, 1, rows, width]`` zero tensor (``width`` defaults to ``Dh``)."""
        return ttnn.from_torch(
            torch.zeros(1, 1, rows, width or self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=self._replicate,
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
        heads, rows, width = x.shape[1], x.shape[2], x.shape[3]  # width: Dh, or index_head_dim for the indexer
        nope_dim = width - self.rope_dim
        nope = ttnn.slice(x, [0, 0, 0, 0], [1, heads, rows, nope_dim])
        rope = ttnn.slice(x, [0, 0, 0, nope_dim], [1, heads, rows, width])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    # ------------------------------------------------------------------ #
    # Q / K=V stems
    # ------------------------------------------------------------------ #
    def _q_stem(self, hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, return_latent: bool = False):
        """``[1, 1, T, D]`` -> roped, per-head-normed queries ``[1, H_local, T, Dh]``.

        ``q_a`` and its norm run replicated; ``q_b`` is column-parallel, so under TP each rank ends up
        with its own ``H / TP`` heads (all ``H`` on one device).
        """
        q = self.q_a_proj(hidden)
        q = ttnn.rms_norm(q, weight=self.q_a_norm_weight, epsilon=self.eps)
        latent = q  # the indexer's ``q_residual``
        q = self.q_b_proj(latent)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q,
            num_heads=self.local_num_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.rms_norm(q, weight=self.q_b_norm_weight, epsilon=self.eps)
        q = self._rope(q, cos, sin)
        return (q, latent) if return_latent else q

    def _kv_stem(self, hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, T, D]`` -> the roped single K=V head ``[1, 1, T, Dh]``."""
        kv = self.kv_proj(hidden)
        kv = ttnn.rms_norm(kv, weight=self.kv_norm_weight, epsilon=self.eps)
        return self._rope(kv, cos, sin)

    # ------------------------------------------------------------------ #
    # compressors
    # ------------------------------------------------------------------ #
    def _finish_entries(
        self,
        pooled: ttnn.Tensor,
        first_position: int = 0,
        tables: Optional[tuple] = None,
        norm_weight: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Norm + RoPE the pooled ``[1, 1, n_win, Dh]`` entries, returned ROW_MAJOR.

        Entry ``w`` is RoPE'd at the absolute position ``first_position + w * rate``: the position
        of its window's first token, so entries stay valid across chunks. ``tables`` (traced prefill)
        hands in those ``(cos, sin)`` as persistent device tensors instead of uploading them here.
        """
        n_win = pooled.shape[2]
        pooled = ttnn.rms_norm(
            pooled, weight=self.c_norm_weight if norm_weight is None else norm_weight, epsilon=self.eps
        )
        if tables is None:
            positions = first_position + torch.arange(n_win) * self.rate
            tables = self._rope_tables(positions)
        cos, sin = tables
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

    def _shift_windows(
        self, cur: ttnn.Tensor, prev_row: ttnn.Tensor, rows_out: Optional[ttnn.Tensor] = None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Shift per-window rows down by one: row ``w`` becomes window ``w-1``'s, row 0 is ``prev_row``.

        ``cur`` is TILE ``[1, 1, n_win, W]`` and ``prev_row`` ROW_MAJOR ``[1, 1, 1, W]``. Returns the
        shifted tensor (TILE) and ``cur``'s last row (ROW_MAJOR), the next chunk's ``prev_row``. ``rows_out``
        (traced prefill) also gets every row of ``cur``, ROW_MAJOR.
        """
        n_win, width = cur.shape[2], cur.shape[3]
        cur_rm = ttnn.to_layout(cur, ttnn.ROW_MAJOR_LAYOUT)
        if rows_out is not None:
            self._write_rows(cur_rm, rows_out, 0)
        head = ttnn.slice(cur_rm, [0, 0, 0, 0], [1, 1, n_win - 1, width])
        shifted = ttnn.to_layout(ttnn.concat([prev_row, head], dim=2), ttnn.TILE_LAYOUT)
        last = ttnn.slice(cur_rm, [0, 0, n_win - 1, 0], [1, 1, n_win, width])
        return shifted, last

    def _initial_csa_overlap(self, head_dim: Optional[int] = None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The "previous window" of a prompt's first CSA window: zero kv and a gate that weighs 0."""
        width = self.rate * (head_dim or self.head_dim)
        kv = ttnn.from_torch(
            torch.zeros(1, 1, 1, width),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=self._replicate,
        )
        gate = ttnn.from_torch(
            torch.full((1, 1, 1, width), _NO_WINDOW_GATE),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=self._replicate,
        )
        return kv, gate

    def _pool_csa(
        self,
        hidden: ttnn.Tensor,
        state: PrefillAttentionState,
        indexer: bool = False,
        rows_out: Optional[tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> ttnn.Tensor:
        """CSA's overlapping pooling: ``[1, 1, T, D]`` -> ``[1, 1, T/rate, Dh]``; updates the overlap state.

        ``rows_out`` = ``(kv, gate)`` (traced prefill) also gets every window's Ca slice (see ``_shift_windows``).

        Every token projects to ``2*Dh`` = ``Ca | Cb``. Entry ``w`` is the softmax-gated combination
        over ``2*rate`` slots: window ``w-1``'s Ca slice (``rate`` slots) then window ``w``'s Cb
        slice (``rate`` slots), with ``position_bias[s]`` added to slot ``s``'s gate before the
        split. The 8-way softmax is written out elementwise in fp32 (max, exp, sum, divide) over
        ``[n_win, Dh]`` slot tensors instead of reshaping to a 4-row window axis.
        """
        rate = self.rate
        if indexer:  # the indexer's own compressor: same scheme at index_head_dim, its own weights and overlap
            dh = self.index_head_dim
            kv_proj, gate_proj, bias_slots = self.i_kv_proj, self.i_gate_proj, self.i_bias_slots
        else:
            dh = self.head_dim
            kv_proj, gate_proj, bias_slots = self.c_kv_proj, self.c_gate_proj, self.c_bias_slots
        wd = 2 * dh
        n_win = hidden.shape[2] // rate

        kv = self._window_view(kv_proj(hidden), n_win, rate * wd)
        gate = self._window_view(gate_proj(hidden), n_win, rate * wd)

        def slot(x: ttnn.Tensor, s: int, lo: int, hi: int) -> ttnn.Tensor:
            return ttnn.slice(x, [0, 0, 0, s * wd + lo], [1, 1, n_win, s * wd + hi])

        # Position bias joins the gate before the Ca / Cb split (and before the overlap is saved).
        gate_slots = [ttnn.add(slot(gate, s, 0, wd), bias_slots[s]) for s in range(rate)]

        # This chunk's Ca slice, all slots side by side [1, 1, n_win, rate * Dh].
        ca_kv = ttnn.concat([slot(kv, s, 0, dh) for s in range(rate)], dim=-1)
        ca_gate = ttnn.concat([ttnn.slice(gate_slots[s], [0, 0, 0, 0], [1, 1, n_win, dh]) for s in range(rate)], dim=-1)

        kv_out, gate_out = rows_out or (None, None)
        if indexer:
            if state.idx_prev_kv is None:
                state.idx_prev_kv, state.idx_prev_gate = self._initial_csa_overlap(dh)
            prev_kv, state.idx_prev_kv = self._shift_windows(ca_kv, state.idx_prev_kv, kv_out)
            prev_gate, state.idx_prev_gate = self._shift_windows(ca_gate, state.idx_prev_gate, gate_out)
        else:
            if state.csa_prev_kv is None:
                state.csa_prev_kv, state.csa_prev_gate = self._initial_csa_overlap()
            prev_kv, state.csa_prev_kv = self._shift_windows(ca_kv, state.csa_prev_kv, kv_out)
            prev_gate, state.csa_prev_gate = self._shift_windows(ca_gate, state.csa_prev_gate, gate_out)

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
        if self.use_indexer:
            keys = self._finish_entries(
                self._pool_csa(hidden, state, indexer=True), state.seq_len, norm_weight=self.i_norm_weight
            )
            state.idx_keys = keys if state.idx_keys is None else ttnn.concat([state.idx_keys, keys], dim=2)
        state.compressed_kv = (
            entries if state.compressed_kv is None else ttnn.concat([state.compressed_kv, entries], dim=2)
        )
        return state.compressed_kv

    # ------------------------------------------------------------------ #
    # attention
    # ------------------------------------------------------------------ #
    def _mask(
        self,
        num_tokens: int,
        start: int,
        num_entries: int,
        padded_keys: int,
        entry_sel: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Additive mask ``[1, 1, T, padded_keys]`` over the key layout ``[tail | chunk | entries | pad]``.

        * Window part (``sliding_window + T`` columns; column ``k`` is the token at absolute position
          ``start - sliding_window + k``): query ``i`` (absolute ``start + i``) sees the ``sliding_window``
          tokens ending at itself, i.e. ``i < k <= i + sliding_window``, and never a position before 0
          (the zero rows the first chunk's empty tail stands in for).
        * Entry part: query ``i`` sees entry ``w`` iff ``w < (start + i + 1) // rate`` (the entry's
          window is fully in the query's past). For CSA this is exactly the indexer's output while
          there are no more than ``index_topk`` entries.
        * Padding columns are masked.

        ``entry_sel`` (lightning indexer) is a device ``[1, 1, T, num_entries]`` additive TILE mask that replaces
        the entry part: 0 on each query's top-``index_topk`` causally visible entries, -inf elsewhere.
        """
        sw = self.sliding_window
        query = torch.arange(num_tokens).view(-1, 1)
        key = torch.arange(sw + num_tokens).view(1, -1)
        window = (key <= query + sw) & (key > query) & (key + (start - sw) >= 0)
        mask = torch.full((num_tokens, padded_keys), float("-inf"))
        mask[:, : sw + num_tokens].masked_fill_(window, 0.0)
        if entry_sel is not None:
            window_cols = sw + num_tokens
            parts = [self._to_device(mask[:, :window_cols].reshape(1, 1, num_tokens, window_cols)), entry_sel]
            pad = padded_keys - window_cols - num_entries
            if pad:
                parts.append(self._to_device(torch.full((1, 1, num_tokens, pad), float("-inf"))))
            return ttnn.concat(parts, dim=3)
        if num_entries:
            entry = torch.arange(num_entries).view(1, -1)
            visible = ((start + torch.arange(num_tokens) + 1) // self.rate).view(-1, 1)
            mask[:, sw + num_tokens : sw + num_tokens + num_entries].masked_fill_(entry < visible, 0.0)
        return self._to_device(mask.reshape(1, 1, num_tokens, padded_keys))

    # ------------------------------------------------------------------ #
    # lightning indexer (CSA)
    # ------------------------------------------------------------------ #
    def _select_entries(
        self,
        latent: ttnn.Tensor,
        hidden: ttnn.Tensor,
        state: PrefillAttentionState,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """The indexer's selection for this chunk: an additive TILE mask ``[1, 1, T, E]`` over all ``E`` entries so far.

        Score ``s(t, w) = sum_h relu(q_{t,h} . k_w) * softmax_scale * (weights_proj(x_t)_h * n_heads^-0.5)`` with
        ``q = rope(q_b_proj(q_residual))`` and ``k`` the indexer keys (:meth:`_compress`); entry ``w`` is causally
        visible to token ``t`` iff ``w < (t + 1) // rate``. The mask is 0 on the ``index_topk`` best visible entries of
        each query (selected by threshold: everything scoring at least the k-th best, so no scatter is needed) and
        -inf elsewhere; a query with fewer than ``index_topk`` visible entries keeps all of them.

        ``state.idx_keys`` must already hold this chunk's keys, so ``E = (seq_len + T) // rate`` (``state.seq_len``
        is the chunk's start).
        """
        scores, cut = self._index_scores(latent, hidden, state, cos, sin)
        k = min(self.index_topk, scores.shape[3])
        theta = ttnn.min(ttnn.topk(scores, k=k, dim=-1, largest=True, sorted=True)[0], dim=-1, keepdim=True)
        return ttnn.add(ttnn.log(ttnn.ge(scores, theta)), cut)

    def _index_scores(
        self,
        latent: ttnn.Tensor,
        hidden: ttnn.Tensor,
        state: PrefillAttentionState,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``(scores, cut)``: the indexer scores ``[1, 1, T, E]`` TILE with the causal cut already added, and the cut."""
        num_tokens, start = hidden.shape[2], state.seq_len
        num_entries = state.idx_keys.shape[2]
        heads = self.index_heads

        q = self.i_q_b_proj(latent)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=heads, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = self._rope(q, cos, sin)  # [1, Hi, T, Di]
        weights = ttnn.multiply(self.i_weights_proj(hidden), self.index_scale)  # [1, 1, T, Hi]

        # ``indexer_score_dsa`` applies token causality (key t visible to query s iff t <= chunk_start + s), which is
        # the wrong relation for compressed entries. Point ``chunk_start`` past every real entry so all of them are
        # visible, score ``T`` phantom (zero) key rows past them that we never read, and cut causally ourselves.
        tile = ttnn.TILE_SIZE
        entry_tiles = -(-num_entries // tile) * tile
        kv_len = -(-(entry_tiles + num_tokens) // 64) * 64
        keys = self.idx_keys_padded(state, kv_len)
        logits = ttnn.experimental.indexer_score_dsa(
            q,
            keys,
            weights,
            chunk_start_idx=entry_tiles,
            kv_len=kv_len,
            program_config=ttnn.IndexerScoreProgramConfig(**_INDEXER_PROGRAM_CONFIG),
            seq_shard_axes=[0] if self.tp_size > 1 else [],  # 1xTP mesh: the SP axis has extent 1 -> no offset
        )  # [1, 1, T, kv_len] bf16 ROW_MAJOR, columns [0, kv_len) written
        scores = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, num_tokens, num_entries])
        scores = ttnn.to_layout(scores, ttnn.TILE_LAYOUT)

        entry = torch.arange(num_entries).view(1, -1)
        visible = ((start + torch.arange(num_tokens) + 1) // self.rate).view(-1, 1)
        cut = self._to_device(torch.where(entry < visible, 0.0, float("-inf")).reshape(1, 1, num_tokens, num_entries))
        return ttnn.add(scores, cut), cut

    def idx_keys_padded(self, state: PrefillAttentionState, rows: int) -> ttnn.Tensor:
        """The index keys as a TILE tensor ``[1, 1, rows, Di]``, zero rows appended up to ``rows``."""
        keys = state.idx_keys
        if rows > keys.shape[2]:
            keys = ttnn.concat([keys, self._zeros_rm(rows - keys.shape[2], self.index_head_dim)], dim=2)
        return ttnn.to_layout(keys, ttnn.TILE_LAYOUT)

    def _attend(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        entries: Optional[ttnn.Tensor],
        state: PrefillAttentionState,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        entry_sel: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """SDPA of ``q`` over ``[tail | chunk | entries | pad]``, then undo V's RoPE. Rolls the KV tail.

        ``entry_sel`` is the lightning indexer's selection mask (see :meth:`_mask`)."""
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

        mask = self._mask(num_tokens, state.seq_len, num_entries, padded_keys, entry_sel)
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

    # ------------------------------------------------------------------ #
    # mask-free attention
    # ------------------------------------------------------------------ #
    def _sliding_attend(self, q: ttnn.Tensor, window: ttnn.Tensor, real_tail: int) -> ttnn.Tensor:
        """Mask-free sliding-window SDPA of ``q`` ``[1, H_local, T, Dh]`` over the ROW_MAJOR ``window`` ``[tail | chunk]``.

        Only the last ``real_tail`` tail rows hold tokens (``min(start, sliding_window)``; the rest stand for
        positions before 0), so the keys are those rows plus the chunk, and ``q`` is front-padded with as many zero
        rows so ``Sq == Sk`` (which ``is_causal`` requires): query ``real_tail + i`` then sees exactly its window
        through ``sliding_window_size``, and the pad rows' outputs are dropped. Returns ``[1, H_local, T, Dh]`` TILE.
        """
        sw = self.sliding_window
        num_tokens = q.shape[2]
        keys = ttnn.slice(window, [0, 0, sw - real_tail, 0], [1, 1, sw + num_tokens, self.head_dim])
        keys = ttnn.to_layout(keys, ttnn.TILE_LAYOUT)
        if real_tail:
            pad = self._q_pad
            if real_tail < sw:
                pad = ttnn.slice(pad, [0, 0, 0, 0], [1, self.local_num_heads, real_tail, self.head_dim])
            q = ttnn.concat([pad, q], dim=2)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            keys,
            keys,
            is_causal=True,
            sliding_window_size=sw,
            scale=self.scaling,
            attention_sink=self.sinks,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=_SDPA_CHUNK,
                k_chunk_size=_SDPA_CHUNK,
                exp_approx_mode=False,
            ),
        )
        ttnn.deallocate(keys)
        if real_tail:
            ttnn.deallocate(q)
            full = attn
            attn = ttnn.slice(
                full, [0, 0, real_tail, 0], [1, self.local_num_heads, real_tail + num_tokens, self.head_dim]
            )
            ttnn.deallocate(full)
        return attn

    def _attend_maskless(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        state: PrefillAttentionState,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """:meth:`_attend` for a mask-free sliding layer (:meth:`_sliding_attend`). Rolls the KV tail."""
        num_tokens = q.shape[2]
        sw = self.sliding_window
        chunk_rm = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
        tail = state.kv_tail if state.kv_tail is not None else self._zeros_rm(sw)
        window = ttnn.concat([tail, chunk_rm], dim=2)  # [1, 1, sw + T, Dh]
        state.kv_tail = ttnn.slice(window, [0, 0, num_tokens, 0], [1, 1, num_tokens + sw, self.head_dim])
        attn = self._sliding_attend(q, window, min(state.seq_len, sw))
        return self._rope(attn, cos, ttnn.neg(sin))

    def _o_proj(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """Grouped output projection: ``[1, H_local, T, Dh]`` -> ``[1, 1, T, D]``.

        Consecutive heads form a group (``H / g`` heads = ``in_per_group`` channels): ``o_a`` maps each
        group to ``o_lora_rank`` independently (one batched matmul), the results are laid side by
        side and ``o_b`` mixes them to the hidden size.

        Under TP a rank holds ``g / TP`` complete groups (its own heads), so ``o_a`` runs locally and
        ``o_b`` is row-parallel over those groups' outputs: each rank's result is a full-``D`` partial
        that one all-reduce sums into the replicated output.
        """
        num_tokens = attn.shape[2]
        g = self.local_o_groups
        heads_per_group = self.num_heads // self.o_groups
        in_per_group = heads_per_group * self.head_dim

        x = ttnn.reshape(attn, [g, heads_per_group, num_tokens, self.head_dim])
        x = ttnn.experimental.nlp_concat_heads(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [g, 1, T, in_per_group]
        x = ttnn.reshape(x, [1, g, num_tokens, in_per_group])

        grouped = ttnn.linear(x, self.o_a_proj, compute_kernel_config=_HIFI4)  # [1, g, T, o_lora_rank]
        grouped = ttnn.concat(
            [ttnn.slice(grouped, [0, gi, 0, 0], [1, gi + 1, num_tokens, self.o_lora_rank]) for gi in range(g)],
            dim=-1,
        )  # [1, 1, T, g_local * o_lora_rank]
        out = self.o_b_proj(grouped)
        if self.tp_size > 1:
            out = _tp_all_reduce(out, self.device)
        return out

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
        if self.is_csa and not self.dense_csa and not self.use_indexer:
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

        Under TP ``hidden`` must be replicated on every rank, and the result is replicated too (after
        the ``o_b`` all-reduce), as is every tensor in ``state``.
        """
        if state is None:
            state = self.new_state()
        self._check_chunk(hidden, state)
        num_tokens = hidden.shape[2]
        start = state.seq_len

        cos, sin = self._rope_tables(start + torch.arange(num_tokens))
        q, latent = self._q_stem(hidden, cos, sin, return_latent=True)
        kv = self._kv_stem(hidden, cos, sin)
        entries = self._compress(hidden, state)
        # Below index_topk entries top-k selects every visible entry (the plain causal set); past it, select.
        selects = self.use_indexer and entries.shape[2] > self.index_topk
        if self.maskless:
            attn = self._attend_maskless(q, kv, state, cos, sin)
        else:
            entry_sel = self._select_entries(latent, hidden, state, cos, sin) if selects else None
            attn = self._attend(q, kv, entries, state, cos, sin, entry_sel)
        out = self._o_proj(attn)

        state.seq_len += num_tokens
        return out

    # ------------------------------------------------------------------ #
    # traced prefill: static buffers, host-side per-chunk inputs, in-place forward
    # ------------------------------------------------------------------ #
    def new_static_buffers(self, chunk_size: int, entry_capacity: int = 0) -> PrefillStaticBuffers:
        """Persistent zeroed state for :meth:`forward_static` over ``chunk_size``-token chunks; ``entry_capacity``
        front-anchored compressed entry rows.

        Allocated once, before any trace exists (allocating on a device that holds a trace is unsafe).
        """
        bufs = PrefillStaticBuffers(
            tail=self._zeros_rm(self.sliding_window), window=self._zeros_rm(self.sliding_window + chunk_size)
        )
        if not self.is_sliding:
            if entry_capacity <= 0 or entry_capacity % ALIGNMENT:
                raise ValueError(f"entry_capacity {entry_capacity} must be a positive multiple of {ALIGNMENT}")
            bufs.entries = self._zeros_rm(entry_capacity)
            if self.is_csa:
                n_win = chunk_size // self.rate
                bufs.prev_kv, bufs.prev_gate = self._initial_csa_overlap()
                bufs.ca_kv, bufs.ca_gate = (self._zeros_fp32_rm(n_win, self.rate * self.head_dim) for _ in range(2))
                if self.use_indexer:
                    width = self.rate * self.index_head_dim
                    bufs.idx_keys = self._zeros_rm(entry_capacity, self.index_head_dim)
                    bufs.idx_prev_kv, bufs.idx_prev_gate = self._initial_csa_overlap(self.index_head_dim)
                    bufs.idx_ca_kv, bufs.idx_ca_gate = (self._zeros_fp32_rm(n_win, width) for _ in range(2))
        return bufs

    def _zeros_fp32_rm(self, rows: int, width: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.zeros(1, 1, rows, width),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=self._replicate,
        )

    def host_tensor(self, t: torch.Tensor, dtype: ttnn.DataType, layout: ttnn.Layout) -> ttnn.Tensor:
        """``t`` as a *host* ttnn tensor (replicated over the ranks) that :func:`ttnn.copy_host_to_device_tensor`
        can write into a persistent device tensor of the same dtype / layout / shape."""
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self._replicate)

    def mask_tables_host(self, num_tokens: int, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(static [T, K], threshold [T, K])`` fp32 with ``K = sliding_window + T + cap``: the traced mask's constants.

        A traced chunk's additive mask is made on device from its start position (a multiple of ``ALIGNMENT``)
        as ``static + (threshold > start) * NEG``. Key layout ``[tail | chunk | entry rows]``:

        * Window part (``sliding_window + T`` columns; column ``k`` is the token at absolute position
          ``start - sliding_window + k``): query ``i`` sees ``i < k <= i + sliding_window`` (``static``), and
          never a position before 0, i.e. column ``k`` is cut while ``sliding_window - k > start``.
        * Entry part: the ``cap`` front-anchored entry rows, entry ``w`` in row ``w``. Query ``i`` sees it iff
          ``w < start / rate + (i + 1) // rate``, i.e. it is cut while ``rate * (w - (i + 1) // rate) + 1 > start``
          (``start`` is a multiple of ``rate``). That also cuts every row not written yet (``w >= E``).

        The thresholds are integers far below ``2**24``, so the fp32 compare on device is exact. The same columns,
        sliced out of the mask, are the lightning indexer's causal cut over the entry rows.
        """
        sw = self.sliding_window
        window_cols = sw + num_tokens
        query = torch.arange(num_tokens).view(-1, 1)
        key = torch.arange(window_cols).view(1, -1)
        static = torch.full((num_tokens, window_cols + cap), float("-inf"))
        static[:, :window_cols].masked_fill_((key <= query + sw) & (key > query), 0.0)
        threshold = torch.empty(num_tokens, window_cols + cap)
        threshold[:, :window_cols] = (sw - key).float()
        if cap:
            static[:, window_cols:] = 0.0
            row = torch.arange(cap).view(1, -1)
            threshold[:, window_cols:] = (self.rate * (row - (query + 1) // self.rate) + 1).float()
        return static, threshold

    def reset_static(self, bufs: PrefillStaticBuffers) -> None:
        """Rewind ``bufs`` to the start of a prompt with plain host->device writes (no allocation)."""
        if not hasattr(self, "_static_reset_host"):
            self._static_reset_host = {}
        host = self._static_reset_host

        def zeros(key: str, shape: tuple, dtype: ttnn.DataType, fill: float = 0.0) -> ttnn.Tensor:
            # The entry buffers are sized to the prepared prompt, so a later prompt of another length (a new
            # traced plan on this same layer) needs its own host tensor.
            if host.get(key, (None, None))[0] != shape:
                host[key] = (shape, self.host_tensor(torch.full(shape, fill), dtype, ttnn.ROW_MAJOR_LAYOUT))
            return host[key][1]

        ttnn.copy_host_to_device_tensor(zeros("tail", tuple(bufs.tail.shape), ttnn.bfloat16), bufs.tail)
        # ``entries`` and ``idx_keys`` are *not* rewound: they are front-anchored, so a run only reads rows below the
        # ones it has written (every other row is cut by the mask / the indexer's causal cut, and what a previous
        # prompt left there is finite), and only the prompt's own rows are exported. Zeroing them would write
        # ``O(max_len)`` bytes from the host per prompt, however short the prompt.
        if bufs.prev_kv is not None:
            shape = tuple(bufs.prev_kv.shape)
            ttnn.copy_host_to_device_tensor(zeros("prev_kv", shape, ttnn.float32), bufs.prev_kv)
            ttnn.copy_host_to_device_tensor(zeros("prev_gate", shape, ttnn.float32, _NO_WINDOW_GATE), bufs.prev_gate)
        if bufs.idx_keys is not None:
            shape = tuple(bufs.idx_prev_kv.shape)
            ttnn.copy_host_to_device_tensor(zeros("idx_prev_kv", shape, ttnn.float32), bufs.idx_prev_kv)
            ttnn.copy_host_to_device_tensor(
                zeros("idx_prev_gate", shape, ttnn.float32, _NO_WINDOW_GATE), bufs.idx_prev_gate
            )

    @staticmethod
    def _write_rows(src: ttnn.Tensor, dst: ttnn.Tensor, first_row: int) -> None:
        """Write ROW_MAJOR ``src`` ``[1, 1, n, W]`` into rows ``[first_row, first_row + n)`` of ``dst``, in place."""
        n, width = src.shape[2], src.shape[3]
        ttnn.experimental.slice_write(src, dst, [0, 0, first_row, 0], [1, 1, first_row + n, width], [1, 1, 1, 1])

    def _write_entries(self, buf: ttnn.Tensor, new: ttnn.Tensor, rows: ttnn.Tensor, cap: int) -> None:
        """Write ``new`` ``[1, 1, n, W]`` into rows ``rows`` (uint32 ``[n]``, on device) of ``buf``, in place.

        The rows are data (they follow the chunk's start), so a fixed-offset ``slice_write`` cannot do this.
        ``indexed_fill`` builds a new tensor, which is copied back into the persistent buffer.

        ``cap`` is the number of rows the step reads (its position tier, see :class:`~..model.TracedPrefill`),
        at most the buffer's. The buffer is front-anchored, so a tier below the buffer's size only ever touches the
        first ``cap`` rows (every row this chunk writes is below ``cap``): the fill and the copy back then cost
        ``O(cap)``, not ``O(buffer)``.
        """
        total = buf.shape[2]
        if cap > total:
            raise ValueError(f"layer {self.layer_idx}: entry buffer has {total} rows, step reads {cap}")
        if cap == total:
            written = ttnn.indexed_fill(rows, buf, new, dim=2)
            ttnn.deallocate(new)
            ttnn.copy(written, buf)
            ttnn.deallocate(written)
            return
        head = ttnn.slice(buf, [0, 0, 0, 0], [1, 1, cap, buf.shape[3]])  # a copy: cap < total
        written = ttnn.indexed_fill(rows, head, new, dim=2)
        ttnn.deallocate(head)
        ttnn.deallocate(new)
        self._write_rows(written, buf, 0)
        ttnn.deallocate(written)

    @staticmethod
    def _entry_prefix(buf: ttnn.Tensor, cap: int) -> tuple[ttnn.Tensor, bool]:
        """``(rows, owned)``: the first ``cap`` rows of the front-anchored entry buffer ``buf``.

        The buffer itself when ``cap`` covers it (``owned`` is False: never free it), else a copy (``owned``)."""
        if cap >= buf.shape[2]:
            return buf, False
        return ttnn.slice(buf, [0, 0, 0, 0], [1, 1, cap, buf.shape[3]]), True

    def _compress_static(
        self, hidden: ttnn.Tensor, bufs: PrefillStaticBuffers, step: PrefillStaticStep
    ) -> Optional[ttnn.Tensor]:
        """:meth:`_compress` into the persistent entry rows: returns them (``bufs.entries``, all ``cap`` rows)."""
        if self.is_sliding:
            return None
        shim = PrefillAttentionState(csa_prev_kv=bufs.prev_kv, csa_prev_gate=bufs.prev_gate)
        if self.is_csa:
            pooled = self._pool_csa(hidden, shim, rows_out=(bufs.ca_kv, bufs.ca_gate))
        else:
            pooled = self._pool_hca(hidden)
        new = self._finish_entries(pooled, tables=step.entry_rope[self.rate])
        self._write_entries(bufs.entries, new, step.entry_rows[self.rate], step.caps[self.layer_type])
        if self.is_csa:
            self._write_rows(shim.csa_prev_kv, bufs.prev_kv, 0)
            self._write_rows(shim.csa_prev_gate, bufs.prev_gate, 0)
            ttnn.deallocate(shim.csa_prev_kv)
            ttnn.deallocate(shim.csa_prev_gate)
        return bufs.entries

    def _index_keys_static(self, hidden: ttnn.Tensor, bufs: PrefillStaticBuffers, step: PrefillStaticStep) -> None:
        """Write this chunk's lightning-indexer keys into ``bufs.idx_keys`` (same rows as the entries)."""
        shim = PrefillAttentionState(idx_prev_kv=bufs.idx_prev_kv, idx_prev_gate=bufs.idx_prev_gate)
        pooled = self._pool_csa(hidden, shim, indexer=True, rows_out=(bufs.idx_ca_kv, bufs.idx_ca_gate))
        new = self._finish_entries(pooled, tables=step.entry_rope[self.rate], norm_weight=self.i_norm_weight)
        self._write_entries(bufs.idx_keys, new, step.entry_rows[self.rate], step.caps[self.layer_type])
        self._write_rows(shim.idx_prev_kv, bufs.idx_prev_kv, 0)
        self._write_rows(shim.idx_prev_gate, bufs.idx_prev_gate, 0)
        ttnn.deallocate(shim.idx_prev_kv)
        ttnn.deallocate(shim.idx_prev_gate)

    def _select_static(
        self,
        latent: ttnn.Tensor,
        hidden: ttnn.Tensor,
        bufs: PrefillStaticBuffers,
        step: PrefillStaticStep,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Trace-safe :meth:`_select_entries`: additive TILE mask ``[1, 1, T, cap]`` over the entry rows.

        The key pad and the causal cut are persistent (``step``); ``cap`` and ``T`` are fixed for the trace, so
        ``topk``'s ``k`` is too. A query with fewer than ``k`` visible entries keeps all of them, because the
        k-th score is then ``-inf`` and the causal cut is added back afterwards.
        """
        scores, cut = self._index_scores_static(latent, hidden, bufs, step, cos, sin)
        k = min(self.index_topk, scores.shape[3])
        theta = ttnn.min(ttnn.topk(scores, k=k, dim=-1, largest=True, sorted=True)[0], dim=-1, keepdim=True)
        return ttnn.add(ttnn.log(ttnn.ge(scores, theta)), cut)

    def _index_scores_static(
        self,
        latent: ttnn.Tensor,
        hidden: ttnn.Tensor,
        bufs: PrefillStaticBuffers,
        step: PrefillStaticStep,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Trace-safe :meth:`_index_scores` over the ``cap`` entry rows: ``(scores, cut)`` ``[1, 1, T, cap]``."""
        cap = step.caps[self.layer_type]
        num_tokens = hidden.shape[2]
        q = self.i_q_b_proj(latent)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q,
            num_heads=self.index_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = self._rope(q, cos, sin)
        weights = ttnn.multiply(self.i_weights_proj(hidden), self.index_scale)

        pad = step.index_pads[self.layer_type]
        keys_rm, owned = self._entry_prefix(bufs.idx_keys, cap)
        self._write_rows(keys_rm, pad, 0)
        if owned:
            ttnn.deallocate(keys_rm)
        kv_len = pad.shape[2]
        keys = ttnn.to_layout(pad, ttnn.TILE_LAYOUT)
        logits = ttnn.experimental.indexer_score_dsa(
            q,
            keys,
            weights,
            chunk_start_idx=cap,
            kv_len=kv_len,
            program_config=ttnn.IndexerScoreProgramConfig(**_INDEXER_PROGRAM_CONFIG),
            seq_shard_axes=[0] if self.tp_size > 1 else [],
        )
        scores = ttnn.to_layout(ttnn.slice(logits, [0, 0, 0, 0], [1, 1, num_tokens, cap]), ttnn.TILE_LAYOUT)
        cut = step.index_cuts[self.layer_type]
        return ttnn.add(scores, cut), cut

    def _attend_static(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        visible: Optional[ttnn.Tensor],
        bufs: PrefillStaticBuffers,
        step: PrefillStaticStep,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        entry_sel: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """:meth:`_attend` with the tail rolled in place and the mask supplied by ``step``.

        ``entry_sel`` ``[1, 1, T, cap]`` (the lightning indexer) is added onto the mask's entry columns.
        ``visible`` is the persistent ``bufs.entries``.

        Mask-free sliding layers run :meth:`_sliding_attend` instead: over the chunk alone on the prompt's first
        chunk (``step.first_chunk``), else over the whole window (a later chunk starts at least ``sliding_window``
        in, since chunks are at least that long).
        """
        num_tokens = q.shape[2]
        sw = self.sliding_window
        chunk_rm = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
        window = ttnn.concat([bufs.tail, chunk_rm], dim=2)  # [1, 1, sw + T, Dh]
        ttnn.deallocate(chunk_rm)
        self._write_rows(window, bufs.window, 0)
        new_tail = ttnn.slice(window, [0, 0, num_tokens, 0], [1, 1, num_tokens + sw, self.head_dim])
        self._write_rows(new_tail, bufs.tail, 0)
        ttnn.deallocate(new_tail)

        if self.maskless:
            if num_tokens < sw:
                raise ValueError(f"mask-free traced chunks ({num_tokens}) must be at least sliding_window ({sw})")
            attn = self._sliding_attend(q, window, 0 if step.first_chunk else sw)
            ttnn.deallocate(window)
            return self._rope(attn, cos, ttnn.neg(sin))

        if visible is None:
            kv_all = window
        else:
            rows, owned = self._entry_prefix(visible, step.caps[self.layer_type])  # the tier's rows only
            kv_all = ttnn.concat([window, rows], dim=2)
            ttnn.deallocate(window)
            if owned:
                ttnn.deallocate(rows)
        if kv_all.shape[2] % _SDPA_CHUNK:
            raise ValueError(f"traced SDPA keys ({kv_all.shape[2]}) must be a multiple of {_SDPA_CHUNK}")
        kv_all = ttnn.to_layout(kv_all, ttnn.TILE_LAYOUT)

        mask = step.masks[self.layer_type]
        if entry_sel is not None:
            window_cols = sw + num_tokens
            window = ttnn.slice(mask, [0, 0, 0, 0], [1, 1, num_tokens, window_cols])
            entries = ttnn.add(
                ttnn.slice(mask, [0, 0, 0, window_cols], [1, 1, num_tokens, window_cols + entry_sel.shape[3]]),
                entry_sel,
            )
            mask = ttnn.concat([window, entries], dim=-1)

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
        ttnn.deallocate(kv_all)
        return self._rope(attn, cos, ttnn.neg(sin))

    def forward_static(self, hidden: ttnn.Tensor, bufs: PrefillStaticBuffers, step: PrefillStaticStep) -> ttnn.Tensor:
        """Trace-safe :meth:`forward`: no host traffic, no growing tensors.

        Everything that depends on the chunk's position (RoPE tables, mask, the entries' RoPE) comes from
        ``step``; the state lives in ``bufs`` and is updated in place. ``hidden`` ``[1, 1, T, D]`` -> ``[1, 1, T, D]``.
        """
        if hidden.shape[2] % ALIGNMENT:
            raise ValueError(f"chunk length {hidden.shape[2]} must be a multiple of {ALIGNMENT}")
        cos, sin = step.rope[self.rope_kind]
        stemmed = self._q_stem(hidden, cos, sin, return_latent=self.use_indexer)
        q, latent = stemmed if self.use_indexer else (stemmed, None)
        kv = self._kv_stem(hidden, cos, sin)
        visible = self._compress_static(hidden, bufs, step)
        entry_sel = None
        if self.use_indexer:
            self._index_keys_static(hidden, bufs, step)
            entry_sel = self._select_static(latent, hidden, bufs, step, cos, sin)
        attn = self._attend_static(q, kv, visible, bufs, step, cos, sin, entry_sel)
        return self._o_proj(attn)
