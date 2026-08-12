# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused TTNN decoder layer for ``meta-models/Muse-Glimmer-30B``.

``FusedDecoder`` is a drop-in replacement for
:class:`~models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder.FunctionalDecoder`:
identical public contract (``from_state_dict`` / ``prefill_forward`` /
``decode_forward`` / ``sliding_kv_tail_len`` / ``forward`` / ``kv_cache``),
identical paged-KV semantics — a numerically equivalent op graph with fewer,
larger, more specialised device ops.

Two deliberate departures, both measured and both recorded in the README's
limitations: ``_decode_rope_tables`` takes the documented ``[1, batch]``
``rope_pos_ids`` only, where the functional layer also tolerated a tile-padded
one; and every RMSNorm runs on a higher-fidelity compute-kernel config than the
op's default (see ``norm_compute_kernel_config`` below).  Everything else in
this module is a topology change at unchanged precision.

What is fused, relative to the functional layer
-----------------------------------------------

Dedicated fused ops (highest priority)

1. **RoPE** — ``slice, slice, neg, concat, mul, mul, add`` per tensor (7 ops x
   {Q, K}, plus two ``tilize``s for the cos/sin slices) collapses to a single
   :func:`ttnn.experimental.rotary_embedding_hf` per tensor.  That op
   implements the HuggingFace ``rotate_half`` convention with
   ``cat(freqs, freqs)`` cos/sin tables, i.e. *exactly* the math
   ``MuseGlimmerTextAttention`` spells out, so no weight or table permutation
   is needed (``rotary_embedding_llama`` would need both — it is the Meta
   odd/even-interleaved convention).
2. **Dense projections** dispatch to
   :func:`ttnn.experimental.minimal_matmul` at prefill row counts and stay on
   ``ttnn.linear`` at decode row counts — the same op, a much better kernel
   above the measured crossover and a worse one below it.  See ``_dense``.
   (The decode KV-cache write was *not* collapsed into
   ``paged_fused_update_cache``; see ``_decode_kv_update`` for the measurement.)
3. **Decode RMSNorm** — the four ``hidden_size``-wide norms ran on **one core**
   in decode (``ttnn.rms_norm`` parallelises an interleaved input over rows,
   and decode has a single tile-row), 110 us each = 14 % of the decode step.
   The fused layer keeps the decode residual stream **width-sharded in L1** and
   uses the sharded multi-core ``ttnn.rms_norm`` program config, which is the
   single largest decode win.

Graph rewrites

4. **Prefill RoPE tables** are stored pre-tilized, so the per-chunk cos/sin no
   longer needs a runtime ``to_layout`` (``TilizeDeviceOperation``); at
   ``start_pos == 0`` the op's own ``cos_seq_len >= seq_len`` contract removes
   the ``ttnn.slice`` as well.
5. **Decode RoPE tables** gather straight into the height-sharded
   ``[1, batch, 1, head_dim]`` layout the decode-mode op wants, replacing the
   four ``ttnn.repeat`` broadcasts (and their untilize/tilize round trips) the
   functional layer needed to line cos/sin up with a plain ``ttnn.mul``.
6. **Decode Q** stays height-sharded from ``nlp_create_qkv_heads_decode``
   through RoPE into the SDPA kernel instead of round-tripping through DRAM.

Op merging

7. ``silu(gate) * up`` -> ``ttnn.mul(gate, up, input_tensor_a_activations=[SILU])``.
8. ``concat_heads * sigmoid(attn_gate_proj(x))`` ->
   ``ttnn.mul(heads, gate, input_tensor_b_activations=[SIGMOID])``.
   The matmul's *pack-time* activation was measured first and rejected on both
   dense kernels: on ``ttnn.linear`` it does not fuse at all (a separate
   ``UnaryDeviceOperation`` still runs) *and* slows the matmul, and on
   ``minimal_matmul`` (the kernel prefill actually uses) ``fused_activation``
   costs 12.10 vs 10.28 ms on the MLP gate shape.  See ``_FusedMLP``.

Program-config retuning that the fused graph makes measurable

9. The prefill SDPA runs at ``q_chunk == k_chunk == 256`` instead of 128.
   ``q_chunk == 2 * k_chunk`` is numerically broken with ``sliding_window_size``
   (functional-stage limitation 1), which is why the functional layer pinned
   128 and never swept the *size*; 256/256 is ~33 % faster at 8192 tokens with
   unchanged PCC, including at the lengths that expose the chunk bug.
   See ``doc/fused_decoder/logs/sdpa_chunk_sweep.log``.

Everything else — the paged prefill/decode contract, the internal prefill
chunking, the sliding-window tail hand-off, the ``qk_scale_factor`` fold into
the SDPA scale, the centered-RMSNorm ``1 + w`` fold — is inherited unchanged
from :class:`FunctionalDecoder`.
"""

from __future__ import annotations

from typing import Any

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
    DEFAULT_PREFILL_CHUNK_SIZE,
    LAYER_KIND_FULL,
    LAYER_KIND_SLIDING,
    MODEL_ID,
    PREFILL_SDPA_MAX_SEQ,
    TILE_SIZE,
    FunctionalDecoder,
    MuseGlimmerLayerConfig,
    PagedAttentionConfig,
    _get_layer_tensor,
    _require_muse_glimmer_text_config,
    _rope_cos_sin,
    _text_config,
    _to_device,
    reference_layer_indices,
    resolve_layer_kind,
)
from models.common.lightweightmodule import LightweightModule

__all__ = [
    "FusedDecoder",
    "LAYER_KIND_FULL",
    "LAYER_KIND_SLIDING",
    "MODEL_ID",
    "reference_layer_indices",
    "resolve_layer_kind",
]

#: Q/K flash-attention chunk seed for **both** prefill SDPA call sites: the
#: in-memory square op (chunk 0, and every chunk of a ``sliding`` layer) and the
#: paged ``chunked_scaled_dot_product_attention`` a ``full`` layer uses for every
#: later chunk.  The paged site can only *halve* it, because that op additionally
#: requires ``chunk_start_idx % q_chunk_size == 0`` — see ``_prefill_sdpa_full``.
#: Must stay ``q_chunk_size == k_chunk_size`` (functional-stage limitation 1).
#:
#: 384 and 512 do not fit L1 (1.93 MB / 2.87 MB of circular buffers against a
#: 1.57 MB budget).  Between 128, 256 and 320, 256 is fastest at 7 of the 9
#: swept lengths on the ``full`` kind and 6 of 9 on ``sliding``, and by a wide
#: margin at the ones that matter: it is the fastest
#: at the 8192-token internal chunk size (8.16 vs 8.60 ms) and at every length
#: below 3008.  320 wins only in a narrow band around 4k (4096: 2.43 vs 2.57 ms;
#: 4128: 2.42 vs 2.59) and by 0.09 % at 8224 on ``sliding``.  A single constant is used rather than
#: a length-dependent rule because that band is two sample points wide and the
#: win there is ~6 % of SDPA, i.e. under 1 % of prefill.  See
#: ``doc/fused_decoder/logs/sdpa_chunk_sweep.log``.
PREFILL_SDPA_CHUNK = 256

#: Largest ``subblock_w`` the sharded LayerNorm program config will use.
MAX_NORM_SUBBLOCK_W = 4

#: Row count at or above which the dense projections switch from ``ttnn.linear``
#: to ``ttnn.experimental.minimal_matmul``.
#:
#: Chosen so the fused layer is **never slower than the functional baseline at
#: any row count**: below the threshold it runs exactly the baseline's kernel.
#: ``ttnn.linear``'s auto-selected program config is not monotone in M (it
#: re-tiles at several points: on the MLP shapes it costs 2.85 ms at 2048 rows,
#: 11.66 at 4096 and 8.98 at 6144), so the crossover is a band rather than a
#: point.  Per-chunk totals from the sweep: at 512-2048 rows ``ttnn.linear``
#: wins overall, at 3072 ``minimal_matmul`` wins by 0.82 ms and from 4096 up it
#: wins by 13-40 ms.  See ``doc/fused_decoder/logs/minimal_matmul_sweep.log``
#: (5 projections x 11 row counts, min of 3 rounds).
MINIMAL_MATMUL_MIN_ROWS = 3072

#: Explicit ``MinimalMatmulConfig`` block sizes, keyed by ``(K, N)`` of the
#: projection, that beat the op's own ``M=K=N=8`` default
#: (``minimal_matmul_program_factory.cpp:22-42``) on **device kernel time**.
#:
#: Value is ``(M_block, K_block, N_block, min_rows)``; the config is used only
#: at ``min_rows`` or more, because both winners were measured to invert on
#: shorter chunks.  Only two of the five projection shapes have one — ``wqkv``
#: and ``mlp_down`` are fastest on the default and get no entry:
#:
#: =========  =============  ======  ======  ======  ======
#: shape      config         @4096   @6144   @8192   shipped
#: =========  =============  ======  ======  ======  ======
#: o_proj     M16 K4 N8      -6.3 %  -6.0 %  +2.8 %  >= 8192
#: mlp g/up   M8 K4 N16      +0.9 %  +1.5 %  +2.9 %  always
#: wqkv       (best cand.)   -       -       -2.6 %  default
#: attn_gate  (best cand.)   -1.3 %  -       -0.0 %  default
#: mlp_down   (best cand.)   -       -       -0.1 %  default
#: =========  =============  ======  ======  ======  ======
#:
#: All five prefill projection shapes were swept; the three with no entry are
#: fastest on the op's own choice.  ``attn_gate``'s best candidate *is* the
#: default (``M8 K8 N8`` re-measures to -0.01 %, i.e. the same kernel).
#:
#: Subblocks follow the op's own rule for ``fp32_dest_acc_en=False``
#: (``2x4`` when ``N >= M`` else ``4x2``), so only the blocking changes.
#: Everything larger was tried and is a hard L1 stop, not an untested guess:
#: ``K_block >= 20``, and every ``M16 * N16`` / ``M16 K8`` / ``N24`` variant,
#: fail with *"Statically allocated circular buffers on core range
#: [0-0 - 10-9] grow to 1.68-2.47 MB which is beyond max L1 size of 1572864 B"*
#: (``program.cpp:1722``).  Measured on the same HiFi2 /
#: ``fp32_dest_acc_en=False`` / ``packer_l1_acc=True`` policy the layer ships,
#: 8 reps per group with the default re-measured between every candidate
#: (default groups reproduce to +-0.1 %).  Host wall-clock cannot resolve these
#: gaps -- the same op A/B'd against *itself* reports -0.5 % to -10.8 % -- so
#: the decision rests on profiler device time.  See
#: ``doc/fused_decoder/logs/prefill_matmul_kblock_{probe,confirm,device,device2,device3,device4}*``.
MINIMAL_MATMUL_BLOCKS = {
    (4096, 6656): (16, 4, 8, 8192),
    (6656, 19968): (8, 4, 16, MINIMAL_MATMUL_MIN_ROWS),
}


def _minimal_matmul_config(rows: int, weight: ttnn.Tensor, grid: ttnn.CoreCoord):
    """The tuned ``MinimalMatmulConfig`` for this projection, or ``None``.

    ``None`` means "let the op choose", which is the right answer for three of
    the five shapes and for every short chunk.
    """
    entry = MINIMAL_MATMUL_BLOCKS.get((weight.shape[-2], weight.shape[-1]))
    if entry is None:
        return None
    m_block, k_block, n_block, min_rows = entry
    if rows < min_rows:
        return None
    subblock_h, subblock_w = (2, 4) if weight.shape[-1] >= rows else (4, 2)
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=grid,
    )


def _dense(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Dense projection, dispatched on row count.

    ``ttnn.experimental.minimal_matmul`` is the same mathematical op as
    ``ttnn.linear`` with a different (and, at prefill shapes, far better) kernel
    — the long-prefill path ``models/common/modules/attention/attention_1d.py``
    opts into for exactly this reason.  At the 8192-row internal prefill chunk
    it is 1.11-2.36x faster on every projection in this layer *and* more
    accurate at the same math fidelity:

    ===========  ==================  ====================  ============
    projection   ttnn.linear @ 8192  minimal_matmul @8192  PCC vs FP32
    ===========  ==================  ====================  ============
    wqkv         2.668 ms            2.331 ms              .999843 -> .999947
    attn_gate    2.273 ms            2.045 ms              .999843 -> .999947
    o_proj       4.876 ms            2.177 ms              .999898 -> .999954
    mlp gate/up  23.657 ms           10.024 ms             .999843 -> .999947
    mlp down     22.617 ms            9.634 ms             .999556 -> .999910
    ===========  ==================  ====================  ============

    Both columns use the same HiFi2 / no-fp32-accumulate compute-kernel config
    that ``ttnn.linear`` picks by default for BF16, so this is a pure kernel
    comparison.  ``minimal_matmul``'s *own* default config is more accurate
    still (PCC .999994) but costs 2.3-2.5 ms more on an 8192-token prefill; it
    is left to the optimized-decoder stage, which owns precision policy.  See
    ``doc/fused_decoder/logs/{prefill_matmul_probe,minimal_matmul_sweep,dense_compute_kernel_probe}.log``
    (the ``attn_gate`` row comes from the crossover sweep, which is the probe
    that covers that shape).

    The ranking inverts below the threshold — at 32 rows ``ttnn.linear`` picks a
    DRAM-sharded 1D config reaching 385 GB/s that ``minimal_matmul`` cannot
    match (0.697 vs 1.057 ms for the MLP gate), and ``mlp_gate_up`` (the widest
    projection, and two of the six dispatches) stays ahead all the way to 2048
    rows while the other four first cross over at 512 and then lose again at
    1536-2048 — so decode and short prefills keep ``ttnn.linear``.  The row count is the whole decision, and it
    cannot mis-fire in decode: ``nlp_create_qkv_heads_decode`` hard-caps
    ``num_users`` at 32 (``..._device_operation.cpp:45-51``), so a decode step is
    always exactly one 32-row tile — two orders of magnitude below the
    threshold.  Both branches are
    PCC-tested and ``test_fused_graph_uses_fused_ops`` asserts which branch each
    prefill length takes.

    Explicit ``MatmulMultiCoreReuseMultiCastProgramConfig`` tilings were the
    first thing tried on the ``SLOW`` rows: seven rectangles (8x{1,2,4,8},
    11x{1,2,4}) on each of the four projections all exceed the L1
    circular-buffer budget (``program.cpp:1722``).  Explicit
    ``MinimalMatmulConfig`` block sizes are swept in
    ``MINIMAL_MATMUL_BLOCKS``, which two of the five projection shapes take;
    the rest pass no ``config=`` and use the op's own choice.  See
    ``doc/fused_decoder/logs/prefill_matmul_probe.log``.
    """
    rows = x.shape[-2]
    if rows >= MINIMAL_MATMUL_MIN_ROWS:
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            memory_config=memory_config,
            dtype=dtype,
            compute_kernel_config=compute_kernel_config,
            config=_minimal_matmul_config(rows, weight, x.device().compute_with_storage_grid_size()),
        )
    return ttnn.linear(x, weight, dtype=dtype, memory_config=memory_config)


def norm_compute_kernel_config(arch):
    """Compute-kernel config for every RMSNorm in this layer.

    **This is the one place the fused layer changes numerical fidelity rather
    than topology**, and it is a deliberate, measured choice rather than a
    default that came along for the ride.

    ``ttnn.rms_norm``'s own default is
    ``HiFi4 / math_approx_mode=True / fp32_dest_acc_en=False /
    packer_l1_acc=False`` (``rmsnorm.cpp:16-20``), which is what the functional
    layer used because it passed no config at all.  This turns the approximate
    reciprocal-sqrt off and FP32 destination accumulation on for a 6656-wide
    BF16 reduction.  Measured in isolation against a float64 reference
    (``doc/fused_decoder/bench/norm_fidelity_probe.py``):

    ========  ==========  ==========  ==================  ==============
    shape     op default  this one    PCC vs f64          max rel. err
    ========  ==========  ==========  ==================  ==============
    prefill   978.27 us   991.78 us   .999928 -> .999998  6.5e-2 -> 4.2e-3
    decode     15.53 us    14.92 us   .999994 -> .999998  1.0e-2 -> 4.8e-3
    ========  ==========  ==========  ==================  ==============

    So it is **free in decode** (the sharded kernel is 3.9 % *faster* with it)
    and costs 13.5 us per prefill norm -- 81 us of a 49,285 us prefill window,
    0.16 % -- for a 15x smaller worst-case error on the op that feeds every
    matmul in the layer.

    It is also worth about **+3.5e-4** of the fused graph's HF-vs-unfused
    accuracy gain at short prefill lengths, where no matmul kernel changes.
    ``doc/fused_decoder/logs/norm_fidelity_control.log`` is the same graph with
    the norms on the op default: the 100-token prefill controls go to -8e-6 and
    +0.0 there, against +3.6e-4 and +3.2e-4 shipped.  That control is why the
    README does not claim the accuracy gain is topology alone.
    """
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi4,  # the op's default fidelity, unchanged
        math_approx_mode=False,  # op default: True
        fp32_dest_acc_en=True,  # op default: False
        packer_l1_acc=True,  # op default: False
    )


def _norm_subblock_w(block_w: int) -> int:
    """Largest divisor of ``block_w`` that is ``<= MAX_NORM_SUBBLOCK_W``."""
    for candidate in range(min(MAX_NORM_SUBBLOCK_W, block_w), 0, -1):
        if block_w % candidate == 0:
            return candidate
    return 1


#: Measured optimum for the width-sharded decode RMSNorm on Blackhole.  A decode
#: step is a *single tile-row*, so past ~8 cores the per-core reduction and the
#: cross-core stats exchange cost more than the extra width parallelism buys:
#: 1c (interleaved) 134.4 us, 4c 29.3, **8c 22.8**, 13c 25.3, 16c 24.4, 26c 28.0,
#: 52c 37.7, 104c 57.6 (``doc/fused_decoder/logs/norm_shard_probe.log``, min of
#: 3 rounds x 200 calls each).
DECODE_NORM_TARGET_CORES = 8


def choose_decode_norm_grid(dim: int, grid: ttnn.CoreCoord) -> tuple[int, int]:
    """Core rectangle for the width-sharded decode RMSNorm over ``dim``.

    The core count must divide ``dim // 32`` for the shard width to stay
    tile-aligned.  For ``dim = 6656`` that is ``208 = 2^4 * 13`` tiles, so the
    legal counts are 1, 2, 4, 8, 13, 16, 26, 52, 104, 208.

    13 (and its multiples) have no rectangle on an 11-wide grid, but that is
    *not* why they are unused: the sharded LayerNorm program factory explicitly
    accepts a **non-rectangular** ``CoreRangeSet`` when the whole height fits on
    one core and the grid is a shard-order prefix of its bounding box
    (``layernorm_device_operation.cpp:185-215``), which is always true for a
    decode step.  13/26/52/104-core prefix grids were built with
    ``ttnn.num_cores_to_corerangeset_in_subcoregrids`` and measured — all four
    are legal, correct (PCC 0.9999964) and *slower* than 8 cores, even at
    ``subblock_w = 4``.  See ``DECODE_NORM_TARGET_CORES``.

    Selection: the most cores at or below the measured optimum that still leave
    ``subblock_w >= 2``, then the squarest rectangle, then the wider one (the
    three 8-core rectangles measured 22.8 / 23.0 / 23.4 us for 4x2 / 8x1 / 2x4).
    On Blackhole that is ``4 x 2`` (8 cores, ``block_w = 26``,
    ``subblock_w = 2``).
    """
    tiles = dim // TILE_SIZE
    best: tuple[int, int] | None = None
    best_key: tuple[int, int] | None = None
    for gx in range(1, grid.x + 1):
        for gy in range(1, grid.y + 1):
            cores = gx * gy
            if cores > DECODE_NORM_TARGET_CORES or tiles % cores:
                continue
            block_w = tiles // cores
            if _norm_subblock_w(block_w) < 2:
                continue
            key = (cores, -abs(gx - gy), gx)
            if best_key is None or key > best_key:
                best_key, best = key, (gx, gy)
    return best or (1, 1)


class _FusedMLP(LightweightModule):
    """SwiGLU MLP with ``silu`` folded into the gating multiply.

    ``down(silu(gate(x)) * up(x))`` is five device ops in the functional layer
    (gate matmul, up matmul, ``ttnn.silu``, ``ttnn.mul``, down matmul); folding
    the activation into the binary op's input makes it four.

    Two other spellings were measured and rejected:

    * ``ttnn.linear(x, w_gate, activation="silu")`` — the matmul's *pack-time*
      activation.  On this build it does not fuse at all for these shapes: the
      profiler still shows a separate 2,128 us ``UnaryDeviceOperation``
      (``doc/fused_decoder/logs/rejected/prefill_perf_report_matmul_activation_*.txt``,
      op id 97), and measured in isolation the matmul itself also slows,
      23.964 -> 26.461 ms.  The two capture rows for that matmul (22,471 and
      23,257 us) are *not* evidence of the slowdown -- the functional baseline
      shows the same ~770 us spread between the same two graph positions with
      no activation at all -- so the rejection rests on the surviving unary op
      and the isolated measurement.  Strictly worse than doing nothing.
    * ``minimal_matmul(..., fused_activation=SILU)`` — the same idea on the
      kernel prefill actually uses.  It does fuse, but costs 12.101 vs
      10.283 ms on this shape (``logs/prefill_matmul_probe.log``).
    * ``ttnn.swiglu`` over a packed ``[up | gate]`` projection — a *composite*
      (two slices + swish + multiply), so it adds ops rather than removing them.

    The kept form is the skill's "input-arg activation -> eltwise binary"
    merge, which really is one kernel: 2.539 ms vs 4.458 ms for
    ``silu`` + ``mul`` at 8192x19968, identical PCC
    (``doc/fused_decoder/logs/op_merge_probes.log``).
    """

    def __init__(
        self,
        gate: ttnn.Tensor,
        up: ttnn.Tensor,
        down: ttnn.Tensor,
        activation_dtype: ttnn.DataType,
        compute_kernel_config=None,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down
        self.activation_dtype = activation_dtype
        self.compute_kernel_config = compute_kernel_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        kwargs = dict(
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        gate = _dense(x, self.gate, **kwargs)
        up = _dense(x, self.up, **kwargs)
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = _dense(hidden, self.down, **kwargs)
        ttnn.deallocate(hidden)
        return out


class _FusedNorm(LightweightModule):
    """Centered RMSNorm with both an interleaved and a width-sharded form.

    Prefill keeps the interleaved kernel (already 110-core and DRAM-bandwidth
    bound at prefill shapes).  Decode uses the sharded multi-core program
    config, which is the whole point: the interleaved kernel parallelises over
    *rows*, and a decode step is a single tile-row, so it lands on one core.
    """

    def __init__(
        self,
        weight_tile: ttnn.Tensor,
        weight_row_major: ttnn.Tensor,
        eps: float,
        compute_kernel_config: Any,
    ) -> None:
        super().__init__()
        self.weight = weight_tile
        self.weight_rm = weight_row_major
        self.eps = eps
        self.compute_kernel_config = compute_kernel_config

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Interleaved (prefill) RMSNorm."""
        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

    def sharded_forward(self, x_sharded: ttnn.Tensor, program_config, memory_config) -> ttnn.Tensor:
        """Width-sharded (decode) RMSNorm; input and output stay in L1."""
        return ttnn.rms_norm(
            x_sharded,
            weight=self.weight_rm,
            epsilon=self.eps,
            program_config=program_config,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )


class FusedDecoder(FunctionalDecoder):
    """Fused TTNN implementation of ``MuseGlimmerTextDecoderLayer``."""

    def __init__(
        self,
        *,
        cos_cache_tile: ttnn.Tensor | None,
        sin_cache_tile: ttnn.Tensor | None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        #: Pre-tilized ``[1, 1, max_seq_len, head_dim]`` prefill RoPE tables.
        #: ``cos_cache`` / ``sin_cache`` (inherited) stay ROW_MAJOR 2-D for the
        #: decode-time ``ttnn.embedding`` gather.
        self.cos_cache_tile = cos_cache_tile
        self.sin_cache_tile = sin_cache_tile

        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        #: Compute-kernel config for the prefill ``minimal_matmul`` dispatches.
        #: Pinned to the *same* HiFi2 / no-fp32-accumulate policy ``ttnn.linear``
        #: selects by default for BF16 inputs, so the fusing stage's before/after
        #: numbers compare like with like and no precision decision is smuggled
        #: into a topology change.  ``minimal_matmul``'s own default is more
        #: accurate (PCC 0.999994 vs 0.999947 against FP32) but measurably
        #: slower; see the work log and ``logs/minimal_matmul_sweep.log``.
        self.dense_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        if getattr(self.mlp, "compute_kernel_config", None) is None:
            self.mlp.compute_kernel_config = self.dense_compute_kernel_config
        self.norm_compute_kernel_config = norm_compute_kernel_config(self.mesh_device.arch())
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.decode_norm_grid = choose_decode_norm_grid(self.config.hidden_size, grid)
        self._decode_norm_cache: dict[int, tuple[Any, ttnn.MemoryConfig]] = {}

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        page_block_size: int = 64,
        max_num_blocks: int | None = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat16,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        **kwargs,
    ) -> "FusedDecoder":
        """Same contract as ``FunctionalDecoder.from_state_dict``.

        All ``torch`` / ``ttnn.from_torch`` work happens here; the runtime path
        is TTNN-only.
        """
        if kwargs:
            raise TypeError(f"Unexpected FusedDecoder.from_state_dict kwargs: {sorted(kwargs)}")
        if mesh_device.get_num_devices() != 1:
            raise ValueError("FusedDecoder is the single-chip stage; use a 1x1 MeshDevice.")

        text_config = _text_config(hf_config)
        _require_muse_glimmer_text_config(text_config)
        layer_kind = resolve_layer_kind(hf_config, layer_idx)

        max_seq_len = int(max_seq_len or text_config.max_position_embeddings)
        if max_seq_len > text_config.max_position_embeddings:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds the HF-advertised context {text_config.max_position_embeddings}"
            )
        if page_block_size % TILE_SIZE != 0:
            raise ValueError(f"page_block_size must be a multiple of {TILE_SIZE}, got {page_block_size}")
        if max_seq_len % TILE_SIZE != 0:
            raise ValueError(f"max_seq_len must be a multiple of {TILE_SIZE}, got {max_seq_len}")
        blocks_per_seq = (max_seq_len + page_block_size - 1) // page_block_size
        if max_num_blocks is None:
            max_num_blocks = max_batch_size * blocks_per_seq
        if max_num_blocks < max_batch_size * blocks_per_seq:
            raise ValueError(
                f"max_num_blocks={max_num_blocks} cannot hold max_batch_size={max_batch_size} x "
                f"{blocks_per_seq} blocks of {page_block_size} tokens"
            )
        if prefill_chunk_size % page_block_size or prefill_chunk_size % TILE_SIZE:
            raise ValueError(
                f"prefill_chunk_size={prefill_chunk_size} must be a multiple of the page block size "
                f"({page_block_size}) and the tile height ({TILE_SIZE})"
            )
        if prefill_chunk_size > PREFILL_SDPA_MAX_SEQ // 2:
            raise ValueError(
                f"prefill_chunk_size={prefill_chunk_size} is too large: the sliding-window prefill "
                f"slice (chunk + window) must stay below the {PREFILL_SDPA_MAX_SEQ}-token SDPA limit"
            )

        config = MuseGlimmerLayerConfig(
            layer_idx=layer_idx,
            layer_kind=layer_kind,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            rms_norm_eps=text_config.rms_norm_eps,
            post_norm_eps=text_config.post_norm_eps,
            qk_scale_factor=text_config.qk_scale_factor,
            sliding_window=text_config.sliding_window if layer_kind == LAYER_KIND_SLIDING else None,
            rope_theta=(float(text_config.layer_rope_theta[layer_idx]) if layer_kind == LAYER_KIND_SLIDING else None),
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            paged_attention_config=PagedAttentionConfig(
                block_size=page_block_size,
                max_num_blocks=max_num_blocks,
            ),
            prefill_chunk_size=prefill_chunk_size,
        )

        norm_ck = norm_compute_kernel_config(mesh_device.arch())

        def norm(name: str, eps: float) -> _FusedNorm:
            weight = _get_layer_tensor(state_dict, layer_idx, f"{name}.weight").to(torch.float32)
            # Centered RMSNorm multiplies by (1 + w); fold the +1 in at setup.
            folded = (1.0 + weight).to(torch.bfloat16)
            tile = _to_device(folded.reshape(1, 1, 1, config.hidden_size), mesh_device=mesh_device, dtype=ttnn.bfloat16)
            # The sharded LayerNorm program factory wants gamma ROW_MAJOR as
            # [1, 1, dim // TILE_SIZE, TILE_SIZE] (one tile-wide row per shard
            # column), not the interleaved [1, 1, 1, dim] tile form.
            row_major = _to_device(
                folded.reshape(1, 1, config.hidden_size // TILE_SIZE, TILE_SIZE),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            return _FusedNorm(tile, row_major, eps, norm_ck)

        def linear_weight(suffix: str) -> torch.Tensor:
            # HF stores nn.Linear weights as [out, in]; ttnn.linear wants [in, out].
            return _get_layer_tensor(state_dict, layer_idx, suffix).to(torch.float32).transpose(-2, -1).contiguous()

        wq = linear_weight("self_attn.q_proj.weight")
        wk = linear_weight("self_attn.k_proj.weight")
        wv = linear_weight("self_attn.v_proj.weight")
        wqkv = torch.cat([wq, wk, wv], dim=-1).unsqueeze(0).unsqueeze(0)
        w_attn_gate = linear_weight("self_attn.gate_proj.weight").unsqueeze(0).unsqueeze(0)
        wo = linear_weight("self_attn.o_proj.weight").unsqueeze(0).unsqueeze(0)

        mlp = _FusedMLP(
            gate=_to_device(
                linear_weight("mlp.gate_proj.weight").unsqueeze(0).unsqueeze(0),
                mesh_device=mesh_device,
                dtype=weight_dtype,
            ),
            up=_to_device(
                linear_weight("mlp.up_proj.weight").unsqueeze(0).unsqueeze(0),
                mesh_device=mesh_device,
                dtype=weight_dtype,
            ),
            down=_to_device(
                linear_weight("mlp.down_proj.weight").unsqueeze(0).unsqueeze(0),
                mesh_device=mesh_device,
                dtype=weight_dtype,
            ),
            activation_dtype=activation_dtype,
        )

        cache_shape = (max_num_blocks, config.num_key_value_heads, page_block_size, config.head_dim)
        k_cache = _to_device(torch.zeros(cache_shape), mesh_device=mesh_device, dtype=kv_cache_dtype)
        v_cache = _to_device(torch.zeros(cache_shape), mesh_device=mesh_device, dtype=kv_cache_dtype)

        cos_cache = sin_cache = cos_cache_tile = sin_cache_tile = None
        if config.uses_rope:
            cos, sin = _rope_cos_sin(max_seq_len, config.head_dim, config.rope_theta)
            # 2-D ROW_MAJOR tables for the decode-time on-device row gather.
            cos_cache = _to_device(
                cos.to(torch.bfloat16), mesh_device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            sin_cache = _to_device(
                sin.to(torch.bfloat16), mesh_device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            # Pre-tilized 4-D tables for prefill: rotary_embedding_hf wants TILE
            # layout, and tilizing a [chunk, head_dim] slice at runtime was two
            # extra device ops per chunk in the functional layer.
            cos_cache_tile = _to_device(
                cos.to(torch.bfloat16).reshape(1, 1, max_seq_len, config.head_dim),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
            )
            sin_cache_tile = _to_device(
                sin.to(torch.bfloat16).reshape(1, 1, max_seq_len, config.head_dim),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
            )

        return cls(
            config=config,
            mesh_device=mesh_device,
            input_layernorm=norm("input_layernorm", config.rms_norm_eps),
            post_attention_layernorm=norm("post_attention_layernorm", config.post_norm_eps),
            pre_feedforward_layernorm=norm("pre_feedforward_layernorm", config.rms_norm_eps),
            post_feedforward_layernorm=norm("post_feedforward_layernorm", config.post_norm_eps),
            mlp=mlp,
            wqkv=_to_device(wqkv, mesh_device=mesh_device, dtype=weight_dtype),
            w_attn_gate=_to_device(w_attn_gate, mesh_device=mesh_device, dtype=weight_dtype),
            wo=_to_device(wo, mesh_device=mesh_device, dtype=weight_dtype),
            k_cache=k_cache,
            v_cache=v_cache,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            cos_cache_tile=cos_cache_tile,
            sin_cache_tile=sin_cache_tile,
            activation_dtype=activation_dtype,
            kv_cache_dtype=kv_cache_dtype,
        )

    # ------------------------------------------------------------ prefill path

    def _prefill_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Prefill SDPA chunking, clamped to the (tile-padded) slice length.

        ``q_chunk_size == k_chunk_size`` is mandatory (functional-stage
        limitation 1: ``q == 2 * k`` silently mis-masks the sliding window).
        The *size* is 256 rather than the functional layer's 128 — measured
        1.5x faster at the 8192-token internal chunk for both kinds with
        identical PCC, and still correct at the lengths that expose the chunk
        bug (2080 / 4128 / 8224).  See ``PREFILL_SDPA_CHUNK`` for why 256 and
        not 320.
        """
        padded = ((seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        chunk = max(TILE_SIZE, min(PREFILL_SDPA_CHUNK, padded))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.prefill_sdpa_grid,
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )

    def _prefill_rope_tables(self, start_pos: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor, bool]:
        """Pre-tilized cos/sin for ``[start_pos, start_pos + length)``.

        ``rotary_embedding_hf`` only requires ``cos_seq_len >= seq_len`` and
        reads rows ``[0, seq_len)``, so a chunk at absolute position 0 can use
        the persistent table directly — no slice, no tilize.  Later chunks slice
        the tiled table; ``start_pos`` is always a page-block (hence tile)
        multiple, so the slice is tile-aligned.

        Returns ``(cos, sin, owned)``; ``owned`` is ``False`` when the
        persistent tables are handed back and must not be deallocated.
        """
        head_dim = self.config.head_dim
        if start_pos == 0:
            return self.cos_cache_tile, self.sin_cache_tile, False
        cos = ttnn.slice(self.cos_cache_tile, [0, 0, start_pos, 0], [1, 1, start_pos + length, head_dim])
        sin = ttnn.slice(self.sin_cache_tile, [0, 0, start_pos, 0], [1, 1, start_pos + length, head_dim])
        return cos, sin, True

    def chunked_sdpa_chunk_size(self, start_pos: int, seq_len: int, blocks_per_seq: int) -> int:
        """q/k chunk for the paged prefill SDPA at this offset.

        Seeded from ``PREFILL_SDPA_CHUNK`` and halved until both of the op's
        constraints hold at once:

        * ``chunk_start_idx % q_chunk_size == 0``
          (``sdpa_device_operation.cpp:355``);
        * ``kv_length >= padded_q + chunk_start_idx`` (``:366-374``), where
          ``kv_length`` is derived from the **page table's** width — so the Q
          padding a chunk size implies must not reach past the end of the user's
          pages.  This is the one that bites when the chunk is *raised*: a
          ``max_seq_len`` whose block count is not a multiple of
          ``PREFILL_SDPA_CHUNK / block_size`` would overrun.  Pinned by
          ``test_multi_chunk_prefill_page_table_bound`` at
          ``max_seq_len = 12416`` (194 blocks), where 256 overruns by 128 tokens
          and 128 fits exactly.

        Halving always terminates at ``TILE_SIZE``, where the padding is zero
        (the caller already tile-padded) and ``start_pos + seq_len <=
        max_seq_len <= kv_length`` holds by construction.
        """
        kv_length = blocks_per_seq * self.config.paged_attention_config.block_size
        chunk = PREFILL_SDPA_CHUNK
        while chunk > TILE_SIZE and (start_pos % chunk or start_pos + -(-seq_len // chunk) * chunk > kv_length):
            chunk //= 2
        if start_pos % chunk:
            raise ValueError(
                f"continuation prefill start_pos={start_pos} is not a multiple of the minimum SDPA "
                f"chunk size {TILE_SIZE}"
            )
        return chunk

    def _prefill_sdpa_full(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        page_table: ttnn.Tensor,
        user_id: int,
        start_pos: int,
    ) -> ttnn.Tensor:
        """Full-attention prefill SDPA, with the *paged* path retuned too.

        Chunk 0 is an in-memory square SDPA and picks up
        ``_prefill_program_config`` automatically.  Every later chunk instead
        reads the whole prefix back out of the paged cache with
        ``chunked_scaled_dot_product_attention``, which builds its own program
        config — the functional layer hard-coded 128 there, so the chunk retune
        would have missed the dominant op of every prefill longer than one
        internal chunk.  This override seeds it from ``PREFILL_SDPA_CHUNK``
        instead and keeps the halving loop, which exists because the op requires
        ``chunk_start_idx % q_chunk_size == 0`` and a caller-level continuation
        may start at any page-block multiple.

        The chunk is a *seed*, not a fixed value: the op requires both
        ``chunk_start_idx % q_chunk_size == 0`` and
        ``kv_length >= padded_q + chunk_start_idx``, and ``kv_length`` is derived
        from the page table's width — so a larger chunk pads Q further and can
        reach past the end of the user's pages.  Both conditions are checked in
        the halving loop below.

        Measured on the real op at the offsets an 8192-chunked prefill produces
        (``doc/fused_decoder/logs/chunked_sdpa_sweep.log``): 36.204 -> 22.831 ms
        at ``chunk_start_idx=8192`` and 109.992 -> 72.277 ms at 32768, i.e.
        **1.59x**, with PCC improving as well (0.99978 -> 0.99982 and
        0.99965 -> 0.99975 against a torch masked-softmax reference over the
        same permuted paged cache).  512 does not fit L1; 320 cannot be used at
        all because it must divide ``chunk_start_idx``, which is a multiple of
        the 8192 prefill chunk.
        """
        cfg = self.config
        if start_pos == 0:
            return ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=cfg.sdpa_scale,
                program_config=self._prefill_program_config(q.shape[-2]),
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
        seq_len = q.shape[-2]
        n_heads = cfg.num_attention_heads
        head_dim = cfg.head_dim
        # Two constraints shrink the chunk, and both must hold at once:
        #   * chunk_start_idx % q_chunk_size == 0 (sdpa_device_operation.cpp:355);
        #   * kv_length >= padded_q + chunk_start_idx (:366-374), where kv_length
        #     comes from the *page table's* width, so the Q padding this chunk
        #     size implies must not reach past the end of the user's pages.
        # The second is why the chunk size cannot simply be raised: a bigger
        # chunk pads further, and a max_seq_len whose block count is not a
        # multiple of PREFILL_SDPA_CHUNK / block_size would overrun.  Halving
        # always terminates at TILE_SIZE, where the padding is zero (the caller
        # already tile-padded) and start_pos + seq_len <= max_seq_len holds.
        chunked_q = self.chunked_sdpa_chunk_size(start_pos, seq_len, page_table.shape[-1])
        pad = (-seq_len) % chunked_q
        q_in = q
        if pad:
            q_in = ttnn.pad(q, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
        user_pt, owns_user_pt = self._page_table_row(page_table, user_id, 0, page_table.shape[-1])
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_in,
            self.k_cache,
            self.v_cache,
            user_pt,
            start_pos,  # chunk_start_idx is positional-only in the ttnn binding
            scale=cfg.sdpa_scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.prefill_sdpa_grid,
                q_chunk_size=chunked_q,
                k_chunk_size=chunked_q,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        if owns_user_pt:
            ttnn.deallocate(user_pt)
        if pad:
            ttnn.deallocate(q_in)
            trimmed = ttnn.slice(out, [0, 0, 0, 0], [1, n_heads, seq_len, head_dim])
            ttnn.deallocate(out)
            out = trimmed
        return out

    # The QKV projection and the attention output-gate projection share their
    # left-hand side (the input_layernorm output), so they are the two halves of
    # a possible shared-LHS packing.  Keeping them behind these two seams is what
    # let the packed variant be measured against the shipped split form without
    # forking the whole forward; see ``doc/fused_decoder/bench/variants.py``.

    def _project_qkv(self, normed: ttnn.Tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        return _dense(
            normed,
            self.wqkv,
            dtype=self.activation_dtype,
            memory_config=memory_config,
            compute_kernel_config=self.dense_compute_kernel_config,
        )

    def _attn_gate(self, normed: ttnn.Tensor) -> ttnn.Tensor:
        """The raw attention output-gate projection.

        The ``sigmoid`` is applied by the gating ``ttnn.mul`` as an input
        activation, not here and not as the matmul's pack-time activation —
        see ``_FusedMLP`` for why the matmul form was rejected.
        """
        return _dense(
            normed,
            self.w_attn_gate,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.dense_compute_kernel_config,
        )

    def _prefill_attention(
        self,
        normed: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        user_id: int,
        start_pos: int,
        sliding_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        need_tail: bool,
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor] | None]:
        cfg = self.config
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads

        xqkv = self._project_qkv(normed)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        q_normed = self._per_head_rmsnorm(q)
        ttnn.deallocate(q)
        k_normed = self._per_head_rmsnorm(k)
        ttnn.deallocate(k)
        q, k = q_normed, k_normed

        if cfg.uses_rope:
            cos, sin, owns_tables = self._prefill_rope_tables(start_pos, q.shape[-2])
            q_rot = ttnn.experimental.rotary_embedding_hf(
                q, cos, sin, is_decode_mode=False, compute_kernel_config=self.rope_compute_kernel_config
            )
            ttnn.deallocate(q)
            k_rot = ttnn.experimental.rotary_embedding_hf(
                k, cos, sin, is_decode_mode=False, compute_kernel_config=self.rope_compute_kernel_config
            )
            ttnn.deallocate(k)
            if owns_tables:
                ttnn.deallocate(cos)
                ttnn.deallocate(sin)
            q, k = q_rot, k_rot

        # Paged KV fill.  ``paged_fill_cache`` does no dtype conversion, so cast
        # to the cache dtype first (decode's update op owns its own repack).
        seq_len = q.shape[-2]
        block_size = cfg.paged_attention_config.block_size
        k_fill = k if k.dtype == self.kv_cache_dtype else ttnn.typecast(k, self.kv_cache_dtype)
        v_fill = v if v.dtype == self.kv_cache_dtype else ttnn.typecast(v, self.kv_cache_dtype)
        chunk_page_table, owns_chunk_pt = self._chunk_page_table(page_table, user_id, start_pos, seq_len)
        ttnn.experimental.paged_fill_cache(self.k_cache, k_fill, chunk_page_table, batch_idx=0, block_size=block_size)
        ttnn.experimental.paged_fill_cache(self.v_cache, v_fill, chunk_page_table, batch_idx=0, block_size=block_size)
        if owns_chunk_pt:
            ttnn.deallocate(chunk_page_table)
        if k_fill is not k:
            ttnn.deallocate(k_fill)
        if v_fill is not v:
            ttnn.deallocate(v_fill)

        next_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None = None
        if cfg.is_sliding:
            attn, next_tail = self._prefill_sdpa_sliding(q, k, v, sliding_tail, need_tail)
        else:
            attn = self._prefill_sdpa_full(q, k, v, page_table, user_id, start_pos)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        out = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        gate = self._attn_gate(normed)
        gated = ttnn.mul(
            out,
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(out)
        ttnn.deallocate(gate)
        projected = _dense(
            gated,
            self.wo,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.dense_compute_kernel_config,
        )
        ttnn.deallocate(gated)
        return projected, next_tail

    # ------------------------------------------------------------- decode path

    def _create_qkv_heads_decode(self, xqkv_l1: ttnn.Tensor):
        """Split the fused QKV into decode-layout head tensors.

        ``overlap_qk_coregrid`` is left at its default (``True``), so Q, K and V
        all land on the same one-core-per-user grid and K's cos/sin can be
        shared with Q's.  (The decode batch ceiling is the op's own hard
        ``num_users <= 32``, ``..._device_operation.cpp:45-51``, not the grid.)
        The alternative is measured in ``bench/variants.py`` — see
        ``_decode_kv_update``.
        """
        cfg = self.config
        return ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_l1,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

    def _decode_norm_configs(self, rows: int):
        """``(program_config, width-sharded memory_config)`` for ``rows`` tokens."""
        cached = self._decode_norm_cache.get(rows)
        if cached is not None:
            return cached
        dim = self.config.hidden_size
        gx, gy = self.decode_norm_grid
        cores = gx * gy
        block_w = dim // cores // TILE_SIZE
        memory_config = ttnn.create_sharded_memory_config(
            shape=(rows, dim // cores),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=_norm_subblock_w(block_w),
            block_h=rows // TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )
        self._decode_norm_cache[rows] = (program_config, memory_config)
        return program_config, memory_config

    def _decode_rope_tables(self, rope_pos_ids: ttnn.Tensor, shard_spec_source: ttnn.Tensor):
        """Per-user cos/sin gathered straight into the decode RoPE layout.

        ``rotary_embedding_hf(is_decode_mode=True)`` wants height-sharded
        ``[1, batch, 1, head_dim]`` cos/sin on the same one-core-per-user grid
        the Q/K head tensors already live on, so the gather ends in exactly the
        layout the kernel reads — no ``ttnn.repeat`` broadcast, no
        untilize/tilize round trip.

        One tolerance is deliberately *not* carried over.  The functional layer
        trimmed ``cos_b[:, :batch]`` when the gather came back longer than
        ``batch`` (``functional_decoder.py:1076-1078``), which accepted a
        tile-padded ``rope_pos_ids``.  The documented contract is
        ``[1, batch]`` — that is what ``decode_position_tensors`` builds and what
        every caller passes — and here the gathered rows are sharded one-per-user
        onto the Q grid, so a longer ``rope_pos_ids`` would not silently
        mis-broadcast: it fails in the shard/RoPE validation instead.  Accepting
        a padded input again would mean re-adding the slice *and* re-deriving the
        shard grid from ``batch`` rather than from Q.
        """
        head_dim = self.config.head_dim
        shard = shard_spec_source.memory_config().shard_spec
        cos_sin_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard.grid, (TILE_SIZE, head_dim), ttnn.ShardOrientation.ROW_MAJOR),
        )

        def gather(table: ttnn.Tensor) -> ttnn.Tensor:
            # rope_pos_ids is exactly [1, batch], so the gather is already
            # batch-sized and needs no trim (models/common/modules/rope/rope_1d.py
            # slices here only because its rot_idxs may be tile-padded).
            rows = ttnn.unsqueeze_to_4D(ttnn.embedding(rope_pos_ids, table, layout=ttnn.TILE_LAYOUT))
            per_user = ttnn.transpose(rows, 1, 2)  # [1, 1, batch, d] -> [1, batch, 1, d]
            ttnn.deallocate(rows)
            sharded = ttnn.interleaved_to_sharded(per_user, cos_sin_memcfg)
            ttnn.deallocate(per_user)
            return sharded

        return gather(self.cos_cache), gather(self.sin_cache)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        rope_pos_ids: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Single-token paged decode; see ``FunctionalDecoder`` for the contract."""
        cfg = self.config
        batch = int(hidden_states.shape[-2])
        if hidden_states.shape[-1] != cfg.hidden_size:
            raise ValueError(f"decode expects hidden size {cfg.hidden_size}, got {hidden_states.shape[-1]}")
        if cfg.uses_rope and rope_pos_ids is None:
            raise ValueError("sliding (RoPE) layers require rope_pos_ids for the on-device cos/sin gather")

        rows = ((batch + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        norm_prg, norm_memcfg = self._decode_norm_configs(rows)

        # The residual stream stays width-sharded in L1 for the whole layer, so
        # every hidden_size-wide RMSNorm runs multi-core and the two residual
        # adds are sharded element-wise ops.
        residual = ttnn.interleaved_to_sharded(hidden_states, norm_memcfg)
        normed_sharded = self.input_layernorm.sharded_forward(residual, norm_prg, norm_memcfg)
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed_sharded)

        # nlp_create_qkv_heads_decode's interleaved DRAM reader zeroes odd Q rows
        # on Blackhole (tt-metal #16667), so the fused QKV has to be staged in
        # L1.  The projection writes there directly instead of the functional
        # layer's DRAM matmul + ``to_memory_config`` copy.
        xqkv_l1 = self._project_qkv(normed, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k, v = self._create_qkv_heads_decode(xqkv_l1)
        ttnn.deallocate(xqkv_l1)
        q_memcfg = q.memory_config()
        kv_memcfg = k.memory_config()

        # ttnn.rms_norm rejects height-sharded inputs, so the scale-less
        # per-head norm round-trips through L1 interleaved (not DRAM).
        q = self._sharded_per_head_rmsnorm(q, q_memcfg)
        k = self._sharded_per_head_rmsnorm(k, kv_memcfg)

        if cfg.uses_rope:
            cos_q, sin_q = self._decode_rope_tables(rope_pos_ids, q)
            q_rot = ttnn.experimental.rotary_embedding_hf(
                q, cos_q, sin_q, is_decode_mode=True, compute_kernel_config=self.rope_compute_kernel_config
            )
            ttnn.deallocate(q)
            k_rot = ttnn.experimental.rotary_embedding_hf(
                k, cos_q, sin_q, is_decode_mode=True, compute_kernel_config=self.rope_compute_kernel_config
            )
            ttnn.deallocate(k)
            ttnn.deallocate(cos_q)
            ttnn.deallocate(sin_q)
            q, k = q_rot, k_rot

        self._decode_kv_update(k, v, current_pos, page_table)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=cfg.sdpa_scale,
            sliding_window_size=cfg.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_sdpa_program_config,
        )
        ttnn.deallocate(q)

        out = self._concat_heads_decode(attn, batch)
        ttnn.deallocate(attn)

        gate = self._attn_gate(normed)
        ttnn.deallocate(normed)
        gated = ttnn.mul(
            out,
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(out)
        ttnn.deallocate(gate)
        attn_out = _dense(
            gated,
            self.wo,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.dense_compute_kernel_config,
        )
        ttnn.deallocate(gated)

        attn_sharded = ttnn.interleaved_to_sharded(attn_out, norm_memcfg)
        ttnn.deallocate(attn_out)
        attn_normed = self.post_attention_layernorm.sharded_forward(attn_sharded, norm_prg, norm_memcfg)
        ttnn.deallocate(attn_sharded)
        hidden = ttnn.add(residual, attn_normed, memory_config=norm_memcfg)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_normed)

        mlp_in_sharded = self.pre_feedforward_layernorm.sharded_forward(hidden, norm_prg, norm_memcfg)
        mlp_in = ttnn.sharded_to_interleaved(mlp_in_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_in_sharded)
        mlp_out = self.mlp(mlp_in)
        ttnn.deallocate(mlp_in)

        mlp_sharded = ttnn.interleaved_to_sharded(mlp_out, norm_memcfg)
        ttnn.deallocate(mlp_out)
        mlp_normed = self.post_feedforward_layernorm.sharded_forward(mlp_sharded, norm_prg, norm_memcfg)
        ttnn.deallocate(mlp_sharded)
        out_sharded = ttnn.add(hidden, mlp_normed, memory_config=norm_memcfg)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_normed)
        out = ttnn.sharded_to_interleaved(out_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_sharded)
        return out

    def _decode_kv_update(
        self, k: ttnn.Tensor, v: ttnn.Tensor, current_pos: ttnn.Tensor, page_table: ttnn.Tensor
    ) -> None:
        """Write this step's K and V into the paged cache, as two calls.

        ``ttnn.experimental.paged_fused_update_cache`` would do both in one
        kernel, but it asserts its two update tensors live on **disjoint** cores
        (``paged_fused_update_cache_device_operation.cpp:341-348``), and
        ``nlp_create_qkv_heads_decode`` emits K and V on Q's grid.

        Its ``overlap_qk_coregrid=False`` mode does move K to a disjoint grid,
        but it is unreachable from here.  The frontend drops the flag entirely
        for an *interleaved* input (``nlp_create_qkv_heads_decode.cpp:23``:
        ``input_tensor.is_sharded() ? overlap_qk_coregrid.value_or(true) : true``),
        and this layer's decode QKV projection is L1 interleaved — which is what
        the op needs after the Blackhole tt-metal #16667 workaround.  A sharded
        input keeps the flag, but the device op then also requires the shard to
        hold the full height on one core *and* ``head_dim % shard_width == 0``
        (``..._device_operation.cpp:56-72``), which together admit only a
        width-sharded QKV with a 32/64/128-wide shard.
        Measured, not just read: with an L1-interleaved input the flag changes
        nothing (identical Q/K/V grids at batch 1, 4 and 32) and the fused cache
        write is rejected at every batch; with a WIDTH_SHARDED input and a shard
        width dividing ``head_dim`` the same call *does* produce disjoint K and V.
        A width-sharded decode QKV is the DRAM-sharded matmul, i.e. the
        optimized-decoder stage.  It would also bring
        ``num_cores >= 2 * num_users``
        (``..._device_operation.cpp:102-111``), which is not binding here — the
        op already caps ``num_users`` at 32 and this grid has 110 cores — but is
        one more constraint the next stage would inherit.  Evidence:
        ``doc/fused_decoder/logs/kv_coregrid_probe.log``.

        The only candidate reachable from this stage is therefore
        ``paged_fused_update_cache`` plus a manual V reshard, which is what
        ``doc/fused_decoder/bench/variants.py::FusedKvUpdateDecoder``
        implements.  It comes out ~0.1 % either way and **sign-flipping between
        the two layer kinds** (better on ``full``, worse on ``sliding``), so the
        reshard costs what the saved dispatch is worth and the two-call form is
        kept.  Exact numbers, with the per-round spread, are in
        ``doc/fused_decoder/logs/variant_sweep.log``.
        """
        block_size = self.config.paged_attention_config.block_size
        n_kv = self.config.num_key_value_heads
        ttnn.experimental.paged_update_cache(
            self.k_cache,
            k,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=block_size,
            num_kv_heads=n_kv,
        )
        ttnn.experimental.paged_update_cache(
            self.v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=block_size,
            num_kv_heads=n_kv,
        )

    def _sharded_per_head_rmsnorm(self, tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
        """Scale-less RMSNorm over ``head_dim`` for a height-sharded decode tensor."""
        interleaved = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tensor)
        normed = ttnn.rms_norm(
            interleaved,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.norm_compute_kernel_config,
        )
        ttnn.deallocate(interleaved)
        sharded = ttnn.to_memory_config(normed, memory_config)
        ttnn.deallocate(normed)
        return sharded
