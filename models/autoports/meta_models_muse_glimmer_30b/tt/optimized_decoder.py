# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized TTNN decoder layer for ``meta-models/Muse-Glimmer-30B``.

``OptimizedDecoder`` is a drop-in replacement for
:class:`~models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder.FusedDecoder`
with the same public contract (``from_state_dict`` / ``prefill_forward`` /
``decode_forward`` / ``sliding_kv_tail_len`` / ``forward`` / ``kv_cache``), the
same paged-KV semantics and the same 131072-token capability.  What changes is
everything the fusing stage explicitly left to this one: **weight precision,
math fidelity, DRAM-sharded decode matmuls, the decode activation layout and the
KV-cache dtype.**

Why those and not more topology
-------------------------------

The fused decoder ended at the BF16 weight-streaming roofline: 93 % of its
2.710 ms decode step was six matmuls moving

``(6656*4608 + 6656*4096 + 4096*6656 + 3*6656*19968) * 2 B = 967,835,648 B``

of BF16 weights, at 383 GB/s -- 75 % of this part's ~512 GB/s.  Only two levers
move that number: fewer bytes, and a matmul that gets closer to peak.  This
stage pulls both.

1. **Fewer bytes.**  Attention weights go BFP8, MLP weights go BFP4:
   ``90.5 + 224.3 = 314.8 MB`` per step, **3.07x less traffic**.
2. **A better matmul.**  ``ttnn.linear``'s auto-selected decode config reaches
   383 GB/s; an explicit
   ``ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` over a
   width-sharded DRAM weight and a width-sharded L1 activation reaches
   **~490 GB/s**, i.e. 96 % of peak.  It also requires the decode activation to
   stay width-sharded in L1 across the whole layer, which the fused stage had
   already established for the residual/norm path.
3. **BFP8 KV cache**, which halves the only op whose cost grows with context:
   the ``full`` (NoPE) layer's decode SDPA at 131071.

Both levers need the same thing from the weight: a DRAM **width-sharded** layout.
``doc/optimized_decoder/logs/weight_layout_probe.log`` establishes that
``ttnn.experimental.minimal_matmul`` accepts exactly that layout (and is
marginally faster on it), so prefill and decode share **one** weight tensor --
no second copy, and the layer's weight footprint drops from 967.8 MB to 314.8 MB.

The costs, both measured and recorded in the README's limitations:

* ``ttnn.linear`` refuses a width-sharded ``input_tensor_b``
  (``matmul_device_operation.cpp:1233``), so the fused stage's "``ttnn.linear``
  below 3072 rows" branch is gone.  Prefill is ``minimal_matmul`` at every row
  count above one tile, with a per-shape ``MinimalMatmulConfig`` swept at every
  material row count to recover the short-prefill gap.
* the DRAM-sharded matmul supports **one M tile only**
  (``matmul_device_operation.cpp:1287``: ``M == 1``), so it serves decode and a
  <=32-row prefill and nothing between.

Layout contract
---------------

The decode step carries two width-sharded L1 layouts, not one, and that is a
deliberate choice (``$optimize`` OPT-011): a single grid cannot serve both the
attention block and the MLP.

* **Boundary grid, ``BOUNDARY_CORES`` = 16.**  Every ``hidden_size``-wide
  residual/norm tensor, the 4608-wide QKV projection output, the 4096-wide
  attention output and gate, and the ``o_proj`` output.  16 divides
  ``6656/32 = 208``, ``4608/32 = 144`` and ``4096/32 = 128`` exactly, so nothing on
  this grid is shard-padded -- which leaves only 1, 2, 4, 8 and 16 as candidates.
  16 is the measured whole-layer winner (1.0916 ms/token against 1.1228 at 8
  cores, 4 cores fails L1) *even though* the fused stage measured its sharded
  RMSNorm as the slower one (24.4 vs 22.8 us) and even though its 13-tile shard
  forces the norm's ``subblock_w`` to 1.  The reason is that the 13-K-tile shard
  is exactly what lets ``wqkv`` and the attention gate run at
  ``in0_block_w = 13``, and on this part ``in0_block_w`` is the field that
  matters most.
* **MLP working grid, ``MLP_WORKING_CORES`` = 26.**  The gate/up projections and
  their 19968-wide output, hence also the ``mlp_down`` matmul.  This is where the
  isolated sweep and the layer disagree hardest: 13 cores is the isolated winner
  for gate/up (0.2300 ms against 0.2301 at 26) and is **illegal in the layer** at
  every ``in0_block_w`` including 2, because at 13 cores the 19968-wide output is
  98 KB per core and, with the residual, the carried ``hidden`` and the second
  gate/up output also resident, the static circular buffers no longer fit under
  them (*"clash with L1 buffers"*, ``program.cpp:1779``).  26 cores halves that to
  49 KB.  Two reshards cross the boundary (at the pre-FF norm output and the MLP
  output), ~2 us each, against the 57 us that keeping ``in0_block_w = 8`` on
  gate/up is worth.

Everything else -- the paged prefill/decode contract, internal 8192-token
prefill chunking, sliding-window tail hand-off, ``qk_scale_factor`` fold, the
centered-RMSNorm ``1 + w`` fold, RoPE via ``rotary_embedding_hf``, the SwiGLU and
attention-gate activation folds, the sharded decode RMSNorm, the prefill SDPA
chunk of 256 -- is inherited unchanged from ``FusedDecoder``.

What was tried and rejected, with numbers, is in
``doc/optimized_decoder/README.md``; the short list is BFP4 attention weights
(3.1 % faster, PCC 0.977 on real weights), packed QKV+gate and packed gate/up
(0.6 % and 2.6 % slower -- a wider output forces a worse ``in0_block_w``), HiFi2
fidelity (69 % slower for +2.4e-4 PCC), BFP8 activations (blocked by
``nlp_create_qkv_heads_decode``), and a BF16 KV cache (9.9 % slower at 131071
once the SDPA chunking is fixed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import (
    FusedDecoder,
    _FusedNorm,
    _norm_subblock_w,
    norm_compute_kernel_config,
)
from models.common.lightweightmodule import LightweightModule

__all__ = [
    "BOUNDARY_CORES",
    "DECODE_FUSED_ACTIVATION",
    "DECODE_MATMUL",
    "DEFAULT_DECODE_SDPA",
    "DEFAULT_PRECISION",
    "MLP_WORKING_CORES",
    "MODEL_ID",
    "LAYER_KIND_FULL",
    "LAYER_KIND_SLIDING",
    "OptimizedDecoder",
    "MCAST_MAX_PER_CORE_M",
    "PREFILL_MCAST2D",
    "PREFILL_MCAST_GRID_X",
    "PREFILL_MINIMAL_BLOCKS",
    "PREFILL_NORM_SHARD_CORES",
    "PREFILL_NORM_SHARD_MAX_ROWS",
    "PROJECTION_ROLES",
    "core_rectangle",
    "rect_width_sharded_l1",
    "PrecisionPolicy",
    "decode_matmul_program_config",
    "prefill_mcast2d_program_config",
    "prefill_mcast2d_spec",
    "reference_layer_indices",
    "resolve_layer_kind",
]


# --------------------------------------------------------------- precision policy

#: The six weighted projections in a decoder layer, in the order a forward pass
#: reaches them.  The precision policy is keyed by these names.
PROJECTION_ROLES = ("wqkv", "attn_gate", "o_proj", "mlp_gate", "mlp_up", "mlp_down")


@dataclass(frozen=True)
class PrecisionPolicy:
    """Named per-tensor-group precision and math fidelity.

    Tuned one group at a time (``$optimize``: "do not use a blunt global dtype
    policy"), so a regression can be assigned.  ``doc/optimized_decoder/work_log.md``
    section 3 has the per-group evidence; the fields are separate precisely so
    that attention-weight precision can be searched independently of MLP-weight
    precision (OPT-007) and of the KV cache (OPT-002).
    """

    name: str
    #: ``wqkv``, the attention output gate and ``o_proj``.
    attn_weight_dtype: ttnn.DataType
    #: MLP gate and up projections.
    mlp_gate_up_weight_dtype: ttnn.DataType
    #: MLP down projection, kept separate because it is the more precision
    #: sensitive of the three (it reduces over the 19968-wide intermediate).
    mlp_down_weight_dtype: ttnn.DataType
    kv_cache_dtype: ttnn.DataType
    #: Activations, residual stream and norm outputs.
    activation_dtype: ttnn.DataType
    #: Math fidelity for the decode (DRAM-sharded) projections.  The *default*
    #: for every role; :attr:`decode_math_fidelity_by_role` overrides it per
    #: group, which is what ``$datatype-sweep`` needs to compare BFP8+LoFi
    #: against BFP8+HiFi2 on the attention projections **without** moving the
    #: MLP off the BFP4+LoFi pairing at the same time.
    decode_math_fidelity: ttnn.MathFidelity
    #: Math fidelity for the prefill ``minimal_matmul`` projections.
    prefill_math_fidelity: ttnn.MathFidelity
    #: ``((role, fidelity), ...)`` decode overrides.  A tuple rather than a dict
    #: because the dataclass is frozen *and* hashed into the generator cache key.
    decode_math_fidelity_by_role: tuple[tuple[str, ttnn.MathFidelity], ...] = ()
    #: The same, for the prefill kernels.
    prefill_math_fidelity_by_role: tuple[tuple[str, ttnn.MathFidelity], ...] = ()
    #: Layer exceptions: ``((layer_indices, ((field, value), ...)), ...)``.  Each
    #: entry replaces those fields on the policy handed to the listed layers, so
    #: a policy can keep the first and last decoder layer at a higher precision
    #: than the inner stack.  :meth:`for_layer` resolves it.
    layer_exceptions: tuple[tuple[tuple[int, ...], tuple[tuple[str, Any], ...]], ...] = ()

    def weight_dtype(self, role: str) -> ttnn.DataType:
        if role in ("wqkv", "attn_gate", "o_proj"):
            return self.attn_weight_dtype
        if role in ("mlp_gate", "mlp_up"):
            return self.mlp_gate_up_weight_dtype
        if role == "mlp_down":
            return self.mlp_down_weight_dtype
        raise KeyError(f"unknown projection role {role!r}")

    def decode_fidelity(self, role: str) -> ttnn.MathFidelity:
        """Decode math fidelity for one projection role."""
        self.weight_dtype(role)  # rejects an unknown role with the same message
        return dict(self.decode_math_fidelity_by_role).get(role, self.decode_math_fidelity)

    def prefill_fidelity(self, role: str) -> ttnn.MathFidelity:
        """Prefill math fidelity for one projection role."""
        self.weight_dtype(role)
        return dict(self.prefill_math_fidelity_by_role).get(role, self.prefill_math_fidelity)

    def for_layer(self, layer_idx: int) -> "PrecisionPolicy":
        """This policy as the layer at ``layer_idx`` should see it.

        Returns ``self`` when no exception lists that index, so the common case
        allocates nothing and the generator cache key is unchanged.
        """
        from dataclasses import replace

        changes: dict[str, Any] = {}
        matched = False
        for indices, fields in self.layer_exceptions:
            if layer_idx in indices:
                matched = True
                changes.update(dict(fields))
        if not matched:
            return self
        changes["name"] = f"{self.name}@layer{layer_idx}"
        # ``layer_exceptions`` must not survive into the per-layer policy: it has
        # already been applied, and leaving it would re-apply on a second call.
        changes["layer_exceptions"] = ()
        return replace(self, **changes)


#: Shipped policy.  BF16 activations/residuals/norms, BFP8 attention weights,
#: BFP4 MLP weights, BFP8 KV cache, LoFi decode and prefill math fidelity.
#:
#: This is the ``$optimize`` fallback starting policy ("BF16 activations and
#: norms, BFP8 attention/MLP weights, BFP8 KV cache if PCC allows it, and
#: selective BFP4 trials for MLP/expert weights") with the BFP4 MLP trial kept
#: because it won on real weights, and with the BFP4 *attention* trial (OPT-007)
#: rejected on measured evidence rather than preference -- see
#: ``doc/optimized_decoder/README.md`` "Precision policy".
DEFAULT_PRECISION = PrecisionPolicy(
    name="attn-bfp8-mlp-bfp4-kv-bfp8-lofi",
    attn_weight_dtype=ttnn.bfloat8_b,
    mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
    mlp_down_weight_dtype=ttnn.bfloat4_b,
    kv_cache_dtype=ttnn.bfloat8_b,
    activation_dtype=ttnn.bfloat16,
    decode_math_fidelity=ttnn.MathFidelity.LoFi,
    prefill_math_fidelity=ttnn.MathFidelity.LoFi,
)


# ------------------------------------------------------------------ decode layout

#: Cores for every ``hidden_size``/4608/4096-wide width-sharded L1 tensor in the
#: decode step: the residual stream, all four hidden-size RMSNorms, the QKV
#: projection output, the attention output and gate, and the ``o_proj`` output.
#:
#: Must divide ``6656/32 = 208``, ``4608/32 = 144`` and ``4096/32 = 128`` tiles so
#: nothing on this grid is shard-padded, which admits only 1, 2, 4, 8 and 16.
#: Measured whole-layer traced decode (sliding / full), everything else fixed:
#: 4 cores fails L1, 8 cores 1.1228 / 1.0961, **16 cores 1.0916 / 1.0652**.
#:
#: 16 wins despite being the *worse* choice for the norm in isolation -- the fused
#: stage's probe puts a 16-core sharded RMSNorm at 24.4 us against 22.8 at 8
#: (``doc/fused_decoder/logs/norm_shard_probe.log``) -- and despite its 13-tile
#: shard forcing the norm's ``subblock_w`` down to 1.  The 13-K-tile shard is what
#: makes ``in0_block_w = 13`` legal for ``wqkv`` and the attention gate, and that
#: is worth more than the norm gives up.
BOUNDARY_CORES = 16

#: Cores for the MLP working shard: the gate/up projection inputs and their
#: 19968-wide output, hence also the ``mlp_down`` input.
#:
#: Phase-specific by measurement, not by taste ($optimize OPT-011).  On the
#: boundary grid the 19968-wide output block leaves only ``in0_block_w <= 2``
#: legal for gate/up (13 and 26 overflow the 1.5 MB static circular-buffer
#: budget), which measures 0.2584 ms per dispatch against 0.2300 ms on a wider
#: grid -- 57 us across the two dispatches, against ~4 us for the two reshards
#: that cross the boundary.
#:
#: 26 rather than 13, which is the *isolated* winner, because 13 fails in the
#: layer: see the ``("mlp_gate", bfloat4_b)`` note in ``DECODE_MATMUL``.
MLP_WORKING_CORES = 26

#: Decode SDPA ``(grid_x, grid_y, q_chunk_size, k_chunk_size)``; ``None`` grid
#: dimensions mean "the whole device compute grid".
#:
#: ``0`` chunk sizes ask ``paged_scaled_dot_product_attention_decode`` to choose
#: them.  This is the one attention-op knob this stage changes, and it is the
#: largest single non-matmul win in it.  On the ``full`` (NoPE) layer at context
#: 131071 -- the only place the decode SDPA is material, because it reads the
#: entire cache -- against the fused stage's inherited ``q=32 / k=64``:
#:
#: =============  ===========  ==========  ===========
#: grid           q/k chunk    BFP8 cache  BF16 cache
#: =============  ===========  ==========  ===========
#: 11x10 (device) 32 / 64      1.5584      1.5569
#: 8x8            32 / 64      1.5577      1.5570
#: 8x8            32 / 128     1.3475      1.4495
#: 8x8            op-chosen    1.2720      1.4487
#: 11x10 (device) op-chosen    **1.2658**  1.4041
#: =============  ===========  ==========  ===========
#:
#: Two things fall out.  The win is the **chunk size**, not the core grid: the
#: op's own choice is 1.2658 against 1.5584, i.e. **19 % of the whole decode
#: step**, while the grid is worth 0.5 %.  And the fixed ``k_chunk = 64`` was
#: *hiding the BFP8 KV cache*: at ``q=32 / k=64`` the reduced cache dtype is worth
#: nothing (1.5584 vs 1.5569, i.e. noise), and with the op's own chunking it is
#: worth 1.4041 -> 1.2658, **10 %**.  A reduced cache dtype measured under a
#: latency-bound attention config would have looked useless.
#: At context 2048 every row here is within 0.5 %, so nothing is traded away.
#: ``doc/optimized_decoder/logs/layer_ab_sdpa.log`` has the sweep.
DEFAULT_DECODE_SDPA = (None, None, 0, 0)

#: ``(role, weight dtype) -> (cores, in0_block_w)`` for the decode DRAM-sharded
#: matmuls.
#:
#: ``cores`` is the width-shard core count of the *activation* (and therefore of
#: the output); ``in0_block_w`` is the K-tile block, which must divide
#: ``K / (32 * cores)``.  Every entry is the measured winner of a full legal
#: sweep over both fields **at that dtype**
#: (``doc/optimized_decoder/logs/decode_matmul_geometry_bfp{4,8}.log``), and the
#: sweep had to be run per dtype rather than once, because the L1
#: circular-buffer budget is dtype-scaled and moves the optimum
#: ($optimize OPT-014).  Concretely: ``in0_block_w = 26`` is the fastest legal
#: value for ``wqkv`` at BFP4 and **illegal** at BFP8 (1,782,400 B of static CBs
#: against a 1,572,864 B budget), so a single table would either leave BFP4 8 %
#: slower or crash BFP8.
#:
#: The **isolated** sweep's own winners -- one matmul on an otherwise empty device,
#: so the core counts here are what that probe preferred, *not* what the layer
#: ships (see the two notes below the table):
#:
#: ===========  =========  ==========  =========  =========  ==========
#: role         BFP4                              BFP8
#: -----------  ---------------------  ---------  ---------  ----------
#: \            cores/bw   ms          why capped cores/bw   ms
#: ===========  =========  ==========  =========  =========  ==========
#: wqkv         8 / 26     0.0569      26 = K/8   8 / 13     0.0701
#: attn_gate    8 / 26     0.0513      26 = K/8   8 / 13     0.0625
#: o_proj       8 / 16     0.0519      32 -> L1   8 / 4      0.0620
#: mlp_gate/up  13 / 8     0.2300      16 -> L1   8 / 2      0.2859
#: mlp_down     13 / 24    0.2242      48 -> L1   8 / 6      0.2820
#: ===========  =========  ==========  =========  =========  ==========
#:
#: Two things this table does *not* say, because both have bitten a reader:
#:
#: * the ``cores`` column is the **activation shard** core count, and for the three
#:   attention roles it is pinned to :data:`BOUNDARY_CORES` (16) by the layer, not
#:   chosen by the sweep -- ``__init__`` rejects any other value, because those
#:   tensors share the residual/norm layout.  The isolated sweep's own winner for
#:   ``wqkv``/BFP8 is 13 cores at ``in0_block_w = 4`` (0.0686 ms,
#:   ``logs/decode_matmul_geometry_bfp8.log.gz``); 16 wins the *whole layer*, which
#:   is the comparison that decides it (work log section 4.3).  So "measured winner
#:   of a full legal sweep" applies to ``in0_block_w`` at the shipped core count,
#:   not to the core count itself;
#: * the MLP roles are free, and both dtypes land on the same 26-core working shard
#:   here, so both pay the two boundary reshards.  The layer reads the core count
#:   out of this table rather than hard-coding :data:`MLP_WORKING_CORES`, so a
#:   future policy can move the MLP without touching the forward pass.
DECODE_MATMUL: dict[tuple[str, ttnn.DataType], tuple[int, int]] = {
    ("wqkv", ttnn.bfloat4_b): (BOUNDARY_CORES, 13),
    ("attn_gate", ttnn.bfloat4_b): (BOUNDARY_CORES, 13),
    ("o_proj", ttnn.bfloat4_b): (BOUNDARY_CORES, 8),
    ("mlp_gate", ttnn.bfloat4_b): (MLP_WORKING_CORES, 8),
    ("mlp_up", ttnn.bfloat4_b): (MLP_WORKING_CORES, 8),
    ("mlp_down", ttnn.bfloat4_b): (MLP_WORKING_CORES, 24),
    # 13 cores is the isolated-probe winner for gate/up (0.2300 ms against
    # 0.2301 at 26) and is **illegal in the layer**: at 13 cores the 19968-wide
    # gate/up output is 98 KB per core, and with the residual, the carried
    # ``hidden`` and the second gate/up output also resident the static circular
    # buffers no longer fit under them ("clash with L1 buffers",
    # program.cpp:1779) at any in0_block_w, including 2.  26 cores halves the
    # per-core output to 49 KB and is 1.1227 ms/token against 1.1234 at 52
    # ($optimize: measure the whole layer, not the isolated op).
    ("wqkv", ttnn.bfloat8_b): (BOUNDARY_CORES, 13),
    ("attn_gate", ttnn.bfloat8_b): (BOUNDARY_CORES, 13),
    ("o_proj", ttnn.bfloat8_b): (BOUNDARY_CORES, 4),
    ("mlp_gate", ttnn.bfloat8_b): (MLP_WORKING_CORES, 4),
    ("mlp_up", ttnn.bfloat8_b): (MLP_WORKING_CORES, 4),
    ("mlp_down", ttnn.bfloat8_b): (MLP_WORKING_CORES, 8),
    # BF16 weights are not a shipped policy -- they are 3.07x the traffic -- but
    # a caller can still ask for them via ``weight_dtype=``, so every role needs
    # a legal entry.  These are the largest ``in0_block_w`` that fits L1 at BF16.
    ("wqkv", ttnn.bfloat16): (BOUNDARY_CORES, 1),
    ("attn_gate", ttnn.bfloat16): (BOUNDARY_CORES, 1),
    ("o_proj", ttnn.bfloat16): (BOUNDARY_CORES, 2),
    ("mlp_gate", ttnn.bfloat16): (MLP_WORKING_CORES, 1),
    ("mlp_up", ttnn.bfloat16): (MLP_WORKING_CORES, 1),
    ("mlp_down", ttnn.bfloat16): (MLP_WORKING_CORES, 3),
}

#: ``role -> ((min_rows, (M_block, K_block, N_block)), ...)`` for prefill
#: ``minimal_matmul``, highest ``min_rows`` first.
#:
#: The fused stage only needed a block table at the 8192-row chunk, because below
#: 3072 rows it dispatched to ``ttnn.linear`` instead.  That branch is gone (the
#: width-sharded weight is illegal for ``ttnn.linear``), so the op's own
#: ``M=K=N=8`` default now has to cover every row count -- and it is weak at the
#: short ones.  Swept per shape at 128 / 512 / 2048 / 4096 / 8192 rows over
#: ``M_block in {4, 8}``, ``K_block in {8, 13, 16, 26}``,
#: ``N_block in {16, 24, 32}`` plus the low-row ``M_block in {1, 2}`` band; see
#: ``doc/optimized_decoder/logs/mm_block_sweep_*.log``.
#: A ``None`` entry means "pass no ``config=``", i.e. use the op's own
#: ``M=K=N=8`` choice, which really is the fastest answer for a few
#: (shape, dtype, row-count) cells.
#:
#: Keyed by ``(role, weight dtype)`` for the same reason ``DECODE_MATMUL`` is: the
#: circular-buffer budget is dtype-scaled, so the BFP4 winners are not merely
#: suboptimal at BFP8, several of them are **illegal** -- the first version of
#: this table was BFP4-only and crashed every BFP8-MLP candidate in prefill with
#: *"Statically allocated circular buffers ... grow to 1593216 B"*.  Measured gains
#: over the op default: +2 % to +27 % on the attention projections, +13 % to
#: +17 % on gate/up, +2 % to +20 % on ``mlp_down``.
#:
#: The pattern across both dtypes: large ``N_block`` (16-32) is what matters, and
#: ``M_block`` wants to track the row count -- 2 at 128-512 rows, 8 from 2048 up.
PREFILL_MINIMAL_BLOCKS: dict[tuple[str, ttnn.DataType], tuple[tuple[int, tuple[int, int, int] | None], ...]] = {
    # ---- BFP4 (the shipped MLP dtype); logs/mm_block_sweep_bfp4.log
    ("wqkv", ttnn.bfloat4_b): ((2048, (8, 8, 16)), (TILE_SIZE, (4, 13, 16))),
    ("attn_gate", ttnn.bfloat4_b): ((2048, (8, 8, 16)), (TILE_SIZE, (4, 8, 16))),
    ("o_proj", ttnn.bfloat4_b): ((8192, None), (2048, (4, 13, 24)), (TILE_SIZE, (4, 8, 24))),
    ("mlp_gate", ttnn.bfloat4_b): ((8192, (4, 8, 32)), (2048, (8, 8, 16)), (TILE_SIZE, (4, 16, 16))),
    ("mlp_up", ttnn.bfloat4_b): ((8192, (4, 8, 32)), (2048, (8, 8, 16)), (TILE_SIZE, (4, 16, 16))),
    ("mlp_down", ttnn.bfloat4_b): ((8192, (8, 13, 16)), (2048, None), (TILE_SIZE, (4, 13, 24))),
    # ---- BFP8 (the shipped attention dtype); logs/mm_block_sweep_bfp8.log
    ("wqkv", ttnn.bfloat8_b): ((8192, None), (2048, (8, 8, 16)), (TILE_SIZE, (2, 8, 24))),
    ("attn_gate", ttnn.bfloat8_b): ((8192, None), (2048, (8, 8, 16)), (TILE_SIZE, (2, 8, 16))),
    ("o_proj", ttnn.bfloat8_b): ((8192, (8, 13, 8)), (2048, (4, 8, 24)), (TILE_SIZE, (2, 8, 24))),
    ("mlp_gate", ttnn.bfloat8_b): ((2048, (8, 8, 16)), (TILE_SIZE, (2, 13, 24))),
    ("mlp_up", ttnn.bfloat8_b): ((2048, (8, 8, 16)), (TILE_SIZE, (2, 13, 24))),
    ("mlp_down", ttnn.bfloat8_b): ((8192, (8, 8, 16)), (2048, (8, 13, 8)), (TILE_SIZE, (2, 13, 24))),
}


#: ``(role, weight dtype) -> ((min_rows, (grid_y, in0_block_w) | None), ...)`` for
#: the prefill 2D-multicast matmul, highest ``min_rows`` first.  A ``None`` spec
#: means "``minimal_matmul`` wins from this row count up".
#:
#: This table exists because the stage's first pass rejected ``ttnn.linear`` for
#: prefill on one API error -- ``MatmulMultiCoreProgramConfig: Input B memory
#: layout must be INTERLEAVED`` (``matmul_device_operation.cpp:1233``) -- which is
#: the *auto-selected* fallback config talking, not a statement about the op.
#: ``validate_matmul_mcast2d_config`` (``:1541-1553``) accepts a ``WIDTH_SHARDED``
#: ``input_tensor_b`` **in DRAM**, and the extra per-``per_core_N`` clause that
#: would have made it useless here is gated on ``buffer_type() != DRAM``
#: (``:1525``).  So an explicit 2D-multicast config reads exactly the weight this
#: stage already ships, and it is 1.3-2.0x faster than ``minimal_matmul``
#: everywhere between one tile and ~1024 rows -- which is precisely the band where
#: the first pass was *slower* than the fused decoder it replaces.
#:
#: Summed over the six dispatches at 128 rows: 3.060 ms of ``minimal_matmul``
#: against **1.762 ms**, versus 2.67 ms for the fused stage's BF16
#: ``ttnn.linear``.  A short prefill goes from 15 % slower than the fused decoder
#: to 1.5x faster than it.  ``logs/prefill_mcast_probe.log`` (182 measurements) and
#: ``logs/prefill_mcast_probe_bigrows.log`` are the sweep; the crossover is where
#: the matmul stops being launch-bound and becomes compute-bound, at which point
#: ``minimal_matmul``'s full 11x10 grid beats the 8-column grid this op is pinned
#: to (see ``PREFILL_MCAST_GRID_X`` below).
#:
#: Speedup over ``minimal_matmul`` at the shipped dtype, best legal candidate:
#:
#: ==========  =====  =====  =====  =====  ======  ======  ======
#: role        64 r   128 r  256 r  512 r  1024 r  2048 r  8192 r
#: ==========  =====  =====  =====  =====  ======  ======  ======
#: wqkv        1.63x  1.64x  1.61x  1.48x  1.75x   0.95x   0.68x
#: attn_gate   1.43x  1.47x  1.44x  1.31x  1.74x   0.93x   0.76x
#: o_proj      1.49x  1.46x  1.40x  1.30x  1.46x   0.93x   0.68x
#: mlp_gate/up 2.00x  1.95x  1.82x  1.54x  0.87x   0.75x   0.67x
#: mlp_down    1.59x  1.57x  1.53x  1.49x  0.88x   0.99x   0.77x
#: ==========  =====  =====  =====  =====  ======  ======  ======
#:
#: The ``>= 2048`` column is why the table hands those rows back rather than
#: assuming the new kernel is uniformly better: ``out_block_h``/``out_block_w``
#: bounding makes the large-row candidates *legal* (they otherwise overflow L1),
#: and they are still 5-33 % slower, because by then the matmul is compute-bound
#: and 80 cores lose to 110.
#: Entries are ``(max_rows, (grid_y, in0_block_w))``, **ascending**: the first band
#: whose ``max_rows`` covers the row count wins, and a row count past the last band
#: falls through to ``minimal_matmul``.
#:
#: The bound is an *upper* one on purpose, and that is the whole reason this table
#: is not keyed the way ``PREFILL_MINIMAL_BLOCKS`` is.  ``grid_y`` fixes
#: ``per_core_M = ceil(rows / 32 / grid_y)``, which sizes the L1 output block; a
#: lower-bound band would apply a ``grid_y`` measured at 1024 rows to *any* larger
#: row count and overflow the 1.5 MB static circular-buffer budget.  It really
#: happens: a batched prefill's per-user 2000-token prompt pads to 2016 rows, which
#: under a lower-bound table took the 1024-row band's ``grid_y = 8``, asked for
#: ``per_core_M = 8``, and threw *"Statically allocated circular buffers on core
#: range [0-0 - 7-7] grow to 1966976 B"* out of six otherwise-passing tests.
#: With ascending bands ``per_core_M`` is at most 4 at every legal row count,
#: which ``test_prefill_mcast_table_is_legal`` asserts at each band's worst case.
PREFILL_MCAST2D: dict[tuple[str, ttnn.DataType], tuple[tuple[int, tuple[int, int]], ...]] = {
    # ---- BFP8 attention projections
    ("wqkv", ttnn.bfloat8_b): ((128, (2, 13)), (256, (4, 13)), (512, (8, 26)), (1024, (8, 16))),
    ("attn_gate", ttnn.bfloat8_b): ((128, (2, 13)), (256, (4, 16)), (512, (8, 26)), (1024, (8, 13))),
    ("o_proj", ttnn.bfloat8_b): ((128, (2, 8)), (256, (4, 8)), (512, (8, 16)), (1024, (8, 16))),
    # ---- BFP4 MLP projections.  The 19968-wide gate/up output makes these the
    # L1-tightest rows in the table: at 512 rows only ``in0_block_w = 8`` fits the
    # 1.5 MB budget (13, 16 and 26 all overflow), and past 512 rows
    # ``minimal_matmul``'s full 11x10 grid wins anyway (0.87x at 1024).
    ("mlp_gate", ttnn.bfloat4_b): ((64, (2, 13)), (128, (4, 13)), (256, (8, 13)), (512, (8, 8))),
    ("mlp_up", ttnn.bfloat4_b): ((64, (2, 13)), (128, (4, 13)), (256, (8, 13)), (512, (8, 8))),
    ("mlp_down", ttnn.bfloat4_b): ((64, (2, 26)), (128, (4, 26)), (512, (8, 26))),
}

#: The 2D-multicast matmul's core-**column** count when ``input_tensor_b`` is
#: width-sharded in DRAM.  It must equal the DRAM bank count, and this is *not*
#: checked by the op.
#:
#: At ``grid_x`` greater than the bank count the op validates, launches, and
#: returns ``inf`` in tens of thousands of output elements -- a silent miscompute.
#: The same grids are correct with a DRAM-*interleaved* ``input_tensor_b``, which
#: isolates it to the width-sharded in1 reader assigning core column ``j`` to
#: weight shard ``j`` and running off the end of the shard set.  Minimal repro:
#: ``bench/prefill_mcast_probe.py --repro`` ->
#: ``logs/mcast_gx_bug_repro.log``; ``test_prefill_mcast_table_is_legal`` pins the
#: constraint so a future grid change cannot reintroduce it quietly.
#:
#: ``None`` means "read it off the device", which is what the layer does; the
#: constant exists to name the rule.
PREFILL_MCAST_GRID_X = None

#: Fold SiLU into the MLP gate matmul and sigmoid into the attention-gate matmul,
#: instead of leaving them on the ``ttnn.mul`` that consumes each.
#:
#: This is the elementwise half of the phase-specific working-shard trade
#: ($optimize OPT-010/OPT-011: the activation "may be fused into the gate matmul
#: **or** into the following binary elementwise op").  Moving the decode
#: activations onto narrow width-sharded L1 grids made the two SFPU multiplies
#: *more* expensive than they were on the fused stage's 110 DRAM-interleaved
#: cores -- SwiGLU 14.23 -> 40.47 us and the attention gate 5.96 -> 14.28 us --
#: because an SFPU transcendental over a fixed tile count does not care that the
#: tensor is now conveniently placed, only how many cores are working on it.
#: The two plain residual adds on the same 16-core shard cost 1.72 us, which is
#: what a multiply without a transcendental costs there.
#:
#: The DRAM-sharded matmul can absorb both: a non-RELU ``fused_activation``
#: compiles to ``SFPU_ACTIVATION`` in
#: ``matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:349``.
#:
#: **Measured, and it loses.**  Both rows from one run,
#: ``bench/layer_ab.py --candidates mlp_bfp4,fused_act``
#: (``doc/optimized_decoder/logs/layer_ab_fused_activation.log``), traced decode
#: ms/token, min of 3 rounds:
#:
#: =====================  ===============  ===============
#: candidate              sliding          full
#: =====================  ===============  ===============
#: activation on the mul  1.0908           1.0602    (shipped)
#: fused into the matmul  1.1393 (+4.4 %)  1.1082 (+4.5 %)
#: =====================  ===============  ===============
#:
#: Both rows report the same prefill and decode PCC to six decimals in that run
#: (0.993759 / 0.993506 on the synthetic harness the A/B uses), so the fold is
#: numerically inert and this is purely a scheduling result: the
#: matmul's ``SFPU_ACTIVATION`` runs on its **12** worker cores -- fixed to the
#: DRAM bank count -- interleaved with the unpack it is already bottlenecked on,
#: while the separate ``ttnn.mul`` gets the MLP's 26-core working shard and the
#: attention gate's 16-core boundary shard.  Fewer, larger ops is the usual
#: direction; here the op it would merge into is the one with the fewest cores.
#: Left ``False``, and the knob is kept so the comparison is one flag away.
DECODE_FUSED_ACTIVATION = False

#: Core count the decode SwiGLU multiply runs on, or ``None`` to leave it on the
#: MLP gate/up output grid.
#:
#: That multiply carries the SFPU SiLU (:data:`DECODE_FUSED_ACTIVATION` measured
#: folding it into the matmul as 4.4 % slower), and an SFPU transcendental costs
#: time per *tile per core*: on the shipped 16-core grid it is **18.0 us** in the
#: full-model decode profile, against **1.9 us** for a plain 6656-wide residual add
#: on the same grid.  It is the largest non-matmul row in the decode layer.
#:
#: The obvious lever -- give ``mlp_gate``/``mlp_up`` a wider output grid -- is not
#: available: the DRAM-sharded matmul requires ``K_tiles % cores == 0`` (*"in DRAM
#: sharded Matmul we don't have support for un-even sharding currently. K: 208,
#: per_core_K: 11"*), so the gate/up core count must divide ``6656/32 = 208`` and
#: ``mlp_down``'s input must divide ``5120/32 = 160``; 16 is the largest count in
#: both sets (8, 13, 16, 26, 52, 104) and (8, 10, 16, 20, 32, 40, 80).  This knob
#: is the other way round: reshard the two operands onto a wider grid for the
#: multiply alone and reshard the product back for ``mlp_down``, paying three
#: reshards to divide the SFPU work.
#:
#: Measured on the reduced two-layer full-model build, traced logits-only decode,
#: min of 3 rounds x 32 replays, one invocation
#: (``doc/optimized_full_model/logs/decode_ab_swiglu.log``); every arm is PCC
#: 1.000000 against the shipped grid and picks the same token:
#:
#: ==================  ==============  ==========
#: gate/up mul grid    ms / 2 layers   delta
#: ==================  ==============  ==========
#: 16 (no reshard)     1.5375          --
#: 20                  1.5408          +0.21 %
#: 32                  1.5357          -0.12 %
#: 40                  1.5306          -0.45 %
#: **80**              **1.5248**      **-0.83 %**
#: ==================  ==============  ==========
#:
#: 80 is the largest core count that divides the 160-tile intermediate width, and
#: the per-round spread is +-0.0005 ms, so the ordering is ~60x the noise.  The
#: saving is per layer, so on the 52-layer model it is 26x the 2-layer delta.
#:
#: The ``mlpN`` family -- moving the gate/up *matmul* output grid instead -- is the
#: cheaper-looking version of this and it is not available: ``mlp8`` is the only
#: other legal count and it measured 1.5781 (+2.6 %).
DECODE_SWIGLU_MUL_CORES: int | None = 80


def resolve_decode_swiglu_mul_cores(
    intermediate_size: int,
    preferred: int | None = DECODE_SWIGLU_MUL_CORES,
) -> int | None:
    """Use the wide SwiGLU grid only when the local MLP width shards exactly.

    The 80-core optimization was measured on P150x4's padded 5,120-wide local
    MLP (160 tiles).  P150 and P150x2 have 624 and 312 local tiles respectively,
    neither divisible by 80.  For those profiles the correctness-preserving
    baseline is to multiply on the gate/up matmul grid and avoid all three
    reshards.  Their latency sweeps can nominate a different exact divisor.
    """
    if intermediate_size % TILE_SIZE:
        raise ValueError(f"intermediate_size={intermediate_size} must be tile aligned")
    if preferred is None:
        return None
    if preferred < 1:
        raise ValueError(f"decode SwiGLU multiply core count must be positive, got {preferred}")
    width_tiles = intermediate_size // TILE_SIZE
    return preferred if width_tiles % preferred == 0 else None


#: Row count up to which the four hidden-size prefill RMSNorms run width-sharded
#: in L1 instead of DRAM interleaved, and the core count they use.
#:
#: ``ttnn.rms_norm`` on a DRAM-interleaved input parallelises over tile *rows*, so
#: a short prefill starves it: the 128-row window's four hidden-size norms cost
#: ~134 us each on **4 cores**, 21 % of the whole window.  The same norm on a
#: 16-core width shard costs 33.0 us including the ``interleaved_to_sharded`` and
#: ``sharded_to_interleaved`` that bracket it -- **4.1x** -- and 57.6 us at 256
#: rows (``logs/sharded_norm_grid_probe_rect.log``).
#:
#: The bound is L1, and it is sharp. Per-core the shard must hold the input and
#: the (non-inplace) output plus the norm's circular buffers:
#:
#: =====  =======  ==========================================
#: rows   16 c     result
#: =====  =======  ==========================================
#: 32     8x2      correct, ``max|diff| = 0.03182``
#: 128    8x2      correct, ``max|diff| = 0.04022``, 33.0 us
#: 256    8x2      correct, ``max|diff| = 0.03786``, 57.6 us
#: 512    8x2      CB overflow: 2,091,904 B against 1,572,864
#: =====  =======  ==========================================
#:
#: 16 rather than 8 cores because 8 is both slower (44.1 us at 128 rows) and
#: L1-blocked at 256; and 16 rather than 26 or 52 because those have no exact
#: rectangle on an 11-wide grid and would hit the silent-miscompute path in
#: :func:`core_rectangle`.
PREFILL_NORM_SHARD_MAX_ROWS = 256
PREFILL_NORM_SHARD_CORES = 16

#: The largest ``per_core_M`` any :data:`PREFILL_MCAST2D` band may ask for.
#:
#: ``per_core_M`` sizes the L1 output block, so it -- not the row count -- is what
#: decides whether a 2D-multicast candidate fits the 1.5 MB static
#: circular-buffer budget.  Every band in the table was measured at
#: ``per_core_M <= 4``, and ``test_prefill_mcast_table_is_legal`` asserts that at
#: each band's worst-case row count, which is what makes an arbitrary logical
#: prefill length safe rather than merely untested.
MCAST_MAX_PER_CORE_M = 4


def prefill_mcast2d_spec(role: str, rows: int, dtype: ttnn.DataType) -> tuple[int, int] | None:
    """``(grid_y, in0_block_w)`` for a prefill 2D-multicast matmul, or ``None``.

    ``None`` -- from an unswept ``(role, dtype)`` or a row count past the last
    band -- means "use ``minimal_matmul``".
    """
    for max_rows, spec in PREFILL_MCAST2D.get((role, dtype), ()):
        if rows <= max_rows:
            return spec
    return None


def _out_subblock(block_h: int, block_w: int) -> tuple[int, int]:
    """Largest ``(h, w)`` dividing the output block with ``h * w <= 8``."""
    best = (1, 1)
    for h in range(1, min(block_h, 8) + 1):
        for w in range(1, min(block_w, 8 // h) + 1):
            if block_h % h == 0 and block_w % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def prefill_mcast2d_program_config(
    rows: int, n: int, grid_y: int, in0_block_w: int, dram_banks: int
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """2D-multicast program config over a DRAM width-sharded weight.

    ``compute_with_storage_grid_size.x`` is pinned to ``dram_banks``; see
    :data:`PREFILL_MCAST_GRID_X` for why anything else silently miscomputes.
    """
    per_core_m = math.ceil(rows / TILE_SIZE / grid_y)
    per_core_n = math.ceil(n / TILE_SIZE / dram_banks)
    subblock_h, subblock_w = _out_subblock(per_core_m, per_core_n)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(dram_banks, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def core_rectangle(num_cores: int, grid: ttnn.CoreCoord) -> tuple[int, int] | None:
    """Widest ``(gx, gy)`` with ``gx * gy == num_cores`` inside ``grid``, or ``None``.

    Whether a shard core set is an exact rectangle is not cosmetic: a sharded
    ``ttnn.rms_norm`` whose ``LayerNormShardedMultiCoreProgramConfig`` grid covers
    *more* cores than the tensor's shard **silently miscomputes** above
    ``block_h = 1`` -- 75,155 non-finite elements at 128 rows on a 16-core shard
    under an 11x2 grid (``doc/optimized_decoder/logs/sharded_norm_grid_probe.log``).
    A rectangular shard lets the program grid match the shard exactly, which is the
    only configuration measured correct at every ``block_h``.

    On an 11x10 grid a 16-core shard is ``8x2``; 13, 26 and 52 divide the 208
    hidden-size tiles but have no rectangle (13 > 11).

    **Only the prefill norm can use this.** A decode boundary tensor's core set is
    not ours to pick: the DRAM-sharded matmul ignores the output shard grid it is
    given and writes the row-major prefix :func:`core_range_set` builds, so the
    decode norms are stuck with a program grid wider than their shard. That is safe
    there and only there, because a decode step is one tile row -- ``block_h == 1``,
    the one case measured correct for every core count -- and
    ``_decode_norm_configs`` raises rather than build the unsafe combination above
    it. Prefill norms start from a DRAM-interleaved tensor this layer shards
    itself, so they get the rectangle.
    """
    for gx in range(min(num_cores, grid.x), 0, -1):
        if num_cores % gx == 0 and num_cores // gx <= grid.y:
            return gx, num_cores // gx
    return None


def core_range_set(num_cores: int, grid: ttnn.CoreCoord) -> ttnn.CoreRangeSet:
    """Row-major prefix of ``grid`` holding exactly ``num_cores`` cores.

    This is not a free choice for anything the decode DRAM-sharded matmul produces.
    That op **ignores the output shard grid it is handed** and writes its result on
    its own storage-core layout, which is exactly this row-major prefix: asking for
    a 16-core ``8x2`` rectangle and getting back ``{[0-0 - 10-0], [0-1 - 4-1]}``
    is what happens (see the ``core_rectangle`` note). So every decode boundary
    tensor is a prefix whether or not a rectangle exists, and the norms that
    consume those tensors have to accept it.

    Both the sharded-LayerNorm program factory
    (``layernorm_device_operation.cpp:185-215``) and the DRAM-sharded matmul accept
    a non-rectangular prefix; the 26-core MLP working shard has no rectangle at all.
    """
    ranges = []
    full_rows = num_cores // grid.x
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
    rest = num_cores % grid.x
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rest - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def rect_width_sharded_l1(rows: int, width: int, cores: int, grid: ttnn.CoreCoord) -> ttnn.MemoryConfig:
    """``[rows, width]`` width-sharded over an exact ``gx x gy`` rectangle of ``cores``.

    Only for tensors this layer shards itself -- i.e. the prefill norms. Anything a
    decode DRAM-sharded matmul produces has the prefix layout instead; see
    :func:`core_range_set`.
    """
    rect = core_rectangle(cores, grid)
    if rect is None:
        raise ValueError(f"{cores} cores have no exact rectangle on a {grid.x}x{grid.y} grid")
    gx, gy = rect
    per_core = math.ceil(width / (TILE_SIZE * cores)) * TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            (rows, per_core),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def width_sharded_l1(rows: int, width: int, cores: int, grid: ttnn.CoreCoord) -> ttnn.MemoryConfig:
    """``[rows, width]`` width-sharded in L1 over ``cores`` cores."""
    per_core = math.ceil(width / (TILE_SIZE * cores)) * TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set(cores, grid), (rows, per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )


def dram_sharded_weight_memcfg(k: int, n: int, mesh_device: ttnn.MeshDevice) -> ttnn.MemoryConfig:
    """Width-sharded DRAM memory config for a ``[k, n]`` projection weight.

    One shard per DRAM bank (8 on this part), which is what
    ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` requires of
    ``input_tensor_b`` (``matmul_device_operation.cpp:1312``) and what
    ``ttnn.experimental.minimal_matmul`` also accepts, so prefill and decode share
    one weight tensor.  ``n`` is padded up to ``32 * dram_cores``; all five
    projection widths in this layer (4608, 4096, 6656, 19968) already divide it.
    """
    dram_grid = mesh_device.dram_grid_size()
    if dram_grid.y != 1:
        raise ValueError(f"DRAM weight sharding assumes a 1-row DRAM grid, got {dram_grid}")
    cores = dram_grid.x
    padded = math.ceil(n / (TILE_SIZE * cores)) * (TILE_SIZE * cores)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))]),
            (k, padded // cores),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def decode_matmul_program_config(
    rows: int, n: int, cores: int, in0_block_w: int, activation: ttnn.UnaryOpType | None = None
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    """DRAM-sharded decode matmul program config.

    ``per_core_M`` is always 1: the op hard-requires a single M tile
    (``matmul_device_operation.cpp:1287``, ``M == 1``), and a decode step is
    always exactly one tile-row because ``nlp_create_qkv_heads_decode`` caps
    ``num_users`` at 32.

    ``activation`` is folded into the matmul rather than left on the following
    elementwise op; see :data:`DECODE_FUSED_ACTIVATION`.
    """
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=rows // TILE_SIZE,
        per_core_N=math.ceil(n / (TILE_SIZE * cores)),
        fused_activation=None if activation is None else ttnn.UnaryWithParam(activation),
    )


def minimal_matmul_blocks(role: str, rows: int, dtype: ttnn.DataType) -> tuple[int, int, int] | None:
    """Tuned ``(M_block, K_block, N_block)`` for a prefill projection, or ``None``.

    ``None`` -- from a missing ``(role, dtype)``, a row count below every
    threshold, or an explicit ``None`` entry -- means "pass no ``config=``", i.e.
    use the op's own ``M=K=N=8`` choice.  An unswept dtype (BF16, which no shipped
    policy uses) therefore degrades to the op default rather than to an entry
    measured for a different tile size.
    """
    for min_rows, blocks in PREFILL_MINIMAL_BLOCKS.get((role, dtype), ()):
        if rows >= min_rows:
            return blocks
    return None


# --------------------------------------------------------------------- submodules


class _OptimizedMLP(LightweightModule):
    """SwiGLU MLP over DRAM width-sharded BFP4 weights.

    ``forward`` is the prefill form (DRAM-interleaved activations,
    ``minimal_matmul``) and is what the inherited ``_prefill_chunk`` calls.
    ``decode_forward`` is the width-sharded L1 form: three DRAM-sharded matmuls
    on the MLP working grid, with the SiLU still folded into the gating multiply
    as the fusing stage established.

    The gate and up projections are kept **separate** rather than packed into one
    39936-wide matmul.  Packing was measured at every legal geometry
    (``doc/optimized_decoder/logs/decode_matmul_geometry_packed.log``): the wide
    output forces ``in0_block_w <= 2``, so the packed matmul costs 0.4851 ms
    against 0.4600 ms for the two separate dispatches -- *before* the slice that
    splits the halves apart.  See ``$optimize`` OPT-010.
    """

    def __init__(
        self,
        gate: ttnn.Tensor,
        up: ttnn.Tensor,
        down: ttnn.Tensor,
        activation_dtype: ttnn.DataType,
        owner: "OptimizedDecoder | None" = None,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down
        self.activation_dtype = activation_dtype
        #: Set by ``OptimizedDecoder.__init__``; carries the compute-kernel
        #: configs and the geometry tables.
        self.owner = owner

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Prefill MLP: DRAM-interleaved in and out."""
        dec = self.owner
        gate = dec._prefill_projection(x, self.gate, role="mlp_gate")
        up = dec._prefill_projection(x, self.up, role="mlp_up")
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = dec._prefill_projection(hidden, self.down, role="mlp_down")
        ttnn.deallocate(hidden)
        return out

    def decode_forward(self, x_sharded: ttnn.Tensor, rows: int) -> ttnn.Tensor:
        """Decode MLP: width-sharded L1 in and out, on the MLP working grid.

        ``rows`` is the *tile-padded* row count (always 32), not the logical
        batch, because it sizes the shard height ($optimize OPT-005).
        """
        dec = self.owner
        # SiLU rides along on the gate matmul when DECODE_FUSED_ACTIVATION is set,
        # so the multiply below is a plain multiply rather than an SFPU one.
        fused = dec.decode_fused_activation
        gate = dec._decode_projection(
            x_sharded, self.gate, role="mlp_gate", rows=rows, activation=ttnn.UnaryOpType.SILU
        )
        up = dec._decode_projection(x_sharded, self.up, role="mlp_up", rows=rows)
        out_memcfg = gate.memory_config()
        wide = dec.decode_swiglu_mul_cores
        if wide is not None:
            # See :data:`DECODE_SWIGLU_MUL_CORES`: three reshards to spread the SFPU
            # SiLU over more cores.  ``mlp_down``'s ``in0_block_w`` is derived from
            # the *gate/up* grid, so the product has to come back to it.
            #
            # 80 divides this checkpoint's 160-tile local intermediate width, and that is
            # a property of (intermediate_size, tp), not a constant.  Round 6 pointed out
            # that a config change would otherwise produce an uneven shard silently rather
            # than failing, so the divisibility that makes the reshard legal is asserted
            # where it is relied on.
            width_tiles = int(gate.shape[-1]) // ttnn.TILE_SIZE
            if width_tiles % wide:
                raise ValueError(
                    f"DECODE_SWIGLU_MUL_CORES={wide} must divide the local intermediate width in "
                    f"tiles ({width_tiles} for shape {tuple(gate.shape)}); an uneven width shard "
                    "is silently wrong rather than an error. Pick a divisor or set it to None."
                )
            wide_memcfg = dec._sharded_memcfg(rows, int(gate.shape[-1]), wide)
            gate_w = ttnn.to_memory_config(gate, wide_memcfg)
            ttnn.deallocate(gate)
            up_w = ttnn.to_memory_config(up, wide_memcfg)
            ttnn.deallocate(up)
            gate, up, mul_memcfg = gate_w, up_w, wide_memcfg
        else:
            mul_memcfg = out_memcfg
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[] if fused else [ttnn.UnaryOpType.SILU],
            dtype=self.activation_dtype,
            memory_config=mul_memcfg,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        if wide is not None:
            narrow = ttnn.to_memory_config(hidden, out_memcfg)
            ttnn.deallocate(hidden)
            hidden = narrow
        out = dec._decode_projection(hidden, self.down, role="mlp_down", rows=rows)
        ttnn.deallocate(hidden)
        return out


# ----------------------------------------------------------------------- the layer


class OptimizedDecoder(FusedDecoder):
    """Optimized TTNN implementation of ``MuseGlimmerTextDecoderLayer``."""

    def __init__(
        self,
        *,
        precision: PrecisionPolicy = DEFAULT_PRECISION,
        decode_matmul: dict[str, tuple[int, int]] | None = None,
        boundary_cores: int = BOUNDARY_CORES,
        decode_sdpa: tuple[int, int, int, int] | None = None,
        decode_fused_activation: bool = DECODE_FUSED_ACTIVATION,
        decode_swiglu_mul_cores: int | None = DECODE_SWIGLU_MUL_CORES,
        sharded_decode_io: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.precision = precision
        self.decode_fused_activation = decode_fused_activation
        self.decode_swiglu_mul_cores = resolve_decode_swiglu_mul_cores(
            self.config.intermediate_size,
            decode_swiglu_mul_cores,
        )
        #: Return the decode residual width-sharded in L1 instead of DRAM
        #: interleaved -- the inter-layer contract described in ``decode_forward``.
        self.sharded_decode_io = sharded_decode_io
        #: **One** persistent zero Q filler, at the full sliding-window length; see
        #: ``_prefill_sdpa_sliding``.  Shorter tails slice it rather than allocate.
        self._q_filler: ttnn.Tensor | None = None
        table = dict(decode_matmul if decode_matmul is not None else DECODE_MATMUL)
        # Resolve ``(role, dtype) -> (cores, in0_block_w)`` down to this policy's
        # ``role -> (cores, in0_block_w)`` once, so the hot path is a dict hit.
        # A caller-supplied table may already be role-keyed, which is what the
        # geometry A/B harness passes.
        self.decode_matmul = {
            role: table[(role, precision.weight_dtype(role))]
            if (role, precision.weight_dtype(role)) in table
            else table[role]
            for role in ("wqkv", "attn_gate", "o_proj", "mlp_gate", "mlp_up", "mlp_down")
        }
        if len({self.decode_matmul[r][0] for r in ("mlp_gate", "mlp_up", "mlp_down")}) != 1:
            raise ValueError(
                "the MLP gate/up/down roles must share a core count: down consumes the gate/up product, "
                f"got {[self.decode_matmul[r] for r in ('mlp_gate', 'mlp_up', 'mlp_down')]}"
            )
        for role in ("wqkv", "attn_gate"):
            if self.decode_matmul[role][0] != boundary_cores:
                raise ValueError(
                    f"{role} must run on the boundary grid ({boundary_cores} cores) so the residual, norms "
                    f"and residual adds need no reshard; got {self.decode_matmul[role][0]}"
                )
        # ``o_proj`` is the one attention role allowed off the boundary grid
        # ($optimize OPT-011).  Its ``in0`` is the *gated attention output*, not
        # the residual, so a narrower working shard costs one ``ttnn.reshard`` of
        # that tensor and nothing else -- the projection's own output goes
        # straight into the row-parallel reduction (multichip) or the boundary
        # memory config (single chip), never back into a residual add.  The
        # reason to want one is that ``o_proj``'s K is the *attention* width, the
        # smallest K in the layer, so the boundary grid can leave it with too few
        # K-tiles per core for a useful ``in0_block_w``.
        oproj_cores = self.decode_matmul["o_proj"][0]
        attn_width = self.config.num_attention_heads * self.config.head_dim
        if attn_width // TILE_SIZE % oproj_cores:
            raise ValueError(
                f"o_proj runs on {oproj_cores} cores, which must divide the {attn_width // TILE_SIZE} "
                f"attention-output tiles so its in0 shard is not padded"
            )
        self.boundary_cores = boundary_cores
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.device_grid = grid
        #: Pins the prefill 2D-multicast matmul's core-column count.  Anything
        #: else silently miscomputes against a width-sharded DRAM weight -- see
        #: :data:`PREFILL_MCAST_GRID_X`.
        self.dram_banks = self.mesh_device.dram_grid_size().x

        if self.config.hidden_size // TILE_SIZE % boundary_cores:
            raise ValueError(
                f"boundary_cores={boundary_cores} must divide {self.config.hidden_size // TILE_SIZE} "
                f"hidden-size tiles so no boundary tensor is shard-padded"
            )
        # ``_decode_norm_configs`` derives the norm grid from ``boundary_cores``
        # instead of the fused stage's ``choose_decode_norm_grid``, so the norms
        # land on exactly the matmuls' core set.
        self.decode_norm_grid = (min(boundary_cores, grid.x), math.ceil(boundary_cores / grid.x))
        self._decode_norm_cache = {}
        self._prefill_norm_cache: dict[int, tuple] = {}
        self._memcfg_cache: dict[tuple[int, int, int], ttnn.MemoryConfig] = {}

        def _ck(fidelity: ttnn.MathFidelity):
            return ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )

        self.decode_compute_kernel_config = _ck(precision.decode_math_fidelity)
        self.dense_compute_kernel_config = _ck(precision.prefill_math_fidelity)
        #: ``role -> compute kernel config``, so a policy can put the attention
        #: projections on a different math fidelity from the MLP ones.  Built by
        #: role rather than by fidelity value so a reader of the built layer can
        #: see exactly what each projection runs at
        #: (``fidelity_report()``), which is what proves the selected policy is
        #: the one the measured matmuls used.
        self.decode_compute_kernel_config_by_role = {
            role: _ck(precision.decode_fidelity(role)) for role in PROJECTION_ROLES
        }
        self.prefill_compute_kernel_config_by_role = {
            role: _ck(precision.prefill_fidelity(role)) for role in PROJECTION_ROLES
        }
        # Attention decode SDPA (OPT-002).  The fused stage inherited the whole
        # 11x10 grid with ``q_chunk=32, k_chunk=64``; this stage measured the
        # ``(8, 8) / q_chunk=0 / k_chunk=0`` candidate the skill names and it is
        # **18 % of the whole decode step** at 131071 on the ``full`` (NoPE)
        # layer, where the SDPA reads the entire cache
        # (1.5586 -> 1.2723 ms/token).  See DEFAULT_DECODE_SDPA.
        gx, gy, q_chunk, k_chunk = decode_sdpa or DEFAULT_DECODE_SDPA
        gx, gy = gx or grid.x, gy or grid.y
        self.decode_sdpa = (gx, gy, q_chunk, k_chunk)
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        if isinstance(self.mlp, _OptimizedMLP):
            self.mlp.owner = self

    # ------------------------------------------------------- precision readback

    #: ``role -> attribute path on the built layer`` for the six projection
    #: weights.  Used only by :meth:`precision_report`.
    _ROLE_WEIGHT_ATTRS = {
        "wqkv": ("wqkv",),
        "attn_gate": ("w_attn_gate",),
        "o_proj": ("wo",),
        "mlp_gate": ("mlp", "gate"),
        "mlp_up": ("mlp", "up"),
        "mlp_down": ("mlp", "down"),
    }

    def precision_report(self) -> dict[str, Any]:
        """What this **built** layer runs at, read off the device tensors.

        The dtype comes from the packed weight tensor and the fidelity from the
        compute-kernel config the projection's ``ttnn.linear`` is handed, so a
        policy field that the constructor silently ignored cannot appear here.
        That is the propagation evidence ``$datatype-sweep`` asks for: a JSON
        field is a request, this is what the matmuls actually got.
        """
        roles: dict[str, dict[str, str]] = {}
        for role, path in self._ROLE_WEIGHT_ATTRS.items():
            weight: Any = self
            for name in path:
                weight = getattr(weight, name)
            roles[role] = {
                "weight_dtype": str(weight.dtype),
                "decode_fidelity": str(self.decode_compute_kernel_config_by_role[role].math_fidelity),
                "prefill_fidelity": str(self.prefill_compute_kernel_config_by_role[role].math_fidelity),
                "decode_cores": self.decode_matmul[role][0],
                "decode_in0_block_w": self.decode_matmul[role][1],
            }
        return {
            "policy_name": self.precision.name,
            "layer_idx": self.config.layer_idx,
            "layer_kind": self.config.layer_kind,
            "activation_dtype": str(self.activation_dtype),
            "kv_cache_dtype": str(self.k_cache.dtype),
            "kv_cache_dtype_requested": str(self.kv_cache_dtype),
            "roles": roles,
        }

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
        weight_dtype: ttnn.DataType | None = None,
        activation_dtype: ttnn.DataType | None = None,
        kv_cache_dtype: ttnn.DataType | None = None,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        precision: PrecisionPolicy = DEFAULT_PRECISION,
        decode_matmul: dict[str, tuple[int, int]] | None = None,
        boundary_cores: int = BOUNDARY_CORES,
        decode_sdpa: tuple[int, int, int, int] | None = None,
        decode_fused_activation: bool = DECODE_FUSED_ACTIVATION,
        decode_swiglu_mul_cores: int | None = DECODE_SWIGLU_MUL_CORES,
        sharded_decode_io: bool = False,
        rope_cache: dict[str, ttnn.Tensor] | None = None,
        **kwargs,
    ) -> "OptimizedDecoder":
        """Same contract as ``FusedDecoder.from_state_dict``, plus ``precision``.

        ``weight_dtype`` / ``activation_dtype`` / ``kv_cache_dtype`` stay in the
        signature so the earlier stages' callers keep working, but they are now
        *overrides* on top of ``precision``: passing ``weight_dtype`` sets every
        projection group to it, which is what a caller asking for a uniform dtype
        means.  Leaving them ``None`` uses the tuned per-group policy.
        """
        if kwargs:
            raise TypeError(f"Unexpected OptimizedDecoder.from_state_dict kwargs: {sorted(kwargs)}")
        if mesh_device.get_num_devices() != 1:
            raise ValueError("OptimizedDecoder is the single-chip stage; use a 1x1 MeshDevice.")

        precision = _override_precision(precision, weight_dtype, activation_dtype, kv_cache_dtype)

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
            folded = (1.0 + weight).to(torch.bfloat16)
            tile = _to_device(folded.reshape(1, 1, 1, config.hidden_size), mesh_device=mesh_device, dtype=ttnn.bfloat16)
            row_major = _to_device(
                folded.reshape(1, 1, config.hidden_size // TILE_SIZE, TILE_SIZE),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            return _FusedNorm(tile, row_major, eps, norm_ck)

        def linear_weight(suffix: str) -> torch.Tensor:
            # HF stores nn.Linear weights as [out, in]; the matmuls want [in, out].
            return _get_layer_tensor(state_dict, layer_idx, suffix).to(torch.float32).transpose(-2, -1).contiguous()

        def projection(tensor: torch.Tensor, role: str) -> ttnn.Tensor:
            """One DRAM width-sharded weight, shared by prefill and decode."""
            k, n = tensor.shape[-2], tensor.shape[-1]
            return ttnn.from_torch(
                tensor.reshape(1, 1, k, n),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=precision.weight_dtype(role),
                memory_config=dram_sharded_weight_memcfg(k, n, mesh_device),
            )

        wq = linear_weight("self_attn.q_proj.weight")
        wk = linear_weight("self_attn.k_proj.weight")
        wv = linear_weight("self_attn.v_proj.weight")
        wqkv = torch.cat([wq, wk, wv], dim=-1)

        mlp = _OptimizedMLP(
            gate=projection(linear_weight("mlp.gate_proj.weight"), "mlp_gate"),
            up=projection(linear_weight("mlp.up_proj.weight"), "mlp_up"),
            down=projection(linear_weight("mlp.down_proj.weight"), "mlp_down"),
            activation_dtype=precision.activation_dtype,
        )

        cache_shape = (max_num_blocks, config.num_key_value_heads, page_block_size, config.head_dim)
        k_cache = _to_device(torch.zeros(cache_shape), mesh_device=mesh_device, dtype=precision.kv_cache_dtype)
        v_cache = _to_device(torch.zeros(cache_shape), mesh_device=mesh_device, dtype=precision.kv_cache_dtype)

        cos_cache = sin_cache = cos_cache_tile = sin_cache_tile = None
        if config.uses_rope:
            if rope_cache is not None:
                # Full-model P150 serving uses this single-chip decoder for every
                # layer.  The 39 sliding layers share one theta, so keeping a copy
                # of the four full-context tables in every layer would waste about
                # 5.2 GB of device DRAM.  The model-level builder validates the
                # common theta before handing this cache in.
                cos_cache = rope_cache["cos"]
                sin_cache = rope_cache["sin"]
                cos_cache_tile = rope_cache["cos_tile"]
                sin_cache_tile = rope_cache["sin_tile"]
            else:
                cos, sin = _rope_cos_sin(max_seq_len, config.head_dim, config.rope_theta)
                cos_cache = _to_device(
                    cos.to(torch.bfloat16),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                sin_cache = _to_device(
                    sin.to(torch.bfloat16),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
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
            wqkv=projection(wqkv, "wqkv"),
            w_attn_gate=projection(linear_weight("self_attn.gate_proj.weight"), "attn_gate"),
            wo=projection(linear_weight("self_attn.o_proj.weight"), "o_proj"),
            k_cache=k_cache,
            v_cache=v_cache,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            cos_cache_tile=cos_cache_tile,
            sin_cache_tile=sin_cache_tile,
            activation_dtype=precision.activation_dtype,
            kv_cache_dtype=precision.kv_cache_dtype,
            precision=precision,
            decode_matmul=decode_matmul,
            boundary_cores=boundary_cores,
            decode_sdpa=decode_sdpa,
            decode_fused_activation=decode_fused_activation,
            decode_swiglu_mul_cores=decode_swiglu_mul_cores,
            sharded_decode_io=sharded_decode_io,
        )

    # -------------------------------------------------------------- projections

    def _sharded_memcfg(self, rows: int, width: int, cores: int) -> ttnn.MemoryConfig:
        key = (rows, width, cores)
        cached = self._memcfg_cache.get(key)
        if cached is None:
            cached = width_sharded_l1(rows, width, cores, self.device_grid)
            self._memcfg_cache[key] = cached
        return cached

    def boundary_memcfg(self, rows: int, width: int) -> ttnn.MemoryConfig:
        """Width-sharded L1 config for a boundary-grid tensor of ``width``."""
        return self._sharded_memcfg(rows, width, self.boundary_cores)

    def _decode_norm_configs(self, rows: int):
        """``(program_config, boundary width-sharded memory_config)`` for ``rows``.

        Overridden so the four hidden-size decode RMSNorms consume and produce
        **exactly** the memory config the boundary-grid matmuls use.  The fused
        stage built the norm's config with ``ttnn.create_sharded_memory_config``
        over a ``gx * gy`` rectangle; here both come from
        :func:`width_sharded_l1`, whose core set is the row-major prefix the
        DRAM-sharded matmul is measured on, so the residual, the norms, the
        ``o_proj`` output and the residual adds all share one spec and no reshard
        is needed to cross between them.  The one measured cost is the norm's core
        rectangle: 8 cores as ``8x1`` instead of ``4x2``, which the fused stage's
        probe puts at 23.0 vs 22.8 us
        (``doc/fused_decoder/logs/norm_shard_probe.log``).
        """
        cached = self._decode_norm_cache.get(rows)
        if cached is not None:
            return cached
        dim = self.config.hidden_size
        cores = self.boundary_cores
        block_w = dim // cores // TILE_SIZE
        memory_config = self._sharded_memcfg(rows, dim, cores)
        # The grid has to cover the row-major prefix the DRAM-sharded matmul wrote,
        # which is wider than the shard whenever ``cores`` is not a multiple of the
        # grid width.  That is safe at block_h == 1 and *only* there.
        gx = min(cores, self.device_grid.x)
        gy = math.ceil(cores / self.device_grid.x)
        if gx * gy != cores and rows // TILE_SIZE > 1:
            raise ValueError(
                f"the decode norm program grid {gx}x{gy} covers {gx * gy} cores against a "
                f"{cores}-core shard, which silently returns inf at block_h={rows // TILE_SIZE} > 1 "
                "(doc/optimized_decoder/logs/sharded_norm_grid_probe.log); a decode step must be one "
                "tile row"
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

    def _decode_projection(
        self,
        x_sharded: ttnn.Tensor,
        weight: ttnn.Tensor,
        *,
        role: str,
        rows: int,
        activation: ttnn.UnaryOpType | None = None,
    ) -> ttnn.Tensor:
        """One DRAM-sharded decode matmul, width-sharded L1 in and out.

        The activation must already be width-sharded on this role's core count;
        ``decode_forward`` and ``_OptimizedMLP.decode_forward`` arrange that, and
        ``_reshard_to`` pays for the one boundary crossing the MLP needs.

        ``activation`` folds a unary op into the matmul's packer/SFPU stage
        (``SFPU_ACTIVATION`` in
        ``matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:349``)
        instead of leaving it on the following ``ttnn.mul``.  See
        :data:`DECODE_FUSED_ACTIVATION`.
        """
        cores, in0_block_w = self.decode_matmul[role]
        n = int(weight.shape[-1])
        program_config = decode_matmul_program_config(
            rows, n, cores, in0_block_w, activation=activation if self.decode_fused_activation else None
        )
        return ttnn.linear(
            x_sharded,
            weight,
            dtype=self.activation_dtype,
            memory_config=self._sharded_memcfg(rows, n, cores),
            program_config=program_config,
            compute_kernel_config=self.decode_compute_kernel_config_by_role[role],
        )

    def _prefill_projection(self, x: ttnn.Tensor, weight: ttnn.Tensor, *, role: str) -> ttnn.Tensor:
        """One prefill projection over the DRAM width-sharded weight.

        Three kernels, by measured row count, all reading the *same* weight
        tensor:

        1. **exactly one M tile** -> the DRAM-sharded decode matmul, 3.8x faster
           than ``minimal_matmul`` (0.0575 vs 0.2168 ms on the ``wqkv`` shape),
           with an explicit reshard in and out because prefill activations are
           DRAM interleaved.  Above one tile the op refuses outright
           (``matmul_device_operation.cpp:1287``, ``M == 1``).
        2. **two tiles to ~1024 rows** -> ``ttnn.linear`` with an explicit
           **2D-multicast** program config, 1.3-2.0x faster than
           ``minimal_matmul``.  See :data:`PREFILL_MCAST2D`: the first pass of
           this stage rejected ``ttnn.linear`` here on the auto-selected config's
           "Input B memory layout must be INTERLEAVED", but the 2D-multicast
           validator accepts a width-sharded *DRAM* in1.
        3. **above the per-role crossover** -> ``minimal_matmul`` with the swept
           per-shape blocking.  Once the matmul is compute- rather than
           launch-bound, ``minimal_matmul``'s full 11x10 grid beats the 8 core
           columns the 2D-multicast path is pinned to by
           :data:`PREFILL_MCAST_GRID_X`.
        """
        rows = int(x.shape[-2])
        if rows == TILE_SIZE:
            cores, _ = self.decode_matmul[role]
            x_sharded = ttnn.interleaved_to_sharded(x, self._sharded_memcfg(rows, int(x.shape[-1]), cores))
            out_sharded = self._decode_projection(x_sharded, weight, role=role, rows=rows)
            ttnn.deallocate(x_sharded)
            out = ttnn.sharded_to_interleaved(out_sharded, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out_sharded)
            return out
        mcast = prefill_mcast2d_spec(role, rows, weight.dtype)
        if mcast is not None:
            grid_y, in0_block_w = mcast
            return ttnn.linear(
                x,
                weight,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.prefill_compute_kernel_config_by_role[role],
                program_config=prefill_mcast2d_program_config(
                    rows, int(weight.shape[-1]), grid_y, in0_block_w, self.dram_banks
                ),
            )
        blocks = minimal_matmul_blocks(role, rows, weight.dtype)
        config = None
        if blocks is not None:
            m_block, k_block, n_block = blocks
            subblock_h, subblock_w = (2, 4) if int(weight.shape[-1]) >= rows else (4, 2)
            config = ttnn.MinimalMatmulConfig(
                M_block_size=m_block,
                K_block_size=k_block,
                N_block_size=n_block,
                subblock_h=subblock_h,
                subblock_w=subblock_w,
                compute_with_storage_grid_size=self.device_grid,
            )
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.activation_dtype,
            compute_kernel_config=self.prefill_compute_kernel_config_by_role[role],
            config=config,
        )

    # The two seams ``FusedDecoder`` kept so a packed shared-LHS variant could be
    # A/B'd without forking the forward.  Both are re-used here; the packed
    # candidate is in ``doc/optimized_decoder/bench/variants.py``.

    def _project_qkv(self, normed: ttnn.Tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        return self._prefill_projection(normed, self.wqkv, role="wqkv")

    def _attn_gate(self, normed: ttnn.Tensor) -> ttnn.Tensor:
        return self._prefill_projection(normed, self.w_attn_gate, role="attn_gate")

    def _prefill_o_proj(self, gated: ttnn.Tensor) -> ttnn.Tensor:
        return self._prefill_projection(gated, self.wo, role="o_proj")

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
        """Prefill attention.

        Structurally identical to ``FusedDecoder._prefill_attention`` -- same head
        creation, per-head norms, ``rotary_embedding_hf``, paged fill, SDPA call
        sites, gating fold.  It is re-spelled here for exactly one reason: the
        fused version's ``o_proj`` goes through the module-level ``_dense``, whose
        sub-3072-row branch is ``ttnn.linear``, and ``ttnn.linear`` cannot take
        this stage's width-sharded weight at all
        (``matmul_device_operation.cpp:1233``).  ``_project_qkv`` and
        ``_attn_gate`` were already overridable seams; ``o_proj`` was not.
        """
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

        # Paged KV fill.  ``paged_fill_cache`` does no dtype conversion, so cast to
        # the cache dtype first -- which this stage actually exercises, because the
        # cache is BFP8 and K/V are BF16 ($optimize OPT-002).
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
        projected = self._prefill_o_proj(gated)
        ttnn.deallocate(gated)
        return projected, next_tail

    # ----------------------------------------------------- sharded prefill norm

    def _prefill_norm_configs(self, rows: int):
        """``(program_config, memory_config)`` for a width-sharded prefill norm."""
        cached = self._prefill_norm_cache.get(rows)
        if cached is not None:
            return cached
        dim = self.config.hidden_size
        cores = PREFILL_NORM_SHARD_CORES
        gx, gy = core_rectangle(cores, self.device_grid)
        block_w = dim // cores // TILE_SIZE
        # An exact rectangle, so the program grid matches the shard exactly; this is
        # the only sharded-norm shape measured correct above block_h == 1.
        memory_config = rect_width_sharded_l1(rows, dim, cores, self.device_grid)
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=_norm_subblock_w(block_w),
            block_h=rows // TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )
        self._prefill_norm_cache[rows] = (program_config, memory_config)
        return program_config, memory_config

    def _prefill_norm(self, norm: _FusedNorm, x: ttnn.Tensor) -> ttnn.Tensor:
        """One hidden-size prefill RMSNorm, sharded when that is legal and faster.

        Above :data:`PREFILL_NORM_SHARD_MAX_ROWS` the shard does not fit L1 and the
        interleaved norm has enough tile rows to fill the grid anyway, so this
        falls through to the inherited call.  The conversions are inside the
        candidate that was measured, not hidden outside it.
        """
        rows = int(x.shape[-2])
        if rows > PREFILL_NORM_SHARD_MAX_ROWS or rows % TILE_SIZE:
            return norm(x)
        program_config, memory_config = self._prefill_norm_configs(rows)
        x_sharded = ttnn.interleaved_to_sharded(x, memory_config)
        out_sharded = norm.sharded_forward(x_sharded, program_config, memory_config)
        ttnn.deallocate(x_sharded)
        out = ttnn.sharded_to_interleaved(out_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_sharded)
        return out

    def _prefill_chunk(
        self,
        hidden_states: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        user_id: int,
        start_pos: int,
        sliding_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        need_tail: bool,
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor] | None]:
        """As ``FunctionalDecoder._prefill_chunk``, with the four norms routed
        through :meth:`_prefill_norm`.

        Identical dataflow otherwise: same residual structure, same
        ``_prefill_attention``, same MLP, same DRAM-interleaved residual adds.
        """
        residual = hidden_states
        normed = self._prefill_norm(self.input_layernorm, residual)
        attn, next_tail = self._prefill_attention(
            normed,
            page_table=page_table,
            user_id=user_id,
            start_pos=start_pos,
            sliding_tail=sliding_tail,
            need_tail=need_tail,
        )
        ttnn.deallocate(normed)
        attn = self._prefill_norm(self.post_attention_layernorm, attn)
        hidden = ttnn.add(residual, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        mlp_in = self._prefill_norm(self.pre_feedforward_layernorm, hidden)
        mlp_out = self.mlp(mlp_in)
        ttnn.deallocate(mlp_in)
        mlp_out = self._prefill_norm(self.post_feedforward_layernorm, mlp_out)
        out = ttnn.add(hidden, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_out)
        return out, next_tail

    # ------------------------------------------------- sliding prefill filler

    def _prefill_sdpa_sliding(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        sliding_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        need_tail: bool,
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor] | None]:
        """As ``FunctionalDecoder._prefill_sdpa_sliding``, with a *persistent* filler.

        The inherited version builds the zero Q filler with ``ttnn.zeros(...,
        device=...)`` at every internal sliding chunk boundary.  That is not a
        device op: ``ttnn::creation_detail::full_impl``
        (``ttnn/cpp/ttnn/operations/creation/creation.cpp:51-73``) allocates a host
        ``std::vector``, fills it, and uploads it.  At the real shape --
        ``[1, 32 heads, 2048 window, 128]`` BF16 = 16,777,216 B -- it showed up as
        a **2015.9 us** op-to-op gap in the warmed two-chunk sliding prefill
        profile, against 33.8 us total for the whole ``full`` window that runs the
        same Python chunk loop without needing a filler.  8.3 GB/s is a host PCIe
        write, not a kernel.

        The filler is a constant: same shape, same dtype, all zeros, every chunk.
        So **one** is built, at the full ``sliding_window`` length, and a shorter
        tail slices it.  Keying a cache on ``tail_len`` instead would be unbounded
        in a way the test suite cannot see: ``tail_len`` is
        ``sliding_kv_tail_len(start_pos) = min(window, start_pos)``, i.e.
        caller-controlled, so a decoder reused across continuation prefills at
        different offsets could accumulate one entry per tile-aligned offset up to
        the window -- 64 entries x up to 8 MB on this model.  One buffer of
        ``n_heads * window * head_dim * 2 B`` = 16,777,216 B is the whole cost, and
        it is recorded in ``doc/context_contract.json`` under
        ``implementation.extra_persistent_buffers``.

        Everything else is byte-for-byte the inherited algorithm, including the
        square ``[previous-window tail | this chunk]`` slice, the front zero
        padding, the output slice, and the tail carry.
        """
        if sliding_tail is None:
            return super()._prefill_sdpa_sliding(q, k, v, sliding_tail, need_tail)

        cfg = self.config
        window = cfg.sliding_window
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        seq_len = q.shape[-2]

        k_tail, v_tail = sliding_tail
        tail_len = int(k_tail.shape[-2])
        if self._q_filler is None or self._q_filler.dtype != q.dtype:
            if self._q_filler is not None:
                ttnn.deallocate(self._q_filler)
            self._q_filler = ttnn.zeros(
                [1, n_heads, window, head_dim],
                dtype=q.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if tail_len == window:
            q_filler, owns_filler = self._q_filler, False
        else:
            # A continuation prefill whose tail is shorter than the window: slice
            # the persistent buffer instead of uploading another one.
            q_filler = ttnn.slice(self._q_filler, [0, 0, 0, 0], [1, n_heads, tail_len, head_dim])
            owns_filler = True

        q_padded = ttnn.concat([q_filler, q], dim=2)
        if owns_filler:
            ttnn.deallocate(q_filler)
        k_cat = ttnn.concat([k_tail, k], dim=2)
        v_cat = ttnn.concat([v_tail, v], dim=2)
        ttnn.deallocate(k_tail)
        ttnn.deallocate(v_tail)
        full = ttnn.transformer.scaled_dot_product_attention(
            q_padded,
            k_cat,
            v_cat,
            is_causal=True,
            scale=cfg.sdpa_scale,
            sliding_window_size=window,
            program_config=self._prefill_program_config(tail_len + seq_len),
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        ttnn.deallocate(q_padded)
        attn = ttnn.slice(full, [0, 0, tail_len, 0], [1, n_heads, tail_len + seq_len, head_dim])
        ttnn.deallocate(full)

        next_tail = None
        if need_tail:
            source_len = int(k_cat.shape[-2])
            tail_start = max(0, source_len - window)
            next_tail = (
                ttnn.clone(
                    ttnn.slice(k_cat, [0, 0, tail_start, 0], [1, n_kv, source_len, head_dim]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.clone(
                    ttnn.slice(v_cat, [0, 0, tail_start, 0], [1, n_kv, source_len, head_dim]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        ttnn.deallocate(k_cat)
        ttnn.deallocate(v_cat)
        return attn, next_tail

    # ------------------------------------------------------------- decode path

    def _reshard_to(self, tensor: ttnn.Tensor, cores: int, rows: int) -> ttnn.Tensor:
        """Move a width-sharded L1 tensor onto ``cores`` cores, or pass it through.

        This is the price of the phase-specific MLP working shard: two of these
        per decode step, ~2 us each, against the 57 us the wider MLP
        ``in0_block_w`` buys ($optimize OPT-011).
        """
        spec = tensor.memory_config().shard_spec
        if spec is not None and spec.grid.num_cores() == cores:
            return tensor
        target = self._sharded_memcfg(rows, int(tensor.shape[-1]), cores)
        out = ttnn.reshard(tensor, target)
        ttnn.deallocate(tensor)
        return out

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        rope_pos_ids: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Single-token paged decode; see ``FunctionalDecoder`` for the contract.

        The whole step stays width-sharded in L1 on the boundary grid, except the
        MLP, which runs on its own working grid (two reshards), and the QKV
        hand-off to ``nlp_create_qkv_heads_decode``, which needs L1 interleaved.
        """
        cfg = self.config
        batch = int(hidden_states.shape[-2])
        if hidden_states.shape[-1] != cfg.hidden_size:
            raise ValueError(f"decode expects hidden size {cfg.hidden_size}, got {hidden_states.shape[-1]}")
        if cfg.uses_rope and rope_pos_ids is None:
            raise ValueError("sliding (RoPE) layers require rope_pos_ids for the on-device cos/sin gather")

        rows = ((batch + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        norm_prg, norm_memcfg = self._decode_norm_configs(rows)

        # Inter-layer residual contract.  The decode residual is width-sharded in
        # L1 on the boundary grid for the whole layer; the only question is
        # whether the *layer boundary* is that layout too.  When the caller hands
        # one in already (``sharded_decode_io``), the entry
        # ``interleaved_to_sharded`` and the exit ``sharded_to_interleaved`` --
        # 2 x 425 KB of DRAM round trip per layer per token -- both disappear, and
        # a stacked model hands one layer's output straight to the next with no
        # conversion and no collective.  An interleaved input is still accepted so
        # the public contract is unchanged.
        aliased_input = hidden_states.is_sharded()
        if aliased_input and hidden_states.memory_config() != norm_memcfg:
            # A sharded input is taken as the residual as-is, so it has to *be*
            # the boundary contract rather than merely be sharded.  Silently
            # accepting a different shard spec here would hand the next layer a
            # wrong answer with no failing test, and this contract is the one
            # full-model bringup is being asked to preserve.
            raise ValueError(
                f"a sharded decode input must use the boundary memory config {norm_memcfg}, got "
                f"{hidden_states.memory_config()}; see doc/optimized_multichip_decoder/README.md"
            )
        residual = hidden_states if aliased_input else ttnn.interleaved_to_sharded(hidden_states, norm_memcfg)
        normed = self.input_layernorm.sharded_forward(residual, norm_prg, norm_memcfg)

        xqkv_sharded = self._decode_projection(normed, self.wqkv, role="wqkv", rows=rows)
        # nlp_create_qkv_heads_decode's interleaved DRAM reader zeroes odd Q rows
        # on Blackhole (tt-metal #16667), so the fused QKV is staged in L1
        # interleaved.  It cannot consume this width-sharded matmul output
        # directly either: the sharded path additionally requires
        # head_dim % shard_width == 0 (..._device_operation.cpp:56-72), and the
        # shard is 576 wide against a 128 head_dim.
        xqkv_l1 = ttnn.sharded_to_interleaved(xqkv_sharded, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(xqkv_sharded)
        q, k, v = self._create_qkv_heads_decode(xqkv_l1)
        ttnn.deallocate(xqkv_l1)
        q_memcfg = q.memory_config()
        kv_memcfg = k.memory_config()

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

        # The gate is computed here rather than before SDPA so that only the
        # 4096-wide gate, not the 6656-wide norm output, has to stay resident
        # across the attention kernel.
        gate = self._decode_projection(
            normed, self.w_attn_gate, role="attn_gate", rows=rows, activation=ttnn.UnaryOpType.SIGMOID
        )
        ttnn.deallocate(normed)
        attn_width = int(out.shape[-1])
        out_sharded = ttnn.interleaved_to_sharded(out, gate.memory_config())
        ttnn.deallocate(out)
        gated = ttnn.mul(
            out_sharded,
            gate,
            input_tensor_b_activations=[] if self.decode_fused_activation else [ttnn.UnaryOpType.SIGMOID],
            dtype=self.activation_dtype,
            memory_config=self._sharded_memcfg(rows, attn_width, self.decode_matmul["attn_gate"][0]),
        )
        ttnn.deallocate(out_sharded)
        ttnn.deallocate(gate)
        # Pass-through when ``o_proj`` shares the gate's grid, which is the
        # default; a narrower ``o_proj`` working shard (OPT-011) pays one reshard
        # of this 32-tile tensor to widen the projection's K block.
        gated = self._reshard_to(gated, self.decode_matmul["o_proj"][0], rows)
        attn_out = self._decode_projection(gated, self.wo, role="o_proj", rows=rows)
        ttnn.deallocate(gated)

        attn_normed = self.post_attention_layernorm.sharded_forward(attn_out, norm_prg, norm_memcfg)
        ttnn.deallocate(attn_out)
        hidden = ttnn.add(residual, attn_normed, memory_config=norm_memcfg)
        if not aliased_input:  # never free a tensor the caller still owns
            ttnn.deallocate(residual)
        ttnn.deallocate(attn_normed)

        mlp_in = self.pre_feedforward_layernorm.sharded_forward(hidden, norm_prg, norm_memcfg)
        mlp_in = self._reshard_to(mlp_in, self.decode_matmul["mlp_gate"][0], rows)
        mlp_out = self.mlp.decode_forward(mlp_in, rows)
        ttnn.deallocate(mlp_in)
        mlp_out = self._reshard_to(mlp_out, self.boundary_cores, rows)

        mlp_normed = self.post_feedforward_layernorm.sharded_forward(mlp_out, norm_prg, norm_memcfg)
        ttnn.deallocate(mlp_out)
        out_sharded = ttnn.add(hidden, mlp_normed, memory_config=norm_memcfg)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_normed)
        if self.sharded_decode_io:
            return out_sharded
        out = ttnn.sharded_to_interleaved(out_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_sharded)
        return out


def _override_precision(
    precision: PrecisionPolicy,
    weight_dtype: ttnn.DataType | None,
    activation_dtype: ttnn.DataType | None,
    kv_cache_dtype: ttnn.DataType | None,
) -> PrecisionPolicy:
    """Apply the earlier stages' flat dtype kwargs on top of a named policy."""
    from dataclasses import replace

    changes: dict[str, Any] = {}
    if weight_dtype is not None:
        changes.update(
            name=f"{precision.name}+weights={weight_dtype}",
            attn_weight_dtype=weight_dtype,
            mlp_gate_up_weight_dtype=weight_dtype,
            mlp_down_weight_dtype=weight_dtype,
        )
    if activation_dtype is not None:
        changes["activation_dtype"] = activation_dtype
    if kv_cache_dtype is not None:
        changes["kv_cache_dtype"] = kv_cache_dtype
    return replace(precision, **changes) if changes else precision
