# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GLMQANormQBProjection: q_a RMSNorm -> q_b projection as ONE dispatch, q_a never leaving L1.

This is the *chained* fusion `GLMQKVAProjection` deliberately was not. Those two projections
share an input rather than feeding each other, so there was no DRAM round-trip between them to
remove and fusing them bought ~0.4 us of the measured 9.43x -- the rest was
`DRAMStreamingMatmul` beating ttnn's matmul. Here the two stages genuinely chain, so the
intermediate is real and so is the boundary that disappears.

WHAT THE MODEL DOES TODAY (models/experimental/glm4_moe_lite/tt/attention_decode.py:327-328):

    q_a = w.q_a_layernorm(q_a, mode="decode")        # reads q_a, writes normed q_a
    q   = attn_linear(q_a, w.w_q_b, ...)             # reads normed q_a back

Three dispatches and two full trips through q_a's 768-wide activation: the norm reads it and
writes a normalized copy, then the matmul reads that copy. The normalized copy exists only to
be consumed by the very next op.

WHAT THIS OP DOES INSTEAD -- deferred normalization (FlashNorm), the same trick
`DeferredRMSNormMatmul` and `qa_projection(defer_norm=True)` use:

    (1/RMS(q_a)) * (q_a @ (gamma . W_q_b))  ==  RMSNorm(q_a, gamma) @ W_q_b

gamma folds into the weight offline, so the matmul runs on the *un-normalized* q_a and the
`1/RMS` scalar is applied in the matmul's DST epilogue. There is no elementwise pass over q_a
at all -- not fused, *absent*. `DRAMStreamingMatmul` already carries a `scalar` CB input for
the MoE routing weight (`out = scalar * (act @ W)`, dram_stream_common.py:390 / kernels/op.hpp:403),
and a per-row `1/RMS` is exactly the same shape of value, so no new kernel is needed.

    q_a ─┬─► SumOfSquares(scalar=1/sqrt(768), epsilon) ──► 1/RMS   [1 tile, same core]
         │                                                   │
         └─► DRAMStreamingMatmul(W', scalar=1/RMS) ◄──────────┘ ──► q [1, 5120]

WHY THE NORM IS REDUNDANT ACROSS CORES, ON PURPOSE. `DRAMStreamingMatmul` wants its activation
*replicated* height-sharded across the 8 DRAM-bank workers -- every core holds the full K (see
`_make_act_tensor`, tests/blaze/micro_ops/common/test_dram_streaming_matmul.py:153). So every core
can compute the identical `1/RMS` from its own replica. Each core reduces 1024 elements against
the 640x1024 matmul it then runs, and in exchange there is **zero cross-core traffic**: no
Gather, no Mcast, no handshake. That matters here specifically -- the Gather is what deadlocks
`glm_routed_expert` at GLM's dims (F11, 16 stuck sender frames), and `DeferredRMSNormMatmul`
reaches for both a Mcast and a Gather. This op reaches for neither.

WHY K IS PADDED 768 -> 1024. `SumOfSquares` reinterprets the 1x32-tile activation row into
standard compute tiles via `interpret_tile`, which needs the width to be a multiple of 512
(HALF) or 1024 (FULL). GLM's `q_lora_rank` is 768 and is a multiple of neither:
`interpret_tile(768)` silently returns one HALF tile, covering 512 of the 768 elements and
producing a wrong RMS. Zero-padding to 1024 lands on exactly one FULL 32x32 tile -- the
configuration `test_deferred_rmsnorm_matmul` validates on silicon (its K=1024 case) -- and
zeros are correct in both consumers: they add nothing to the sum of squares, and they multiply
the zero-padded rows 768..1023 of W'. The mean must still divide by the *logical* 768, so the
reduce scalar is `1/sqrt(768)`, not `1/sqrt(1024)`.

PRECISION. The scalar is read out of the CB as bf16 (`kernels/op.hpp:322` keeps 16 of 32 bits),
so `1/RMS` carries ~8 mantissa bits. It is applied uniformly to every element of the output row,
and PCC is invariant under a uniform scaling, so this costs the PCC gate essentially nothing --
but it does mean the op is not bit-exact against a fp32 norm. The accumulation that *does*
matter is the sum of 1024 squares inside `SumOfSquares`: set `fp32_dest_acc_en=True` on the
**FusedProgram**, not on this emit. RMSNorm at GLM's dims measured PCC 0.9865 accumulating in
bf16 against 0.9999 in fp32, and `RMSNorm.emit` deletes its own copy of the flag because the
program's ComputeConfigDescriptor is authoritative (blaze/ops/rmsnorm/op.py:143).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...blaze_op import BlazeOp, FusedOp, Input, Output
from ...fused_program import CBHandle, FusedProgram
from ...utils import compute_subblock_w, interpret_tile, round_up
from ..dram_streaming_matmul import DRAMStreamingMatmul
from ..dram_streaming_matmul.common import dram_bank_worker_cores
from ..sum_of_squares import SumOfSquares

# Storage tile width of a decode activation row. Blaze's 1x32 decode tile is the reason this
# path computes one real row where ttnn pads to 32.
TILE_W = 32

# Elements in one FULL (32x32) interpreted compute tile.
FULL_TILE_ELEMENTS = 32 * 32

# `mul_reduce_scalar_tile` addresses dest slots as 32x32 tiles, so the reduce cannot span more
# than 8 of them -- the same limit `interpret_tile_padded` enforces.
MAX_NORM_TILES = 8


def _pad_to_full_tile(logical_k: int) -> tuple[int, int]:
    """Round the norm width up to FULL-tile alignment. Returns ``(padded_k, num_tiles)``.

    Deliberately *not* ``interpret_tile_padded``, which minimizes padding and therefore answers
    HALF (16x32) x2 for GLM's 768. Both geometries cover the same 1024 elements and both are
    real code paths, but FULL x1 at exactly this width is the one
    ``test_deferred_rmsnorm_matmul`` exercises on silicon (its ``K=1024`` case), and the HALF
    path exists for ``TpRmsNorm``'s per-head scatter. With no way to validate on hardware from
    here, matching the validated geometry is worth the extra padding.
    """
    padded_k = round_up(logical_k, FULL_TILE_ELEMENTS)
    num_tiles = padded_k // FULL_TILE_ELEMENTS
    if num_tiles > MAX_NORM_TILES:
        raise ValueError(
            f"q_lora_rank {logical_k} pads to {padded_k} = {num_tiles} FULL compute tiles, over "
            f"mul_reduce_scalar_tile's limit of {MAX_NORM_TILES}"
        )
    tile, derived = interpret_tile(padded_k)
    # A FULL-aligned width must interpret as FULL tiles; assert rather than assume, because a
    # HALF answer here would silently halve the reduce's coverage.
    if tuple(tile.tile_shape) != (32, 32) or derived != num_tiles:
        raise AssertionError(
            f"interpret_tile({padded_k}) gave {tuple(tile.tile_shape)} x{derived}, expected (32, 32) x{num_tiles}"
        )
    return padded_k, num_tiles


@dataclass(frozen=True)
class GLMQANormQBLayout:
    """Every shape/arg this op derives, computed without touching a device.

    Split out from ``emit`` so the derivation is testable on CPU: the padding decision and the
    ``1/sqrt(logical_k)`` vs ``padded_k`` distinction are the two things most likely to be got
    wrong silently, and neither needs silicon to check.
    """

    logical_k: int
    """q_lora_rank -- the true width of q_a, and the N the RMS mean divides by."""

    padded_k: int
    """logical_k rounded up to a tile-aligned width for SumOfSquares' reduce."""

    norm_num_tiles: int
    """Compute tiles SumOfSquares reduces over (interpreted, not storage, tiles)."""

    norm_tile_shape: tuple[int, int]
    """Interpreted compute-tile geometry for the reduce -- always FULL (32, 32); see
    :func:`_pad_to_full_tile`."""

    rms_scalar: float
    """Reduce scalar. ``1/sqrt(logical_k)``; the kernel squares it to form the ``1/N`` mean."""

    n_total: int
    """q_b output width -- num_attention_heads * qk_head_dim."""

    num_banks: int
    """DRAM banks == matmul cores == cores the norm runs redundantly on."""

    per_core_n: int
    per_core_n_tiles: int
    act_num_pages: int
    """Storage pages of the activation CB the matmul sees: padded_k / 32."""

    subblock_k: int
    num_subblocks_k: int
    subblock_w: int

    @property
    def pad_elements(self) -> int:
        """Zero elements appended to q_a, and zero rows appended to W'."""
        return self.padded_k - self.logical_k


def derive_layout(
    *,
    q_lora_rank: int,
    num_attention_heads: int,
    qk_head_dim: int,
    num_banks: int,
    fp32_dest_acc_en: bool = True,
    subblock_k: int | None = None,
) -> GLMQANormQBLayout:
    """Derive every shape this op needs from the model's hparams. No device required.

    ``num_banks`` is the DRAM bank count, which is also the matmul's core count and therefore
    the number of cores the redundant norm runs on -- ``DRAMStreamingMatmul`` pins one worker
    per bank and there is no way to ask it for a different width.
    """
    if q_lora_rank <= 0:
        raise ValueError(f"q_lora_rank must be positive, got {q_lora_rank}")
    if num_banks <= 0:
        raise ValueError(f"num_banks must be positive, got {num_banks}")

    padded_k, norm_num_tiles = _pad_to_full_tile(q_lora_rank)

    n_total = num_attention_heads * qk_head_dim
    # DRAMStreamingMatmul gives each bank worker a disjoint, equal N slice, and the slice must
    # be a whole number of output tiles. GLM's 20 * 256 = 5120 over 8 banks is 640 = 20 tiles.
    if n_total % (num_banks * TILE_W) != 0:
        raise ValueError(
            f"q_b output width {n_total} (= {num_attention_heads} heads x {qk_head_dim}) must be "
            f"a multiple of num_banks * {TILE_W} = {num_banks * TILE_W} so each bank worker owns "
            "a whole number of output tiles; pad the weight's N to adopt this op"
        )
    per_core_n = n_total // num_banks
    per_core_n_tiles = per_core_n // TILE_W

    # Mirror emit_dram_stream's own subblock_k derivation (common.py:291) so the layout this
    # dataclass reports is the layout the kernel is actually given.
    act_num_pages = padded_k // TILE_W
    sub_k = act_num_pages // 4 if subblock_k is None else subblock_k
    sub_k = max(1, sub_k)
    while act_num_pages % sub_k != 0 and sub_k > 1:
        sub_k -= 1

    return GLMQANormQBLayout(
        logical_k=q_lora_rank,
        padded_k=padded_k,
        norm_num_tiles=norm_num_tiles,
        norm_tile_shape=(32, 32),
        rms_scalar=1.0 / math.sqrt(float(q_lora_rank)),
        n_total=n_total,
        num_banks=num_banks,
        per_core_n=per_core_n,
        per_core_n_tiles=per_core_n_tiles,
        act_num_pages=act_num_pages,
        subblock_k=sub_k,
        num_subblocks_k=act_num_pages // sub_k,
        subblock_w=compute_subblock_w(
            per_core_n_tiles,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dst_full_sync_en=False,
        ),
    )


def fold_gamma_into_qb(q_b_weight_k_by_n, gamma, *, padded_k: int):
    """Offline weight transform: ``W'[k, n] = gamma[k] * W_q_b[k, n]``, zero-padded to padded_k.

    A torch-level helper so the fold and the pad are defined in one place and can be checked
    without a device. ``q_b_weight_k_by_n`` is [q_lora_rank, N] -- note HF stores q_b_proj as
    [N, q_lora_rank], so it must already be transposed. Runs in float and leaves the caller to
    cast, because the cast is where the fold loses precision and that should be visible.
    """
    if q_b_weight_k_by_n.ndim != 2:
        raise ValueError(f"expected a 2-D [K, N] weight, got shape {tuple(q_b_weight_k_by_n.shape)}")
    k = q_b_weight_k_by_n.shape[0]
    gamma_flat = gamma.reshape(-1)
    if gamma_flat.numel() != k:
        raise ValueError(f"gamma has {gamma_flat.numel()} elements but the weight's K is {k}")
    if padded_k < k:
        raise ValueError(f"padded_k {padded_k} is smaller than the weight's K {k}")

    folded = gamma_flat.float().reshape(k, 1) * q_b_weight_k_by_n.float()
    if padded_k == k:
        return folded
    # Rows past the logical K multiply the activation's zero pad, so their value is irrelevant
    # to the result -- zero is simply the honest choice.
    return torch.nn.functional.pad(folded, (0, 0, 0, padded_k - k))


class GLMQANormQBProjection(FusedOp):
    """Deferred q_a RMSNorm folded into the q_b projection: one dispatch, no q_a round-trip.

    ``q_b_weights`` must already carry the folded gamma and the K zero-pad -- see
    :func:`fold_gamma_into_qb` -- on top of the DRAM-width-shard and column-major tile shuffle
    every ``DRAMStreamingMatmul`` weight needs.
    """

    name: str = "glm_qa_norm_qb_projection"
    math_fidelity: str = "LoFi"
    math_approx_mode: bool = False

    q_a: Input = Input()
    q_b_weights: Input = Input()
    q_out: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        ua = user_args or {}
        cls.emit(
            f,
            tensors["q_a"],
            tensors["q_b_weights"],
            q_out=output if output is not None else tensors.get("q_out"),
            logical_k=ua["logical_k"],
            epsilon=ua["epsilon"],
            prefix=ua.get("prefix", "glm_qa_norm_qb"),
            fp32_dest_acc_en=bool(ua.get("fp32_dest_acc_en", True)),
            subblock_k=ua.get("subblock_k"),
            fast_approx_rsqrt=bool(ua.get("fast_approx_rsqrt", False)),
        )

    @staticmethod
    def emit(
        f: FusedProgram,
        q_a,
        q_b_weights,
        *,
        q_out=None,
        logical_k: int,
        epsilon: float,
        prefix: str = "glm_qa_norm_qb",
        fp32_dest_acc_en: bool = True,
        subblock_k: int | None = None,
        fast_approx_rsqrt: bool = False,
    ) -> CBHandle:
        """Emit the norm reduce and the scaled q_b matmul into one FusedProgram.

        ``q_a`` is the **un-normalized** q_a, zero-padded to the width
        :func:`derive_layout` reports and replicated height-sharded over the DRAM-bank workers.
        ``logical_k`` is the true ``q_lora_rank`` (768), which is what the RMS mean divides by --
        passing the padded width here is the one silent-wrongness this signature is shaped to
        prevent. ``epsilon`` is the model's ``rms_norm_eps``.
        """
        if logical_k <= 0:
            raise ValueError(f"logical_k must be positive, got {logical_k}")

        def child(name: str) -> str:
            return BlazeOp.child_prefix(prefix, name)

        # The norm MUST run on exactly the matmul's cores, because it reads the matmul's own
        # replica of the activation and hands the scalar over in that core's L1 with no NOC hop.
        # Deriving both from the same function is what guarantees they cannot drift.
        _, matmul_cores = dram_bank_worker_cores(f.device)

        padded_k, num_tiles = _pad_to_full_tile(logical_k)

        # 1/RMS(q_a) = rsqrt(mean(q_a^2) + eps), computed independently on every bank worker.
        # `scalar` is squared by the LLK to form the 1/N mean, so it carries the LOGICAL width
        # while `width` carries the PADDED one for the tile reinterpretation. The pad is zeros,
        # which contribute nothing to the sum, so the two are consistent.
        #
        # pop_input=True: this CB view of q_a is drained the same way the matmul drains its own
        # view, so a looped program's DM1 re-push stays balanced on both.
        recip = SumOfSquares.emit(
            f,
            q_a,
            prefix=child("qa_rms_recip"),
            cores=matmul_cores,
            scalar=1.0 / math.sqrt(float(logical_k)),
            epsilon=epsilon,
            fast_approx=fast_approx_rsqrt,
            num_tiles=num_tiles,
            pop_input=True,
            width=padded_k,
        )

        # q = (1/RMS) * (q_a @ W'). The scalar rides the matmul's DST epilogue, so the
        # normalization costs no separate pass over the activation and no extra CB traffic.
        # index_offset=0 selects lane 0 of the scalar tile, which is where SumOfSquares packs
        # its result (its kernel documents the rest of the tile as undefined).
        return DRAMStreamingMatmul.emit(
            f,
            q_a,
            q_b_weights,
            index=None,
            bias=None,
            out=q_out,
            scalar=recip,
            pop_scalar=True,
            prefix=child("q_b_proj"),
            fp32_dest_acc_en=fp32_dest_acc_en,
            subblock_k=subblock_k,
            # Mutually exclusive with `scalar` in both the host assert and the kernel's
            # static_assert -- q_b has no activation anyway.
            fused_activation=None,
            index_offset=0,
            wait_for_out=False,
            pop_index=False,
            # Last consumer of this CB view of the activation.
            pop_act=True,
        )

    @staticmethod
    def golden(q_a, q_b_weights_folded, *, logical_k: int, epsilon: float):
        """Reference for what the device computes: ``(1/RMS(q_a)) * (q_a @ W')``.

        Takes the *folded, padded* weight and the padded activation -- i.e. it mirrors the
        device's arithmetic rather than the model's formulation. Use
        :func:`golden_from_unfolded` to check the identity that actually matters.
        """
        x = q_a.float()
        # Reduce over the logical width only; the pad is zeros, so slicing and not slicing agree.
        mean_sq = x[..., :logical_k].pow(2).sum(-1, keepdim=True) / float(logical_k)
        recip = torch.rsqrt(mean_sq + epsilon)
        return recip * (x @ q_b_weights_folded.float())

    @staticmethod
    def golden_from_unfolded(q_a, q_b_weight_k_by_n, gamma, *, epsilon: float):
        """Reference for the model call this replaces: ``RMSNorm(q_a, gamma) @ W_q_b``.

        This is the equality the op rests on. ``q_a`` is un-padded [.., q_lora_rank] and
        ``q_b_weight_k_by_n`` is the un-folded [q_lora_rank, N] weight.
        """
        x = q_a.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
        normed = (x / rms) * gamma.reshape(-1).float()
        return normed @ q_b_weight_k_by_n.float()
