# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GDN rows chain (``ttnn/gdn.py``'s slab / rows path) as torch, op for op, with the chain's rounding points.

Shared by the prefill rows kernels (``gdn_pre_rows`` / ``gdn_post_rows``, the 2048-row slab around the unchanged
chunk prims) and the MTP verify-rows scan (the k + 1 consecutive rows of one tile): the two program families mirror the
same chain ops, so they take one reference and one set of rounding rules.

What this is: a STRUCTURAL reference.  Every function is one op of today's chain and packs where the chain's op packs
(one bf16 rounding per bf16 op, fp32 kept where the chain keeps fp32).  Where the device's arithmetic inside an op is
a hardware rule (an SFPU transcendental, the FPU's 19-bit source path, a 16-bit destination's narrowing), the function
takes a :class:`Rounding` policy whose fields name the candidate rules; the lane's device pin tool (a dev tool)
selects the rule the op follows, and the kernels are then written as the op's own LLK calls.  The functions that are exact by construction (the row shifts, the GQA copy, the layouts, ``x * 1.0``) carry
no policy.  Per-device local shapes throughout; ``T`` = the pass's rows (2048 for the slab, 32 for a verify tile).

The recurrence is here too (the last section): the two chunk prims' per-chunk arithmetic in the order their compute
kernels run it (``chunk_prep_reference`` / ``chunk_scan_reference``: the chain's recurrence, TF32-class operands under
the ``tf32`` source rule), the serial per-row delta rule with every prefix state (``serial_scan_reference``: the verify
rows scan's form and the tt/gdn.py oracle's), the masked commit of a verify pass in both forms, the verify tile's row
mask and its selector-driven history.  Two policies beside the chain's ``DEFAULT``: ``ORACLE`` (tt/gdn.py's rounding
points, the fused gdn_step's ``reference_step``) and ``GDN_STEP`` (the same with the q scale applied to the read-out
in fp32, the one place the landed gdn_step program departs from the oracle's order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import torch
import torch.nn.functional as F

TILE = 32
HEADS = 12  # value heads per device
QK_HEADS = 4  # key heads per device
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM  # 512
VALUE_WIDTH = HEADS * HEAD_DIM  # 1536
QKV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH  # 2560
A_COLUMN = QKV_WIDTH + VALUE_WIDTH  # 4096
B_COLUMN = A_COLUMN + TILE  # 4128
PROJECTION_WIDTH = B_COLUMN + TILE  # 4160
CONV_KERNEL = 4
HISTORY_ROWS = CONV_KERNEL - 1
QK_L2_NORM_EPS = 1.0e-6  # gdn.py QK_L2_NORM_EPS; the op gets eps / HEAD_DIM
RMS_NORM_EPS = 1.0e-6  # gdn.py RMS_NORM_EPS (the gated norm)
QK_SCALE = HEAD_DIM**-0.5
SOFTPLUS_THRESHOLD = 20.0

BF16, FP32 = torch.bfloat16, torch.float32
Narrow = Callable[[torch.Tensor], torch.Tensor]


def bf16_rne(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 round to nearest even (torch's cast; the packer's rounding in the pinned cases so far)."""

    return x.float().to(BF16)


def bf16_trunc(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 by dropping the low 16 bits (a 16-bit destination that does not round)."""

    bits = x.float().contiguous().view(torch.int32) & -65536
    return bits.view(FP32).to(BF16)


def bf16_rna(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 round to nearest, ties AWAY from zero: what a 16-bit destination's hardware narrowing does to the
    SFPU multiply-add of ``ttnn.mac`` (pinned on one die 2026-09-25: 0 of 1.6 M elements off, where RNE misses the
    exact midpoints)."""

    bits = x.float().contiguous().view(torch.int32)
    magnitude = ((bits & 0x7FFFFFFF) + 0x8000) & -65536
    return (magnitude | (bits & -2147483648)).view(FP32).to(BF16)


def plus_zero(x: torch.Tensor) -> torch.Tensor:
    """-0.0 -> +0.0 (the sign of a zero result is not kept by the SFPU ops pinned so far: the bf16 multiply's clamp,
    ``mac``, the fp32 -> bf16 typecast, ``rms_norm`` and the fp32 multiply; the fp32 ``add`` keeps it)."""

    return torch.where(x == 0, torch.zeros((), dtype=x.dtype), x)


def tf32(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> the FPU source registers' 19-bit format (10 mantissa bits kept, truncation), returned as fp32."""

    bits = x.float().contiguous().view(torch.int32) & -8192  # clear the low 13 mantissa bits
    return bits.view(FP32).clone()


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


NARROWINGS: dict[str, Narrow] = {"rne": bf16_rne, "trunc": bf16_trunc, "rna": bf16_rna}
SOURCE_PATHS: dict[str, Narrow] = {"exact": identity, "tf32": tf32}


@dataclass(frozen=True)
class Rounding:
    """The candidate hardware rules the pin tool selects between.  The defaults are the chain's rules as read from
    the Blackhole kernels (2026-09-25; the device pin confirms them): every bf16 binary_ng / ternary / unary op is an
    SFPU op in a 16-bit destination that rounds its fp32 result to bf16 with RNE in software (``mul_binary_tile``,
    ``add_binary_tile<NearestEven>``, ``mac_tile<Float16_b>``, ``silu_tile``); the fp32 ops (the a + dt_bias add, the
    neg_exp_A multiply, sigmoid, softplus) run in a 32-bit destination with their fp32 operands unpacked straight to
    the destination (no 19-bit source path) and store fp32; a python-float scalar of a bf16 multiply is rounded to bf16
    (RNE) on the host before it reaches the device; ``ttnn.typecast(fp32 -> bf16)`` is an explicit SFPU RNE.

    ``pack``: how a bf16 op's result leaves the destination (RNE, or a 16-bit dest's truncation).  ``mac_fused``:
    ``ttnn.mac`` as one fused fp32 multiply-add (the SFPU ``mad``) or as a product rounded before the add.
    ``fp32_source``: an fp32 operand read exactly (unpacked to the destination) or through the FPU's 19-bit source
    registers.  ``scalar_bf16``: the python-float scalar rounded to bf16 first.  ``mul_zero_clamp``: the bf16 SFPU
    multiply's FPU-compatibility rule ``0 * x = +0`` (a negative zero never comes out of a bf16 multiply).

    The oracle-side rules (tt/gdn.py's rounding points, which the fused gdn_step program follows instead of the
    chain's; ``ORACLE`` / ``GDN_STEP`` below): ``conv_packs``: the FIR sum packed once (fp32 sum of the four products,
    one bf16 rounding, then SiLU) or after every op (the chain's five).  ``qk_l2``: the q/k unit vectors by the chain's
    ``rms_norm(eps / 128)`` then ``x 128^-0.5`` (``rms_scaled``) or by the oracle's direct bf16 chain
    ``x * rsqrt(sum(x^2) + eps)`` (``l2_direct``).  ``q_scale_point``: the attention scale ``128^-0.5`` as the
    composite's bf16 multiply on q (``q_bf16``, the chain), as the oracle's fp32 multiply on q before the recurrence
    (``q_fp32``), or as the gdn_step program's fp32 multiply on the read-out ``o`` (``o_fp32``).  ``beta_bf16``: beta
    as the oracle's bf16 sigmoid of the bf16 b (the chain keeps fp32).  ``epilogue``: the gated norm as the chain's
    ops (weighted rms_norm pack, bf16 sigmoid pack, bf16 multiply) or as the oracle's (fp32 unit packed to bf16, bf16
    weight multiply, the fp32 sigmoid multiplied in fp32 and packed once).
    """

    pack: str = "rne"
    mac_fused: bool = True
    fp32_source: str = "exact"
    scalar_bf16: bool = True
    mul_zero_clamp: bool = True
    # ``ttnn.mac``'s narrowing (a 16-bit destination's hardware rounding: ties away from zero), separate from ``pack``
    # (the software RNE of the multiply / typecast / silu): pinned on one die 2026-09-25.
    mac_pack: str = "rna"
    # Whether a zero result loses its sign (+0) in mac, the fp32 -> bf16 typecast, rms_norm and the fp32 multiply
    # (pinned: every one of them returns +0 where IEEE gives -0; the fp32 add keeps the sign).
    zero_sign_plus: bool = True
    conv_packs: int = 5
    qk_l2: str = "rms_scaled"
    q_scale_point: str = "q_bf16"
    beta_bf16: bool = False
    epilogue: str = "chain"

    @property
    def narrow(self) -> Narrow:
        return NARROWINGS[self.pack]

    @property
    def source(self) -> Narrow:
        return SOURCE_PATHS[self.fp32_source]

    @property
    def narrow_mac(self) -> Narrow:
        return NARROWINGS[self.mac_pack]

    def zero_sign(self, x: torch.Tensor) -> torch.Tensor:
        return plus_zero(x) if self.zero_sign_plus else x


DEFAULT = Rounding()
# tt/gdn.py's rounding points (``reference.gated_delta_recurrent`` with its callers' conv / gate / epilogue expressions,
# = ``ttnn/fused/gdn_step``'s ``reference_step``): one conv pack, the direct bf16 l2 chain, q scaled in fp32 before
# the recurrence, bf16 beta, the oracle epilogue.  Every other rule as the chain's (none of them reaches this path).
ORACLE = Rounding(conv_packs=1, qk_l2="l2_direct", q_scale_point="q_fp32", beta_bf16=True, epilogue="oracle")
# The landed fused gdn_step program: the oracle's rounding points except the attention scale, which its compute kernel
# applies to the read-out in fp32 after the ``q @ S`` matmul (``ttnn/fused/gdn_step/kernels/compute.cpp``, the
# ``mul_unary_tile(j, QK_SCALE)`` of the recurrence phase); the verify-rows scan program mirrors this policy.
GDN_STEP = replace(ORACLE, q_scale_point="o_fp32")


# ----------------------------------------------------------------------------------------------------- exact parts


def split_projection(projected: torch.Tensor) -> dict[str, torch.Tensor]:
    """``[T, 4160]`` bf16 -> qkv ``[T, 2560]``, z ``[T, 1536]``, a / b ``[T, 32]`` (columns 0..11 valid): the chain's
    slices (data movement)."""

    if projected.shape[-1] != PROJECTION_WIDTH or projected.dtype != BF16:
        raise ValueError(
            f"projection must be [T, {PROJECTION_WIDTH}] bf16, got {tuple(projected.shape)} {projected.dtype}"
        )
    return {
        "qkv": projected[:, :QKV_WIDTH],
        "z": projected[:, QKV_WIDTH:A_COLUMN],
        "a": projected[:, A_COLUMN:B_COLUMN],
        "b": projected[:, B_COLUMN:PROJECTION_WIDTH],
    }


def fir_taps(qkv: torch.Tensor, history: torch.Tensor) -> list[torch.Tensor]:
    """The four FIR tap inputs ``[T, 2560]`` bf16: tap t (t = 0..2) is the window ``[history rows 0..2 | qkv]`` shifted
    by t rows (``_shifted_rows_slab`` / the 0/1 selects of the 32-row form), tap 3 is qkv itself.  Exact."""

    rows = qkv.shape[0]
    window = torch.cat([history[:HISTORY_ROWS], qkv], dim=0)
    return [window[t : t + rows] for t in range(HISTORY_ROWS)] + [qkv]


def history_next(qkv: torch.Tensor) -> torch.Tensor:
    """The next pass's history tile ``[32, 2560]`` bf16: the last three rows of qkv in rows 0..2, rows 3..31 zero
    (``commit_rows_full``).  Exact."""

    tile = torch.zeros(TILE, qkv.shape[-1], dtype=qkv.dtype)
    tile[:HISTORY_ROWS] = qkv[-HISTORY_ROWS:]
    return tile


def gqa_expand(x: torch.Tensor) -> torch.Tensor:
    """``[T, 512]`` (4 key heads) -> ``[T, 12, 128]``: value head hv reads key head hv // 3 (the chain's 0/1
    ``qk_expand`` matmul under HiFi4 / fp32 accumulation: one 1.0 term per element).  Exact."""

    heads = x.reshape(x.shape[0], QK_HEADS, HEAD_DIM)
    return heads.repeat_interleave(HEADS // QK_HEADS, dim=1)


def to_prim_qk(x: torch.Tensor) -> torch.Tensor:
    """``[T, 12, 128]`` -> the prims' pad-free ``[12, T // 32, 32, 128]`` (head, chunk, row in chunk, dim).  A view."""

    rows = x.shape[0]
    return x.permute(1, 0, 2).reshape(HEADS, rows // TILE, TILE, HEAD_DIM)


def to_prim_vec(x: torch.Tensor) -> torch.Tensor:
    """``[T, 12]`` -> the prims' ``[12, T // 32, 32, 1]`` (beta / g: one column per (head, chunk))."""

    rows = x.shape[0]
    return x.permute(1, 0).reshape(HEADS, rows // TILE, TILE, 1)


def fold_heads(x: torch.Tensor) -> torch.Tensor:
    """Head-major ``[12, T, 128]`` -> token-major ``[T, 1536]`` (the 12 head slices + concat).  Exact."""

    return x.permute(1, 0, 2).reshape(x.shape[1], VALUE_WIDTH)


# ---------------------------------------------------------------------------------------------- the chain's ops


def _mul_clamp(x: torch.Tensor, y: torch.Tensor, product: torch.Tensor, rounding: Rounding) -> torch.Tensor:
    if rounding.mul_zero_clamp:
        product = torch.where((x == 0) | (y == 0), torch.zeros((), dtype=product.dtype), product)
    return product


def multiply_bf16(x: torch.Tensor, y: torch.Tensor, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``ttnn.multiply`` on bf16 operands (broadcasting rows or full tiles): the SFPU product of the exact bf16 values
    in fp32 (``0 * x = +0`` under the clamp), one pack."""

    xf, yf = x.float(), y.float()
    return rounding.narrow(_mul_clamp(xf, yf, xf * yf, rounding))


def multiply_scalar_bf16(x: torch.Tensor, scalar: float, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``ttnn.multiply(x_bf16, python_float)``: the scalar as fp32 (or as bf16 first under ``scalar_bf16``), one pack."""

    s = torch.tensor(scalar, dtype=FP32)
    if rounding.scalar_bf16:
        s = s.to(BF16).float()
    xf = x.float()
    return rounding.narrow(_mul_clamp(xf, s, xf * s, rounding))


def mac_bf16(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``ttnn.mac(a, b, c)`` on bf16: the SFPU ternary ``a * b + c`` (fused fp32 multiply-add by default; or the
    product narrowed first), narrowed by the 16-bit destination (``mac_pack``: ties away from zero), +0 for a zero."""

    if rounding.mac_fused:
        return rounding.zero_sign(rounding.narrow_mac(torch.addcmul(c.float(), a.float(), b.float())))
    return rounding.zero_sign(rounding.narrow_mac(rounding.narrow(a.float() * b.float()).float() + c.float()))


def silu_bf16(x: torch.Tensor, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``ttnn.silu`` on bf16: torch's silu of the exact values, one pack.  NOT the SFPU's polynomial: the kernel calls
    the op's own LLK; this reference bounds the op to a few bf16 ulps (the pin tool records the deviation)."""

    return rounding.narrow(F.silu(x.float()))


def rms_norm_bf16(
    x: torch.Tensor, eps: float, weight: torch.Tensor | None = None, rounding: Rounding = DEFAULT
) -> torch.Tensor:
    """``ttnn.rms_norm(x_bf16, epsilon=eps[, weight=w_bf16])`` over the last dimension: x * rsqrt(mean(x^2) + eps)
    [* w], one pack.  STRUCTURAL ONLY: the op's default compute config on this runtime is ``math_approx_mode=True``,
    ``fp32_dest_acc_en=False`` (rmsnorm.cpp), so the device squares, sums (the FPU reduce against a bf16 1.0 scaler
    tile), scales by 1/W (an SFPU multiply whose bf16 store truncates), adds a TRUNCATED bf16 eps, takes the 10-bit
    approximate rsqrt and multiplies (and multiplies by the weight) all in a 16-bit destination with bf16
    intermediates: no torch expression reproduces it; the kernel mirror is the op's LLK sequence itself
    (``gdn_pre_rows`` / ``gdn_post_rows`` compute kernels) and the reference bounds it to a few bf16 ulps."""

    xf = x.float()
    unit = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        unit = unit * weight.float()
    return rounding.zero_sign(rounding.narrow(unit))


def typecast_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """``ttnn.typecast(bf16 -> fp32)``: exact."""

    return x.float()


def typecast_to_bf16(x: torch.Tensor, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``ttnn.typecast(fp32 -> bf16)``: one narrowing (RNE); a zero loses its sign."""

    return rounding.zero_sign(rounding.narrow(x))


def sigmoid_fp32(x: torch.Tensor) -> torch.Tensor:
    """``ttnn.sigmoid`` on fp32: torch's sigmoid (the SFPU's own polynomial on the device; structural)."""

    return torch.sigmoid(x.float())


def add_fp32_row(x: torch.Tensor, row: torch.Tensor, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``ttnn.add(x_fp32, row_fp32)`` (row broadcast): on fp32 operands binary_ng takes the SFPU add in a 32-bit
    destination whatever ``fast_and_approximate_mode`` says, the row filled to a tile by the reader and both operands
    unpacked straight to the destination: an exact fp32 sum (``fp32_source`` = ``tf32`` is the FPU alternative the
    pin tool rules out)."""

    return rounding.source(x.float()) + rounding.source(row.float())


def softplus_fp32(x: torch.Tensor) -> torch.Tensor:
    """``ttnn.softplus(x, beta=1, threshold=20)`` on fp32: torch's (structural; the SFPU's exp / log on the device)."""

    return F.softplus(x.float(), beta=1.0, threshold=SOFTPLUS_THRESHOLD)


def multiply_fp32_row(row: torch.Tensor, x: torch.Tensor, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``ttnn.multiply(row_fp32, x_fp32)`` (row broadcast): the SFPU product in a 32-bit destination of two exactly
    unpacked fp32 operands (no bf16 narrowing; a zero result comes out +0); ``fp32_source`` names the FPU alternative
    the pin tool ruled out (2026-09-25: exact on 24,576 elements, tf32 off on two thirds)."""

    return rounding.zero_sign(rounding.source(row.float()) * x.float())


# ------------------------------------------------------------------------------------------------- the two programs


def conv_silu(taps: list[torch.Tensor], weights: torch.Tensor, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``_causal_conv_rows``' arithmetic: multiply(tap0, w0), mac x 3, silu; five packs.  ``weights`` ``[4, 2560]``.

    Under ``conv_packs = 1`` (the oracle, ``reference_step``): the four products summed in fp32 in tap order, one bf16
    pack, then torch's bf16 SiLU (the oracle's own expressions, so the result is bitwise the oracle's)."""

    if rounding.conv_packs == 1:
        conv = sum(tap.float() * weight.float() for tap, weight in zip(taps, weights)).to(BF16)
        return F.silu(conv)
    if rounding.conv_packs != CONV_KERNEL + 1:
        raise ValueError(f"conv_packs is 1 (the oracle) or {CONV_KERNEL + 1} (the chain), got {rounding.conv_packs}")
    conv = multiply_bf16(taps[0], weights[0], rounding)
    for t in range(1, CONV_KERNEL):
        conv = mac_bf16(taps[t], weights[t], conv, rounding)
    return silu_bf16(conv, rounding)


def l2_norm_direct(x: torch.Tensor) -> torch.Tensor:
    """The oracle's l2 normalization on a bf16 tensor over the last dimension, ``x * rsqrt(sum(x * x) + eps)`` with
    every op in bf16 (``reference._l2_norm``, called by ``gated_delta_recurrent`` with ``l2_normalize_qk=True``; the
    fused gdn_step's bf16 l2 chain follows it).  Structural for the device; bitwise the oracle in torch."""

    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + QK_L2_NORM_EPS)


def qk_prepare(x: torch.Tensor, *, composite_scale: bool, rounding: Rounding = DEFAULT) -> torch.Tensor:
    """``_make_chunk_inputs``' q / k: expand, ``rms_norm(eps / 128)``, ``x 128^-0.5`` (into the rows buffer) and, for q,
    the composite's own ``q * scale`` (``chunk_gated_delta_rule.cpp``).  ``[T, 512]`` -> ``[T, 12, 128]`` bf16.

    Under ``qk_l2 = "l2_direct"`` (the oracle) the unit vectors come from :func:`l2_norm_direct` and carry no scale
    (the attention scale is the scan's, ``q_scale_point``); ``composite_scale`` must then be False."""

    heads = gqa_expand(x)
    if rounding.qk_l2 == "l2_direct":
        if composite_scale:
            raise ValueError("the oracle's l2 chain carries no bf16 scale: the scan applies it (q_scale_point)")
        return l2_norm_direct(heads)
    if rounding.qk_l2 != "rms_scaled":
        raise ValueError(f"qk_l2 is 'rms_scaled' (the chain) or 'l2_direct' (the oracle), got {rounding.qk_l2!r}")
    normed = rms_norm_bf16(heads, QK_L2_NORM_EPS / HEAD_DIM, rounding=rounding)
    scaled = multiply_scalar_bf16(normed, QK_SCALE, rounding)
    return multiply_scalar_bf16(scaled, QK_SCALE, rounding) if composite_scale else scaled


def gates(
    a: torch.Tensor, b: torch.Tensor, dt_bias: torch.Tensor, neg_exp_A: torch.Tensor, rounding: Rounding = DEFAULT
):
    """beta and the log decay g, fp32 ``[T, 12]``: sigmoid(fp32 b); neg_exp_A * softplus(fp32 a + dt_bias).

    Under ``beta_bf16`` (the oracle) beta is torch's bf16 sigmoid of the bf16 b (``reference_step``'s
    ``b.sigmoid()``), returned as those bf16 values in an fp32 tensor."""

    if rounding.beta_bf16:
        beta = torch.sigmoid(b[:, :HEADS]).float()
    else:
        beta = sigmoid_fp32(typecast_to_fp32(b[:, :HEADS]))
    shifted = add_fp32_row(typecast_to_fp32(a[:, :HEADS]), dt_bias, rounding)
    g = multiply_fp32_row(neg_exp_A, softplus_fp32(shifted), rounding)
    return beta, g


def pre_reference(
    projected: torch.Tensor,
    history: torch.Tensor,
    conv_weights: torch.Tensor,
    dt_bias: torch.Tensor,
    neg_exp_A: torch.Tensor,
    rounding: Rounding = DEFAULT,
) -> dict[str, torch.Tensor]:
    """``gdn_pre_rows``: from the projection ``[T, 4160]`` bf16 and the history tile to the prims' inputs.

    Returns ``conv`` ``[T, 2560]`` bf16, ``q`` / ``k`` ``[T, 12, 128]`` bf16 (q with the composite's scale folded
    under the chain's ``q_scale_point = "q_bf16"``; unscaled unit vectors under the oracle policies, whose scale the
    scan applies), ``v`` ``[T, 1536]`` bf16, ``beta`` / ``g`` ``[T, 12]`` fp32, and their prim layouts ``q_c`` / ``k_c``
    ``[12, T/32, 32, 128]``, ``beta_c`` / ``g_c`` ``[12, T/32, 32, 1]``, plus ``z`` ``[T, 1536]`` bf16 (untouched: the
    post program's input) and ``history_next`` ``[32, 2560]`` bf16 (the slab's; a verify pass uses
    :func:`history_after_commit`)."""

    parts = split_projection(projected)
    conv = conv_silu(fir_taps(parts["qkv"], history), conv_weights, rounding)
    q = qk_prepare(conv[:, :QK_WIDTH], composite_scale=rounding.q_scale_point == "q_bf16", rounding=rounding)
    k = qk_prepare(conv[:, QK_WIDTH : 2 * QK_WIDTH], composite_scale=False, rounding=rounding)
    v = conv[:, 2 * QK_WIDTH :]  # x * 1.0 (the all-ones row mask): exact
    beta, g = gates(parts["a"], parts["b"], dt_bias, neg_exp_A, rounding)
    return {
        "conv": conv,
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "g": g,
        "q_c": to_prim_qk(q),
        "k_c": to_prim_qk(k),
        "beta_c": to_prim_vec(beta),
        "g_c": to_prim_vec(g),
        "z": parts["z"],
        "history_next": history_next(parts["qkv"]),
    }


def post_reference(
    o: torch.Tensor, z: torch.Tensor, norm_weight: torch.Tensor, rounding: Rounding = DEFAULT
) -> torch.Tensor:
    """``gdn_post_rows``: the scan's ``o`` ``[12, T, 128]`` fp32 and z ``[T, 1536]`` bf16 to the gated ``[T, 1536]``
    bf16 (``_gate_and_project_rows`` before the out-projection): typecast, gated rms_norm(weight, 1e-6), the fp32
    sigmoid of z packed to bf16, one bf16 multiply, the head fold."""

    heads = typecast_to_bf16(o, rounding)  # [12, T, 128]
    if rounding.epilogue == "oracle":
        # reference_step: fp32 unit of the bf16 o, packed to bf16; the bf16 weight multiply; the fp32 sigmoid of z
        # multiplied in fp32 and packed once.  torch's own expressions, so bitwise the oracle.
        of = heads.float()
        unit = of * torch.rsqrt(of.square().mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
        normalized = norm_weight * unit.to(BF16)
        return (fold_heads(normalized) * torch.sigmoid(z.float())).to(BF16)
    if rounding.epilogue != "chain":
        raise ValueError(f"epilogue is 'chain' or 'oracle', got {rounding.epilogue!r}")
    normalized = rms_norm_bf16(heads, RMS_NORM_EPS, weight=norm_weight, rounding=rounding)
    sig = typecast_to_bf16(sigmoid_fp32(typecast_to_fp32(z)), rounding)  # [T, 1536]
    return multiply_bf16(fold_heads(normalized), sig, rounding)


# ------------------------------------------------------------------------ the recurrence: the chunk prims, the serial scan

# The prims' constant tiles (``gdn.py`` ``chunk_constant_tiles``; the composite's ``make_quadrant_masks``): eye / tril /
# ones ``[32, 32]`` and the three 32x32 quadrant masks of the WY inverse (top-left, bottom-right, bottom-left).
_EYE = torch.eye(TILE)
_TRIL = torch.tril(torch.ones(TILE, TILE))
_ONES = torch.ones(TILE, TILE)
_LOW = torch.arange(TILE).unsqueeze(1) < TILE // 2
_LOW_COL = torch.arange(TILE).unsqueeze(0) < TILE // 2
_Q_TL = (_LOW & _LOW_COL).float()
_Q_BR = (~_LOW & ~_LOW_COL).float()
_Q_10 = (~_LOW & _LOW_COL).float()
HORNER_TERMS = 16  # a strictly-lower 16x16 block is nilpotent at 16: invert16 sums N^0 .. N^15


def _mm_kt(a: torch.Tensor, b: torch.Tensor, rounding: Rounding) -> torch.Tensor:
    """``a [M, K] @ b [K, N]`` as the prims' ``mm`` helper runs it: one output tile at a time, the K dimension walked in
    32-wide k-tiles in ASCENDING order accumulating in the fp32 destination (``chunk_gdn_prep.cpp`` /
    ``chunk_gdn_scan.cpp``, ``mm(...)``: ``matmul_tiles`` over ``ki`` then one pack), both operands through the source
    registers (``rounding.source``).  Inside one k-tile the FPU's 32-term order is the hardware's; torch's is used here
    (structural)."""

    if a.shape[-1] % TILE or a.shape[-1] != b.shape[-2]:
        raise ValueError(f"contraction {a.shape[-1]} must be whole tiles and match {b.shape[-2]}")
    acc = None
    for kt in range(a.shape[-1] // TILE):
        block = rounding.source(a[..., kt * TILE : (kt + 1) * TILE].float()) @ rounding.source(
            b[..., kt * TILE : (kt + 1) * TILE, :].float()
        )
        acc = block if acc is None else acc + block
    return acc


def _fpu(op: str, x: torch.Tensor, y: torch.Tensor, rounding: Rounding) -> torch.Tensor:
    """An FPU eltwise or broadcast op in the fp32 destination (``ew`` / ``bcast_*`` of the prims' kernels): both
    operands through the source registers, IEEE fp32 result."""

    xs, ys = rounding.source(x.float()), rounding.source(y.float())
    if op == "add":
        return xs + ys
    if op == "sub":
        return xs - ys
    if op == "mul":
        return xs * ys
    raise ValueError(op)


def _invert16(nq: torch.Tensor, rounding: Rounding) -> torch.Tensor:
    """``invert16`` (``chunk_gdn_prep.cpp``, the 16-quadrant Horner): ``out = I + Nq``, then fourteen times
    ``out = I + Nq @ out`` (= sum of ``Nq^k`` for k < 16, nilpotent at 16).  ``nq`` is the strictly-lower 16-block
    isolated in one quadrant of a 32x32 tile (the rest zero); ``I`` is the full 32x32 identity, as the kernel adds it.
    """

    out = _fpu("add", _EYE, nq, rounding)
    for _ in range(2, HORNER_TERMS):
        out = _fpu("add", _EYE, _mm_kt(nq, out, rounding), rounding)
    return out


def _invert_block(neg_n: torch.Tensor, rounding: Rounding) -> torch.Tensor:
    """``invert_block`` (``chunk_gdn_prep.cpp``): ``(I - negN)^-1`` for the strictly-lower 32x32 ``negN`` (=
    ``-strictly_lower(k_beta k^T L)``) split into 16-quadrants: ``Bi00 = (I - N00)^-1``, ``Bi11 = (I - N11)^-1`` by
    :func:`_invert16`, ``off = Bi11 @ N10 @ Bi00`` (two tile matmuls), assembled as ``Qtl*Bi00 + Qbr*Bi11 + off`` in
    that order of FPU ops.  NOT ``torch.linalg.inv``: the power series' rounding is the prim's."""

    n00 = _fpu("mul", neg_n, _Q_TL, rounding)
    bi00 = _invert16(n00, rounding)
    n11 = _fpu("mul", neg_n, _Q_BR, rounding)
    bi11 = _invert16(n11, rounding)
    n10 = _fpu("mul", neg_n, _Q_10, rounding)
    off = _mm_kt(_mm_kt(bi11, n10, rounding), bi00, rounding)
    tl = _fpu("mul", bi00, _Q_TL, rounding)
    br = _fpu("mul", bi11, _Q_BR, rounding)
    return _fpu("add", _fpu("add", tl, br, rounding), off, rounding)


def chunk_prep_reference(
    q_c: torch.Tensor,
    k_c: torch.Tensor,
    v: torch.Tensor,
    g_c: torch.Tensor,
    beta_c: torch.Tensor,
    rounding: Rounding = DEFAULT,
) -> dict[str, torch.Tensor]:
    """``prim::chunk_gdn_prep`` per (head, chunk) item, in the order of its compute kernel
    (``ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_prep.cpp``, the
    ``kernel_main`` phases P1 .. dl and the helpers ``mm`` / ``ew`` / ``bcast_cols_mul`` / ``bcast_rows_sub`` / ``expc`` /
    ``invert_block``; compute config HiFi4, fp32 destination, approximations off, the phased program factory's
    ``compute_cfg``).  Inputs are :func:`pre_reference`'s prim layouts: ``q_c`` / ``k_c`` ``[12, NC, 32, 128]`` bf16 (q
    with the composite's scale, the chain's ``q_bf16``), ``v`` flat ``[T, 1536]`` bf16 (the composite's flat-v form the
    prep reader addresses: head h at columns 128h..), ``g_c`` / ``beta_c`` ``[12, NC, 32, 1]`` fp32.  Every FPU op reads
    its operands through the source registers (``rounding.source``: exact or TF32); the SFPU ``exp`` is torch's
    (structural).  Returns the prim's seven fp32 hand-off tensors: ``v_beta`` / ``kd`` / ``q_decay`` ``[12, NC, 32, 128]``,
    ``intra`` / ``t_inv`` ``[12, NC, 32, 32]``, ``k_dec_t`` ``[12, NC, 128, 32]``, ``dl`` ``[12, NC, 1, 1]`` (= exp(sum g)
    as the kernel forms it: ``decayfac[0] * decay_exp[0]``)."""

    heads, chunks = q_c.shape[0], q_c.shape[1]
    if (heads, q_c.shape[2], q_c.shape[3]) != (HEADS, TILE, HEAD_DIM) or k_c.shape != q_c.shape:
        raise ValueError(
            f"q_c / k_c must be [{HEADS}, NC, {TILE}, {HEAD_DIM}], got {tuple(q_c.shape)} {tuple(k_c.shape)}"
        )
    if v.shape != (chunks * TILE, VALUE_WIDTH):
        raise ValueError(f"v must be the flat [T, {VALUE_WIDTH}] of {chunks} chunks, got {tuple(v.shape)}")
    if g_c.shape != (HEADS, chunks, TILE, 1) or beta_c.shape != g_c.shape:
        raise ValueError(f"g_c / beta_c must be [{HEADS}, NC, {TILE}, 1], got {tuple(g_c.shape)} {tuple(beta_c.shape)}")
    out = {
        "v_beta": torch.empty(HEADS, chunks, TILE, HEAD_DIM),
        "kd": torch.empty(HEADS, chunks, TILE, HEAD_DIM),
        "q_decay": torch.empty(HEADS, chunks, TILE, HEAD_DIM),
        "intra": torch.empty(HEADS, chunks, TILE, TILE),
        "k_dec_t": torch.empty(HEADS, chunks, HEAD_DIM, TILE),
        "dl": torch.empty(HEADS, chunks, 1, 1),
        "t_inv": torch.empty(HEADS, chunks, TILE, TILE),
    }
    for h in range(HEADS):
        for c in range(chunks):
            q, k = q_c[h, c], k_c[h, c]
            vv = v[c * TILE : (c + 1) * TILE, h * HEAD_DIM : (h + 1) * HEAD_DIM]
            g, beta = g_c[h, c], beta_c[h, c]  # [32, 1] columns
            # P1: v_beta = v * beta, k_beta = k * beta (column broadcasts)
            v_beta = _fpu("mul", vv, beta, rounding)
            k_beta = _fpu("mul", k, beta, rounding)
            # P2: decay = tril @ g (the in-chunk cumulative sum as a 0/1 tile matmul), decay_exp, decay_row
            decay = _mm_kt(_TRIL, g, rounding)
            decay_exp = torch.exp(decay)
            decay_row = decay.transpose(0, 1)
            # L_mask = tril(exp(decay_i - decay_j)): ones * decay_i, - decay_j, * tril, exp, * tril
            l_mask = _fpu("mul", _ONES, decay, rounding)
            l_mask = _fpu("sub", l_mask, decay_row, rounding)
            l_mask = _fpu("mul", l_mask, _TRIL, rounding)
            l_mask = torch.exp(l_mask)
            l_mask = _fpu("mul", l_mask, _TRIL, rounding)
            # decayfac = exp(g_sum - decay), g_sum = ones @ g
            g_sum = _mm_kt(_ONES, g, rounding)
            decayfac = torch.exp(_fpu("sub", g_sum, decay, rounding))
            # N = strictly_lower(k_beta @ k^T * L_mask) as negN = diag(kk_m) - kk_m; T_inv = (I + N)^-1 = (I - negN)^-1
            kk = _mm_kt(k_beta, k.transpose(0, 1), rounding)
            kk_m = _fpu("mul", kk, l_mask, rounding)
            diag = _fpu("mul", kk_m, _EYE, rounding)
            neg_n = _fpu("sub", diag, kk_m, rounding)
            t_inv = _invert_block(neg_n, rounding)
            # kd = k_beta * decay_exp (the un-premultiplied hand-off: the scan applies T_inv after the subtraction)
            kd = _fpu("mul", k_beta, decay_exp, rounding)
            # intra = (q @ k^T) * L_mask; q_decay = q * decay_exp; k_dec_t = transpose(k * decayfac); dl
            intra = _fpu("mul", _mm_kt(q, k.transpose(0, 1), rounding), l_mask, rounding)
            q_decay = _fpu("mul", q, decay_exp, rounding)
            k_dec_t = _fpu("mul", k, decayfac, rounding).transpose(0, 1)
            dl = _fpu("mul", decayfac, decay_exp, rounding)[0, 0]
            out["v_beta"][h, c], out["kd"][h, c], out["q_decay"][h, c] = v_beta, kd, q_decay
            out["intra"][h, c], out["k_dec_t"][h, c], out["t_inv"][h, c] = intra, k_dec_t, t_inv
            out["dl"][h, c, 0, 0] = dl
    return out


def chunk_scan_reference(
    prep: dict[str, torch.Tensor], s0: torch.Tensor, rounding: Rounding = DEFAULT
) -> tuple[torch.Tensor, torch.Tensor]:
    """``prim::chunk_gdn_scan`` per head, chunk after chunk, in the order of its compute kernel
    (``.../device/kernels/compute/chunk_gdn_scan.cpp``, ``kernel_main``): ``kdS = kd @ S``; ``diff = v_beta - kdS``;
    ``v_new = T_inv @ diff``; ``o_inter = q_decay @ S``; ``intra_v = intra @ v_new``; ``o = o_inter + intra_v``;
    ``s_upd = k_dec_t @ v_new``; ``stmp = S * dl``; ``S = stmp + s_upd``.  The device runs one core per (head, 32-column
    V-block) and every op is per output tile, so the full-width torch form here has the same per-tile order.
    ``s0`` ``[12, 128, 128]`` fp32 (the committed state).  Returns ``o`` ``[12, T, 128]`` fp32 head-major (the prim's
    ``output_head_major`` form, T = NC x 32) and the final state ``[12, 128, 128]`` fp32."""

    heads, chunks = prep["kd"].shape[:2]
    if s0.shape != (HEADS, HEAD_DIM, HEAD_DIM) or heads != HEADS:
        raise ValueError(f"state must be [{HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {tuple(s0.shape)}")
    o = torch.empty(HEADS, chunks * TILE, HEAD_DIM)
    final = torch.empty_like(s0, dtype=FP32)
    for h in range(HEADS):
        state = s0[h].float()
        for c in range(chunks):
            kd_s = _mm_kt(prep["kd"][h, c], state, rounding)
            diff = _fpu("sub", prep["v_beta"][h, c], kd_s, rounding)
            v_new = _mm_kt(prep["t_inv"][h, c], diff, rounding)
            o_inter = _mm_kt(prep["q_decay"][h, c], state, rounding)
            intra_v = _mm_kt(prep["intra"][h, c], v_new, rounding)
            o[h, c * TILE : (c + 1) * TILE] = _fpu("add", o_inter, intra_v, rounding)
            s_upd = _mm_kt(prep["k_dec_t"][h, c], v_new, rounding)
            stmp = _fpu("mul", state, prep["dl"][h, c], rounding)
            state = _fpu("add", stmp, s_upd, rounding)
        final[h] = state
    return o, final


def serial_scan_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    s0: torch.Tensor,
    *,
    rounding: Rounding = DEFAULT,
    scale_point: str | None = None,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-row gated delta rule with every prefix state: the verify-rows scan program's form (one recurrence step per
    real row of the tile, the state carried row to row) and the oracle's (``reference.gated_delta_recurrent``, whose
    loop this is, op for op and shape for shape, so ``fp32_source = "exact"`` is bitwise the oracle).

    ``q`` / ``k`` ``[T, 12, 128]`` bf16 unit vectors (:func:`qk_prepare`), ``v`` ``[T, 12, 128]`` bf16, ``beta`` / ``g``
    ``[T, 12]`` fp32, ``s0`` ``[12, 128, 128]`` fp32.  Per row j: ``S' = S * exp(g_j)``; ``v_read = k_j @ S'``;
    ``delta = (v_j - v_read) * beta_j``; ``S = S' + k_j^T delta``; ``o_j = q_j @ S``.  The matmul operands (k and S, k
    and delta, q and S) pass through ``rounding.source``; the decay multiply and the delta are exact fp32 (the SFPU
    path of the gdn_step program).  ``scale_point`` (default ``rounding.q_scale_point``): ``q_bf16`` = q already
    carries the scale (the chain's rows buffers), ``q_fp32`` = ``q * (1 / sqrt(128))`` in fp32 before the loop (the
    oracle), ``o_fp32`` = ``o_j * 128^-0.5`` in fp32 after the read-out (the gdn_step program).  ``row_mask`` ``[T]``
    (0/1 fp32; the commit's ``arange <= a``): ``beta_j <- beta_j * m_j`` and ``exp(g_j) <- exp(g_j) * m_j + (1 - m_j)``,
    so a masked row is the exact identity ``S * 1.0 + 0`` (the verify-rows scan's commit mode).  Returns ``o`` ``[T, 12,
    128]`` fp32 and ``states`` ``[T + 1, 12, 128, 128]`` fp32 (``states[0] = s0``, ``states[j + 1]`` after row j)."""

    rows = q.shape[0]
    if q.shape != (rows, HEADS, HEAD_DIM) or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"q / k / v must be [T, {HEADS}, {HEAD_DIM}], got {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}"
        )
    if beta.shape != (rows, HEADS) or g.shape != beta.shape or s0.shape != (HEADS, HEAD_DIM, HEAD_DIM):
        raise ValueError("beta / g must be [T, 12] and the state [12, 128, 128]")
    scale_point = rounding.q_scale_point if scale_point is None else scale_point
    if scale_point not in ("q_bf16", "q_fp32", "o_fp32"):
        raise ValueError(f"scale_point is 'q_bf16', 'q_fp32' or 'o_fp32', got {scale_point!r}")
    src = rounding.source
    # the oracle's layout: [batch 1, heads, rows, dim] fp32, its exact expressions below
    query, key, value = (t.unsqueeze(0).transpose(1, 2).contiguous().float() for t in (q, k, v))
    beta_h, decay_h = (t.unsqueeze(0).transpose(1, 2).contiguous().float() for t in (beta, g))
    if scale_point == "q_fp32":
        query = query * (1.0 / math.sqrt(HEAD_DIM))
    state = s0.unsqueeze(0).float().clone()
    states = [state[0].clone()]
    output = torch.empty(1, HEADS, rows, HEAD_DIM, dtype=FP32)
    for index in range(rows):
        q_t, k_t, v_t = query[:, :, index], key[:, :, index], value[:, :, index]
        decay = decay_h[:, :, index].exp()
        beta_t = beta_h[:, :, index]
        if row_mask is not None:
            m = row_mask[index].float()
            decay = decay * m + (1.0 - m)
            beta_t = beta_t * m
        state = state * decay.unsqueeze(-1).unsqueeze(-1)
        remembered = (src(state) * src(k_t).unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - remembered) * beta_t.unsqueeze(-1)
        state = state + src(k_t).unsqueeze(-1) * src(delta).unsqueeze(-2)
        read = (src(state) * src(q_t).unsqueeze(-1)).sum(dim=-2)
        output[:, :, index] = read * torch.tensor(QK_SCALE, dtype=FP32) if scale_point == "o_fp32" else read
        states.append(state[0].clone())
    return output[0].transpose(0, 1).contiguous(), torch.stack(states)


def committed_row_mask(rows: int, accepted: int) -> torch.Tensor:
    """The commit's ``arange <= accepted`` over the tile's rows (``build_rows_selectors``: row 0, the base token, is
    always committed; ``accepted = -1`` is the all-masked passthrough kept for the reference's own study)."""

    if not -1 <= accepted < rows:
        raise ValueError(f"accepted must be in [-1, {rows}), got {accepted}")
    return (torch.arange(rows) <= accepted).float()


def masked_commit_reference(
    pre: dict[str, torch.Tensor], s0: torch.Tensor, accepted: int, form: str, rounding: Rounding = DEFAULT
) -> torch.Tensor:
    """The state a pass commits after ``accepted + 1`` rows, from ``pre`` (:func:`pre_reference`'s dict, masked or not)
    and the committed state ``s0``.  ``form = "chunk"``: today's ``commit_rows`` (``gdn.py``): beta and g of the rows
    past the prefix zeroed by the exact fp32 mask multiply, then the prims (:func:`chunk_prep_reference` +
    :func:`chunk_scan_reference`) over the whole tile.  ``form = "serial"``: the verify-rows scan's commit mode:
    :func:`serial_scan_reference` over the whole tile with ``row_mask`` (masked rows exact identities), whose final
    state is the prefix state ``states[accepted + 1]``.  Returns ``[12, 128, 128]`` fp32."""

    rows = pre["beta"].shape[0]
    mask = committed_row_mask(rows, accepted)
    if form == "chunk":
        beta_m, g_m = pre["beta"] * mask[:, None], pre["g"] * mask[:, None]
        prep = chunk_prep_reference(pre["q_c"], pre["k_c"], pre["v"], to_prim_vec(g_m), to_prim_vec(beta_m), rounding)
        return chunk_scan_reference(prep, s0, rounding)[1]
    if form == "serial":
        v = pre["v"].reshape(rows, HEADS, HEAD_DIM)
        _, states = serial_scan_reference(
            pre["q"], pre["k"], v, pre["beta"], pre["g"], s0, rounding=rounding, row_mask=mask
        )
        return states[rows]
    raise ValueError(f"form is 'chunk' or 'serial', got {form!r}")


def mask_rows(pre: dict[str, torch.Tensor], rows: int, rounding: Rounding = DEFAULT) -> dict[str, torch.Tensor]:
    """The verify tile's padding: rows ``>= rows`` of q / k / v (bf16 ``x * 0.0``: the chain's ``row_mask_bf16`` /
    ``row_mask_bf16_col`` multiplies, +0 under the clamp) and of beta / g (fp32 ``x * 0.0``, the chain's
    ``row_mask_fp32`` multiply: IEEE, so a negative g leaves -0.0) are zero, rows below are untouched; the prim layouts
    are rebuilt from the masked tensors.  The other keys pass through."""

    total = pre["beta"].shape[0]
    if not 1 <= rows <= total:
        raise ValueError(f"rows must be in [1, {total}], got {rows}")
    keep = (torch.arange(total) < rows).float()
    keep_bf16 = keep.to(BF16)
    out = dict(pre)
    out["q"] = multiply_bf16(pre["q"], keep_bf16[:, None, None], rounding)
    out["k"] = multiply_bf16(pre["k"], keep_bf16[:, None, None], rounding)
    out["v"] = multiply_bf16(pre["v"], keep_bf16[:, None], rounding)
    out["beta"] = pre["beta"] * keep[:, None]
    out["g"] = pre["g"] * keep[:, None]
    out["q_c"], out["k_c"] = to_prim_qk(out["q"]), to_prim_qk(out["k"])
    out["beta_c"], out["g_c"] = to_prim_vec(out["beta"]), to_prim_vec(out["g"])
    return out


def history_after_commit(qkv: torch.Tensor, history: torch.Tensor, accepted: int) -> torch.Tensor:
    """The next pass's history tile ``[32, 2560]`` bf16 after a verify commit of ``accepted + 1`` rows: rows 0..2 are
    logical window rows ``accepted + 1 .. accepted + 3`` of ``[history rows 0..2 | qkv]`` (the ``history_select`` of
    ``build_rows_selectors`` applied by ``_advance_history_rows``, ``gdn.py``), rows 3..31 zero.  Exact."""

    if not 0 <= accepted < qkv.shape[0]:
        raise ValueError(f"accepted must be in [0, {qkv.shape[0]}), got {accepted}")
    window = torch.cat([history[:HISTORY_ROWS], qkv], dim=0)
    tile = torch.zeros(TILE, qkv.shape[-1], dtype=qkv.dtype)
    tile[:HISTORY_ROWS] = window[accepted + 1 : accepted + 1 + HISTORY_ROWS]
    return tile
