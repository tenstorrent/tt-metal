# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar SFPU port registry for the Blackhole SFPI parity set.

The SFPU parity dashboard (``/api/sfpu-parity``) lists every distinct SFPU compute kernel
per architecture. Filtering it to "implemented on Blackhole in pure ``sfpi::``" *and*
"absent on Quasar" yields 57 kernels. This module is the single source of truth for that
set: what each kernel provides, which header gates it, and whether it has landed on Quasar
yet.

Why a gate at all
-----------------
The tests for these 57 are written *before* the kernels are ported. A test cannot dispatch
to a kernel that does not exist -- the C++ ``#include`` would not resolve -- so every test
that drives a parity op filters its sweep through :func:`is_ported`. Today that yields zero
collected variants and a green suite; the moment a kernel header lands in the Quasar tree,
its full sweep activates with no test edit.

The gate resolves against the filesystem rather than a hand-maintained flag, so nothing has
to be flipped when a kernel lands. The C++ dispatcher gates the matching branch on
``__has_include`` of the *same* header basename, which makes it structurally impossible for
the Python and C++ sides to disagree about what is available.

Header basenames, not paths
---------------------------
Ported Quasar SFPU kernels live in more than one tree -- ``hw/ckernels/quasar/metal/
llk_api/llk_sfpu/`` (most), ``tt_llk_quasar/common/inc/sfpu/``, and
``tt_llk_quasar/common/inc/experimental/`` -- and all three are on the compiler's include
path as bare-name roots (see ``test_config.py``). Since there is no way to know which tree a
future port will choose, entries record the *basename* and :func:`is_ported` searches every
root. ``__has_include("ckernel_sfpu_erf.h")`` resolves the same way, so the gate is
location-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

from .llk_params import MathOperation

# ─────────────────────────────────────────────────────────────────────────────
# Where a ported Quasar SFPU kernel may live
# ─────────────────────────────────────────────────────────────────────────────

# helpers/ -> python_tests/ -> tests/ -> tt-llk/
_TT_LLK_ROOT = Path(__file__).resolve().parents[3]
# tt-llk/ -> tt_metal/
_TT_METAL_ROOT = _TT_LLK_ROOT.parent

# Mirrors the Quasar include roots in test_config.py. Order is the compiler's search order,
# so the first hit is the header the build would actually pick up.
QUASAR_SFPU_HEADER_ROOTS: Tuple[Path, ...] = (
    _TT_METAL_ROOT / "hw" / "ckernels" / "quasar" / "metal" / "llk_api" / "llk_sfpu",
    _TT_LLK_ROOT / "tt_llk_quasar" / "common" / "inc" / "sfpu",
    _TT_LLK_ROOT / "tt_llk_quasar" / "common" / "inc" / "experimental",
)


class Arity(Enum):
    """Which Quasar test harness drives the kernel."""

    UNARY = "unary"  # test_eltwise_unary_sfpu_quasar
    BINARY = "binary"  # test_eltwise_binary_sfpu_quasar
    TERNARY = "ternary"  # test_sfpu_ternary_quasar
    STRUCTURAL = "structural"  # test_sfpu_structural_quasar (not element-wise)
    HELPER = "helper"  # no entry point of its own; covered through its callers


@dataclass(frozen=True)
class PortEntry:
    """One row of the parity set.

    Attributes:
        kernel: Dashboard kernel name. The header is always ``ckernel_sfpu_<kernel>.h``.
        ops: The MathOperation values this kernel provides. Several kernels provide more
            than one (``unary_comp`` provides six comparison modes, ``bitwise`` three).
            Empty only for :attr:`Arity.HELPER` rows.
        arity: Which harness drives it.
        call: Calculate-step symbols the C++ dispatcher must reference. The drift-guard
            test checks each one appears in the dispatcher once the header exists.
        init: Init-step symbols, empty when the op is stateless.
        has_approx: Whether ApproximationMode changes the result. Only true where the
            Blackhole kernel actually branches on APPROXIMATION_MODE or forwards it into
            ``sfpu_reciprocal`` -- carrying a template parameter it ignores does not count.
            Ops with this set sweep both modes and so emit twice the variants.
        int_formats: Whether the op operates on integer rather than float data, and so
            sweeps the integer format set instead of the float one.
        covered_by: HELPER rows only -- the ops whose stimuli exercise this header.
        note: Free text carried into failure messages and review.
    """

    kernel: str
    ops: Tuple[MathOperation, ...]
    arity: Arity
    call: Tuple[str, ...]
    init: Tuple[str, ...] = ()
    has_approx: bool = False
    int_formats: bool = False
    covered_by: Tuple[MathOperation, ...] = ()
    note: str = ""

    @property
    def header(self) -> str:
        """Header basename the gate resolves, and the spelling ``__has_include`` uses."""
        return f"ckernel_sfpu_{self.kernel}.h"

    @property
    def guard_macro(self) -> str:
        """The ``QSR_HAS_*`` macro the C++ dispatcher defines when the header resolves."""
        return f"QSR_HAS_{self.kernel.upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# The parity set: 57 kernels
#
# Rows are grouped by arity and alphabetical within a group, matching the dashboard.
# `call` / `init` names are the Blackhole entry points; a Quasar port is expected to keep
# them, and the drift guard reports it if one does not.
# ─────────────────────────────────────────────────────────────────────────────

_UNARY: Tuple[PortEntry, ...] = (
    # `activations` is an op-templated kernel; Hardsigmoid is its only instantiation today.
    PortEntry(
        "activations",
        (MathOperation.Hardsigmoid,),
        Arity.UNARY,
        call=("calculate_activation",),
        init=("hardsigmoid_init",),
        note="ActivationType-templated; Hardsigmoid is the only ActivationType implemented",
    ),
    PortEntry("add1", (MathOperation.Add1,), Arity.UNARY, call=("calculate_add1",)),
    PortEntry(
        "bitwise",
        (MathOperation.BitwiseAnd, MathOperation.BitwiseOr, MathOperation.BitwiseXor),
        Arity.UNARY,
        call=("calculate_sfpu_unary_bitwise",),
        init=("bitwise_and_init", "bitwise_or_init", "bitwise_xor_init"),
        int_formats=True,
        note="unary bitwise against a compile-time scalar; distinct from binary_bitwise",
    ),
    PortEntry(
        "bitwise_not",
        (MathOperation.BitwiseNot,),
        Arity.UNARY,
        call=("calculate_bitwise_not",),
        init=("bitwise_not_init",),
        int_formats=True,
    ),
    PortEntry(
        "cast_fp32_to_fp16a",
        (MathOperation.CastFp32ToFp16a,),
        Arity.UNARY,
        call=("cast_fp32_to_fp16a",),
    ),
    PortEntry(
        "cbrt",
        (MathOperation.Cbrt,),
        Arity.UNARY,
        call=("calculate_cube_root",),
        init=("cube_root_init",),
    ),
    PortEntry(
        "celu",
        (MathOperation.Celu,),
        Arity.UNARY,
        call=("calculate_celu",),
        init=("celu_init",),
    ),
    PortEntry(
        "digamma",
        (MathOperation.Digamma,),
        Arity.UNARY,
        call=("calculate_digamma",),
        init=("digamma_init",),
    ),
    PortEntry(
        "elu",
        (MathOperation.Elu,),
        Arity.UNARY,
        call=("calculate_elu",),
        init=("elu_init",),
    ),
    PortEntry(
        "erf",
        (MathOperation.Erf,),
        Arity.UNARY,
        call=("calculate_erf",),
        init=("erf_init",),
        has_approx=True,
        note="APPROX feeds sfpu_reciprocal's NR iteration count",
    ),
    PortEntry(
        "erfc",
        (MathOperation.Erfc,),
        Arity.UNARY,
        call=("calculate_erfc",),
        init=("erfc_init",),
    ),
    PortEntry(
        "erfinv",
        (MathOperation.Erfinv,),
        Arity.UNARY,
        call=("calculate_erfinv",),
        init=("erfinv_init",),
    ),
    PortEntry(
        "exp2",
        (MathOperation.Exp2,),
        Arity.UNARY,
        call=("calculate_exp2",),
        init=("exp2_init",),
    ),
    PortEntry(
        "expm1",
        (MathOperation.Expm1,),
        Arity.UNARY,
        call=("calculate_expm1",),
        init=("expm1_init",),
    ),
    PortEntry(
        "fmod",
        (MathOperation.Fmod,),
        Arity.UNARY,
        call=("calculate_fmod",),
        init=("init_fmod",),
        note="unary fmod against a fixed divisor (2.0f); binary_fmod takes two tiles",
    ),
    PortEntry(
        "hardmish",
        (MathOperation.Hardmish,),
        Arity.UNARY,
        call=("hardmish",),
        init=("hardmish_init",),
    ),
    PortEntry(
        "hardshrink",
        (MathOperation.Hardshrink,),
        Arity.UNARY,
        call=("calculate_hardshrink",),
        init=("hardshrink_init",),
    ),
    PortEntry(
        "hardtanh",
        (MathOperation.Hardtanh,),
        Arity.UNARY,
        call=("calculate_hardtanh",),
        init=("hardtanh_init",),
    ),
    PortEntry(
        "heaviside",
        (MathOperation.Heaviside,),
        Arity.UNARY,
        call=("calculate_heaviside",),
        init=("heaviside_init",),
    ),
    PortEntry(
        "i0",
        (MathOperation.I0,),
        Arity.UNARY,
        call=("calculate_i0",),
        init=("i0_init",),
    ),
    PortEntry(
        "i1",
        (MathOperation.I1,),
        Arity.UNARY,
        call=("calculate_i1",),
        init=("i1_init",),
        has_approx=True,
        note="APPROX feeds sfpu_reciprocal's NR iteration count",
    ),
    PortEntry(
        "identity",
        (MathOperation.Identity,),
        Arity.UNARY,
        call=("calculate_identity", "calculate_identity_uint"),
        note="float and unsigned-integer entry points; the harness picks by format",
    ),
    PortEntry(
        "lgamma",
        (MathOperation.Lgamma,),
        Arity.UNARY,
        call=("calculate_lgamma_stirling",),
        init=("lgamma_stirling_init",),
    ),
    PortEntry(
        "logical_not",
        (MathOperation.LogicalNotUnary,),
        Arity.UNARY,
        call=("calculate_logical_not",),
        init=("logical_not_unary_init",),
    ),
    PortEntry(
        "polygamma",
        (MathOperation.Polygamma,),
        Arity.UNARY,
        call=("calculate_polygamma",),
        init=("polygamma_init",),
        has_approx=True,
        note="APPROX selects the RECIP mode constant",
    ),
    PortEntry(
        "prelu",
        (MathOperation.Prelu,),
        Arity.UNARY,
        call=("calculate_prelu",),
        init=("prelu_init",),
    ),
    PortEntry(
        "rdiv",
        (MathOperation.Rdiv,),
        Arity.UNARY,
        call=("calculate_rdiv",),
        init=("rdiv_init",),
        has_approx=True,
        note="the only parity kernel with a literal if constexpr (APPROXIMATION_MODE) branch",
    ),
    PortEntry(
        "remainder",
        (MathOperation.Remainder,),
        Arity.UNARY,
        call=("calculate_remainder",),
        init=("init_remainder",),
        note="unary remainder against a fixed divisor (2.0f)",
    ),
    PortEntry(
        "rpow",
        (MathOperation.Rpow,),
        Arity.UNARY,
        call=("calculate_rpow",),
        init=("sfpu_binary_pow_init",),
    ),
    PortEntry(
        "selu",
        (MathOperation.Selu,),
        Arity.UNARY,
        call=("calculate_selu",),
        init=("selu_init",),
    ),
    PortEntry(
        "sign",
        (MathOperation.Sign,),
        Arity.UNARY,
        call=("calculate_sign",),
        init=("sign_init",),
    ),
    PortEntry(
        "softshrink",
        (MathOperation.Softshrink,),
        Arity.UNARY,
        call=("calculate_softshrink",),
        init=("softshrink_init",),
    ),
    PortEntry(
        "softsign",
        (MathOperation.Softsign,),
        Arity.UNARY,
        call=("calculate_softsign",),
        init=("init_softsign",),
        has_approx=True,
        note="APPROX feeds sfpu_reciprocal's NR iteration count",
    ),
    PortEntry(
        "tanhshrink",
        (MathOperation.Tanhshrink,),
        Arity.UNARY,
        call=("calculate_tanhshrink",),
        init=("tanhshrink_init",),
    ),
    PortEntry(
        "unary_comp",
        (
            MathOperation.UnaryGt,
            MathOperation.UnaryLt,
            MathOperation.UnaryGe,
            MathOperation.UnaryLe,
            MathOperation.UnaryNe,
            MathOperation.UnaryEq,
        ),
        Arity.UNARY,
        call=(
            "calculate_unary_gt",
            "calculate_unary_lt",
            "calculate_unary_ge",
            "calculate_unary_le",
            "calculate_unary_ne",
            "calculate_unary_eq",
        ),
        init=(
            "unary_gt_init",
            "unary_lt_init",
            "unary_ge_init",
            "unary_le_init",
            "unary_ne_init",
            "unary_eq_init",
        ),
        note="compare against a fixed scalar (UNARY_COMP_THRESHOLD = 0.5)",
    ),
    PortEntry(
        "unary_power",
        (MathOperation.UnaryPower,),
        Arity.UNARY,
        call=("calculate_unary_power",),
        init=("sfpu_unary_pow_init",),
        note="carries _float_to_int32_positive_ coverage for the conversions helper",
    ),
    PortEntry(
        "unary_shift",
        (MathOperation.LeftShift, MathOperation.RightShift),
        Arity.UNARY,
        call=("calculate_left_shift", "calculate_right_shift"),
        init=("left_shift_init", "right_shift_init"),
        int_formats=True,
    ),
    PortEntry(
        "xielu",
        (MathOperation.Xielu,),
        Arity.UNARY,
        call=("calculate_xielu",),
        init=("xielu_init",),
    ),
)

_BINARY: Tuple[PortEntry, ...] = (
    PortEntry(
        "atan2",
        (MathOperation.SfpuAtan2,),
        Arity.BINARY,
        call=("calculate_sfpu_atan2",),
        init=("calculate_sfpu_atan2_init",),
    ),
    PortEntry(
        "binary_bitwise",
        (
            MathOperation.SfpuBitwiseAnd,
            MathOperation.SfpuBitwiseOr,
            MathOperation.SfpuBitwiseXor,
        ),
        Arity.BINARY,
        call=("calculate_sfpu_binary_bitwise",),
        int_formats=True,
    ),
    PortEntry(
        "binary_fmod",
        (MathOperation.SfpuBinaryFmod, MathOperation.SfpuFmodInt32),
        Arity.BINARY,
        call=("calculate_sfpu_binary_fmod", "calculate_fmod_int32"),
        init=("fmod_binary_init", "fmod_int32_init"),
        has_approx=True,
        note="APPROX reaches the shared div_floor_init",
    ),
    PortEntry(
        "binary_pow",
        (MathOperation.SfpuElwpow,),
        Arity.BINARY,
        call=("calculate_sfpu_binary_pow",),
        init=("sfpu_binary_pow_init",),
        note="carries _float_to_int32_positive_ coverage for the conversions helper",
    ),
    PortEntry(
        "binary_remainder",
        (
            MathOperation.SfpuBinaryRemainder,
            MathOperation.SfpuRemainderInt32,
            MathOperation.SfpuRemainderUint32,
        ),
        Arity.BINARY,
        call=(
            "calculate_sfpu_binary_remainder",
            "calculate_remainder_int32",
            "calculate_remainder_uint32",
        ),
        init=("remainder_binary_init", "remainder_int32_init", "remainder_uint32_init"),
        has_approx=True,
        note="APPROX reaches the shared div_floor_init",
    ),
    PortEntry(
        "div_int32",
        (MathOperation.SfpuDivInt32,),
        Arity.BINARY,
        call=("calculate_div_int32",),
        init=("div_init",),
        int_formats=True,
    ),
    PortEntry(
        "div_int32_floor",
        (MathOperation.SfpuDivInt32Floor,),
        Arity.BINARY,
        call=("calculate_div_int32_floor", "calculate_div_int32_trunc"),
        init=("div_floor_init", "div_trunc_init"),
        has_approx=True,
        int_formats=True,
    ),
    PortEntry(
        "isclose",
        (MathOperation.SfpuIsclose,),
        Arity.BINARY,
        call=("calculate_sfpu_isclose",),
        init=("isclose_init",),
    ),
    PortEntry(
        "logsigmoid",
        (MathOperation.SfpuLogsigmoid,),
        Arity.BINARY,
        call=("calculate_logsigmoid",),
        init=("logsigmoid_init",),
        note="registered as a BinaryOp on Blackhole even though the math is unary",
    ),
    PortEntry(
        "mask",
        (MathOperation.SfpuMask,),
        Arity.BINARY,
        call=("calculate_mask", "calculate_int_mask", "calculate_mask_posinf"),
        init=("mask_init",),
    ),
    PortEntry(
        "rsub_int32",
        (MathOperation.SfpuRsubInt32,),
        Arity.BINARY,
        call=("calculate_rsub_int",),
        int_formats=True,
    ),
)

_TERNARY: Tuple[PortEntry, ...] = (
    PortEntry(
        "addcdiv",
        (MathOperation.SfpuAddcdiv,),
        Arity.TERNARY,
        call=("calculate_addcdiv",),
        init=("init_addcdiv",),
        has_approx=True,
        note="uses sfpu_reciprocal internally; carries float32_to_bf16_rne coverage",
    ),
    PortEntry(
        "addcmul",
        (MathOperation.SfpuAddcmul,),
        Arity.TERNARY,
        call=("calculate_addcmul",),
    ),
    PortEntry(
        "lerp",
        (MathOperation.SfpuLerp,),
        Arity.TERNARY,
        call=("calculate_lerp",),
        note="carries float32_to_bf16_rne coverage for the conversions helper",
    ),
    PortEntry(
        "snake_beta",
        (MathOperation.SfpuSnakeBeta,),
        Arity.TERNARY,
        call=("calculate_snake_beta",),
        init=("snake_beta_init",),
    ),
)

_STRUCTURAL: Tuple[PortEntry, ...] = (
    PortEntry(
        "alt_complex_rotate90",
        (MathOperation.AltComplexRotate90,),
        Arity.STRUCTURAL,
        call=("calculate_alt_complex_rotate90",),
        init=("alt_complex_rotate90_init",),
        note="rotates interleaved (re, im) pairs; reads a neighbouring lane, not element-wise",
    ),
    PortEntry(
        "int_sum",
        (MathOperation.IntSumRow, MathOperation.IntSumCol),
        Arity.STRUCTURAL,
        call=("calculate_sum_int_row", "calculate_sum_int_col"),
        init=("sum_int_init",),
        int_formats=True,
        note="reduction along a tile axis",
    ),
    PortEntry(
        "tiled_prod",
        (MathOperation.TiledProd,),
        Arity.STRUCTURAL,
        call=("calculate_tiled_prod",),
        init=("tiled_prod_init",),
        note="running product across the tile",
    ),
)

_HELPER: Tuple[PortEntry, ...] = (
    # ckernel_sfpu_conversions.h holds exactly two sfpi helpers and no entry point of its
    # own, so it has no test of its own either. Its callers -- all of them inside the
    # parity set -- carry targeted stimuli that drive both helpers; see the
    # CONVERSIONS_COVERAGE table below for what each contributes.
    PortEntry(
        "conversions",
        (),
        Arity.HELPER,
        call=("_float_to_int32_positive_", "float32_to_bf16_rne"),
        covered_by=(
            MathOperation.UnaryPower,
            MathOperation.SfpuElwpow,
            MathOperation.SfpuLerp,
            MathOperation.SfpuAddcdiv,
        ),
        note="helper-only header: no dispatchable entry point",
    ),
)

QUASAR_SFPU_PARITY: Tuple[PortEntry, ...] = (
    _UNARY + _BINARY + _TERNARY + _STRUCTURAL + _HELPER
)

# What each carrier op must inject so the conversions helpers are genuinely exercised
# rather than merely reachable. Consumed by the harnesses to seed extra stimuli, and by
# the drift guard to check the claim is still backed by a live op.
CONVERSIONS_COVERAGE: Dict[MathOperation, str] = {
    MathOperation.UnaryPower: "_float_to_int32_positive_ -- integer-valued and near-half exponents",
    MathOperation.SfpuElwpow: "_float_to_int32_positive_ -- integer-valued and near-half exponents",
    MathOperation.SfpuLerp: "float32_to_bf16_rne -- round-nearest-even ties at a Float16_b output",
    MathOperation.SfpuAddcdiv: "float32_to_bf16_rne -- round-nearest-even ties at a Float16_b output",
}


# ─────────────────────────────────────────────────────────────────────────────
# Derived lookups
# ─────────────────────────────────────────────────────────────────────────────

_ENTRY_BY_OP: Dict[MathOperation, PortEntry] = {
    op: entry for entry in QUASAR_SFPU_PARITY for op in entry.ops
}

_ENTRY_BY_KERNEL: Dict[str, PortEntry] = {e.kernel: e for e in QUASAR_SFPU_PARITY}


def entry_for(op: MathOperation) -> Optional[PortEntry]:
    """The parity entry providing *op*, or None when *op* is not in the parity set."""
    return _ENTRY_BY_OP.get(op)


def entry_for_kernel(kernel: str) -> Optional[PortEntry]:
    """The parity entry for a dashboard kernel name, or None."""
    return _ENTRY_BY_KERNEL.get(kernel)


def resolve_header(entry: PortEntry) -> Optional[Path]:
    """Where *entry*'s header actually is, searching the include roots in compiler order.

    Returns None when the kernel has not been ported to Quasar yet.
    """
    for root in QUASAR_SFPU_HEADER_ROOTS:
        candidate = root / entry.header
        if candidate.is_file():
            return candidate
    return None


def is_ported(op: MathOperation) -> bool:
    """Whether the Quasar kernel providing *op* exists in the tree.

    Ops outside the parity set are reported as ported: they are not gated by this module,
    and answering False would silently drop sweeps that already run today.
    """
    entry = entry_for(op)
    if entry is None:
        return True
    return resolve_header(entry) is not None


def ported_ops() -> FrozenSet[MathOperation]:
    """Every parity op whose Quasar kernel is present."""
    return frozenset(op for op in _ENTRY_BY_OP if is_ported(op))


def unported_ops() -> FrozenSet[MathOperation]:
    """Every parity op still waiting on its Quasar kernel."""
    return frozenset(op for op in _ENTRY_BY_OP if not is_ported(op))


def parity_ops(arity: Optional[Arity] = None) -> FrozenSet[MathOperation]:
    """Every parity op, optionally restricted to one harness."""
    return frozenset(
        op
        for entry in QUASAR_SFPU_PARITY
        if arity is None or entry.arity is arity
        for op in entry.ops
    )


def entries(arity: Optional[Arity] = None) -> Tuple[PortEntry, ...]:
    """Parity entries, optionally restricted to one harness."""
    if arity is None:
        return QUASAR_SFPU_PARITY
    return tuple(e for e in QUASAR_SFPU_PARITY if e.arity is arity)


def port_status_summary() -> str:
    """One line per unported kernel, for a skip reason or a failure message."""
    missing = sorted(
        e.kernel for e in QUASAR_SFPU_PARITY if e.ops and resolve_header(e) is None
    )
    if not missing:
        return "all parity kernels ported"
    return (
        f"{len(missing)} parity kernels not yet ported to Quasar: {', '.join(missing)}"
    )
