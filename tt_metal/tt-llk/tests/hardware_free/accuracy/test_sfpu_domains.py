# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for sfpu_domains: the gates that decide where a special or a signed
zero survives the pipeline, the probe spacing that keeps an edge probe distinct in a
16-bit Dest, and the exp family's two ceilings. Pure metadata, no device."""

import math
import re
import struct
from pathlib import Path

import pytest
import test_eltwise_binary_sfpu
import torch
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BinarySFPUGolden,
    UnarySFPUGolden,
    sfpu_max,
    sfpu_min,
    sfpu_order_key_elementwise,
    sfpu_total_order_key,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    ReducePool,
    Tilize,
)
from helpers.sfpu_domains import (
    _BINARY_SPECIALS_NOT_READY,
    BINARY_SPECIALS_READY_OPS,
    GENERATED_NAN_SIGN_OPS,
    Operand,
    dest_truncation_mask,
    edge_pair_values,
    edge_values,
    for_op,
    generated_nan_sign_is_asserted,
    nan_sign_is_unspecified,
    nan_survives_to_l1,
    ops_with_singularity,
    probe_spacing_format,
    sfpu_unary_ops,
    specials_safe,
)

F32, F16, F16B = DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b
BFP8, BFP4 = DataFormat.Bfp8_b, DataFormat.Bfp4_b
D_NO, D_YES = DestAccumulation.No, DestAccumulation.Yes
NO, YES = ApproximationMode.No, ApproximationMode.Yes

# ── Where a special survives ──────────────────────────────────────────────────

# The 5x5x2 matrix measured on Wormhole over the isinf/isnan predicates, and the seven
# cells it accepts: a Float32 input carries specials at either Dest into any non-block
# output (Float16 only with the 32-bit Dest); a Float16_b input only at dest_acc=No; a
# Float16 or block-float input never.
_MEASURED_FORMATS = [F32, F16, F16B, BFP8, BFP4]
_ACCEPTED = {
    (F32, F32, D_NO),
    (F32, F16B, D_NO),
    (F32, F32, D_YES),
    (F32, F16, D_YES),
    (F32, F16B, D_YES),
    (F16B, F32, D_NO),
    (F16B, F16B, D_NO),
}

# The {Float16_b, Float32} matrix the two edge sweeps parametrize over.
_EDGE_SWEEP_CELLS = [
    (inp, out, dest_acc)
    for inp in (F16B, F32)
    for out in (F16B, F32)
    for dest_acc in (D_NO, D_YES)
]


def test_specials_safe_is_the_measured_matrix():
    """Every measured cell against the checked-in verdict; anything unmeasured (an
    integer, Fp8 or MX input, an MX output) is refused rather than guessed."""
    for inp in _MEASURED_FORMATS:
        for out in _MEASURED_FORMATS:
            for dest_acc in (D_NO, D_YES):
                cell = (inp, out, dest_acc)
                assert specials_safe(*cell) is (cell in _ACCEPTED), cell
    for unlisted in (
        DataFormat.Tf32,
        DataFormat.Int32,
        DataFormat.UInt32,
        DataFormat.Fp8_e4m3,
        DataFormat.MxFp8P,
        DataFormat.Bfp2_b,
    ):
        assert not specials_safe(unlisted, F32, D_NO)
    for mx_output in (DataFormat.MxFp8P, DataFormat.MxFp8R, DataFormat.MxFp4):
        assert not specials_safe(F32, mx_output, D_YES)


def test_a_flag_is_read_by_its_value_and_only_its_own_enum_is_accepted():
    """`bool(DestAccumulation.No)` is True, so a truthiness read would flip whole rows.
    One normaliser (`data_format_inference._two_state_flag`) serves every predicate: a
    member or a bool is read by its value; an int or the other two-state enum is
    refused rather than guessed."""
    assert specials_safe(F16B, F16B, D_NO)
    assert not specials_safe(F16B, F16B, D_YES)
    assert specials_safe(F16B, F16B, False) and not specials_safe(F16B, F16B, True)
    assert for_op(MathOperation.Exp, F32, approx_mode=YES).spec_A.high == 16.0
    for bad in (1, 0, None, "Yes", ApproximationMode.Yes):
        with pytest.raises(TypeError):  # allow-pytest.raises: host-only test
            specials_safe(F32, F32, bad)
    for bad in (1, DestAccumulation.Yes):
        with pytest.raises(TypeError):  # allow-pytest.raises: host-only test
            for_op(MathOperation.Exp, F32, approx_mode=bad)
    with pytest.raises(TypeError):  # allow-pytest.raises: host-only test
        probe_spacing_format(F32, 1)


def test_unpacks_to_dest():
    """Only a 32-bit input at dest_acc=Yes unpacks to Dest, and a tilized Float32 never
    does; a bool is accepted, anything else but a DestAccumulation is refused."""
    assert unpacks_to_dest(F32, D_YES)
    assert unpacks_to_dest(DataFormat.Int32, True)
    assert not unpacks_to_dest(F32, D_NO)
    assert not unpacks_to_dest(F16B, D_YES)
    assert not unpacks_to_dest(F32, D_YES, Tilize.Yes)
    assert unpacks_to_dest(DataFormat.Int32, D_YES, Tilize.Yes)
    for bad in (1, 0, None, "Yes", 1.0, ApproximationMode.Yes):
        with pytest.raises(TypeError):  # allow-pytest.raises: host-only test
            unpacks_to_dest(F32, bad)


def test_nan_survives_only_into_a_32_bit_dest_and_a_32_bit_pack():
    """On the edge sweeps' matrix exactly one cell carries a NaN to L1 as a NaN; the
    other five accepted cells are where a generated NaN's sign becomes a +/-inf."""
    carrying = [c for c in _EDGE_SWEEP_CELLS if specials_safe(*c)]
    assert len(carrying) == 6
    assert [c for c in carrying if nan_survives_to_l1(*c)] == [(F32, F32, D_YES)]


def test_generated_nan_sign_gate_is_the_narrowing_cells_on_wormhole_only():
    """The binary family's gate has no op argument: exactly the cells that narrow, and
    nothing on Blackhole, whose SFPMAD promises the canonical NaN."""
    for cell in _EDGE_SWEEP_CELLS:
        if not specials_safe(*cell):
            continue
        narrows = not nan_survives_to_l1(*cell)
        assert generated_nan_sign_is_asserted(*cell, on_wormhole=True) == narrows
        assert not generated_nan_sign_is_asserted(*cell, on_wormhole=False)


def test_nan_sign_gate_ignores_ops_that_forward_a_nan():
    """Neg, Abs and Identity move the sign bit rather than inventing one, so their NaN
    sign is a datum and stays asserted on every cell."""
    for op in (MathOperation.Neg, MathOperation.Abs, MathOperation.Identity):
        assert op not in GENERATED_NAN_SIGN_OPS
        for cell in _EDGE_SWEEP_CELLS:
            assert not nan_sign_is_unspecified(op, *cell)


def test_nan_sign_gate_skips_only_the_narrowing_cells_of_ops_that_invent_a_nan():
    """The positive side: every op that invents its NaN is gated on exactly the cells
    that narrow a NaN to +/-inf, and keeps Float32->Float32 at dest_acc=Yes, where the
    sign is readable. A gate that grew would switch off specials probes silently."""
    assert {op.name for op in GENERATED_NAN_SIGN_OPS} == {
        "Cos",
        "Fmod",
        "GeluAppx",
        "Hardmish",
        "Mish",
        "Rsqrt",
        "ScalarRsub",
        "Silu",
        "Sin",
        "Softsign",
        "Tan",
    }
    carrying = [c for c in _EDGE_SWEEP_CELLS if specials_safe(*c)]
    narrowing = [c for c in carrying if c != (F32, F32, D_YES)]
    for op in GENERATED_NAN_SIGN_OPS:
        gated = [c for c in carrying if nan_sign_is_unspecified(op, *c)]
        assert gated == narrowing, op.name


def test_every_float_binary_op_is_classified_for_cat_b():
    """Enrolled or recorded as not ready, for every float op the binary sweep drives, so
    an op with cat B off is a decision rather than an omission."""
    candidates = (
        test_eltwise_binary_sfpu._CLASSIFIED_STIMULI_OPS
        - test_eltwise_binary_sfpu._INT_DRIVEN_BINARY_OPS
    )
    classified = set(BINARY_SPECIALS_READY_OPS) | set(_BINARY_SPECIALS_NOT_READY)
    assert sorted(op.name for op in candidates - classified) == []
    assert sorted(op.name for op in classified - candidates) == []


# ── Probe spacing ─────────────────────────────────────────────────────────────

# Everything edge_values() can be asked about.
_PROBED_OPS = sorted(
    set(sfpu_unary_ops()) | set(ops_with_singularity()), key=lambda op: op.name
)


def _as_16bit_dest(value: float) -> float:
    """*value* as a 16-bit Dest holds it: fp32 with the low mantissa half dropped."""
    if value != value or value in (float("inf"), float("-inf")):
        return value
    raw = struct.unpack("<I", struct.pack("<f", value))[0] & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def _distinct_key(value: float):
    """Identity as delivered, keeping the two zeros apart."""
    delivered = _as_16bit_dest(value)
    if delivered == 0.0:
        return ("zero", struct.pack("<f", delivered)[-1] & 0x80)
    return ("value", delivered)


@pytest.mark.parametrize(
    "input_format,output_format",
    [(i, o) for i in (F16B, F32) for o in (F16B, F32)],
    ids=lambda f: f.name,
)
@pytest.mark.parametrize("operand", [Operand.A, Operand.B], ids=lambda o: o.name)
def test_edge_probes_stay_distinct_in_a_16bit_dest(
    input_format, output_format, operand
):
    """A probe that quantizes onto its own boundary tests the boundary twice while
    reading as coverage; modelled here as an independent `& 0xFFFF0000`."""
    for op in _PROBED_OPS:
        probes = edge_values(op, input_format, output_format, operand, dest_acc=D_NO)
        delivered = [_distinct_key(v) for v in probes]
        assert len(set(delivered)) == len(probes), (op.name, probes)


def test_acosh_above_pole_probe_is_widened_at_dest_acc_no():
    """The measured case: the fp32 probe 0x3F800002 above 1.0 arrives as 1.0 in a
    16-bit Dest, so the step widens to bfloat16's; a 32-bit Dest keeps the tight one."""
    for output_format in (F32, F16B):
        assert edge_values(MathOperation.Acosh, F32, output_format, dest_acc=D_NO) == [
            1.0,
            1.015625,
        ]
        assert edge_values(MathOperation.Acosh, F32, output_format, dest_acc=D_YES) == [
            1.0,
            1.0000002384185791,
        ]


def test_tight_probes_are_not_loosened_where_they_survive():
    """Stepping down from 1.0 crosses a binade and survives the truncation that destroys
    the step up, and a zero-pole's probe keeps its exponent, so neither is widened."""
    two_fp32_ulps_at_one = 1.0 - 2 * 2**-23
    for op in (
        MathOperation.Asin,
        MathOperation.Acos,
        MathOperation.Atanh,
        MathOperation.Erfinv,
    ):
        probes = edge_values(op, F32, F32, dest_acc=D_NO)
        assert two_fp32_ulps_at_one in probes and -two_fp32_ulps_at_one in probes
    for op in (MathOperation.Sqrt, MathOperation.Log, MathOperation.Reciprocal):
        probes = edge_values(op, F32, F32, dest_acc=D_NO)
        assert min(abs(v) for v in probes if v != 0.0) == 2 * 2**-23


def test_pow_edge_pairs_include_negative_zero_exponent():
    """The setsgn(pow, 0) guard is only as good as the pairs that would fail without it:
    on the pipeline that delivers a signed zero, exponent -0.0 is crossed with a
    positive, a negative and a zero base; on the datacopy path it is not claimed."""

    def negative_zero(v):
        return v == 0.0 and math.copysign(1.0, v) < 0.0

    pairs = edge_pair_values(MathOperation.SfpuElwpow, F32, F32, dest_acc=D_YES)
    bases = [a for a, b in pairs if negative_zero(b)]
    assert any(a > 0 for a in bases) and any(a < 0 for a in bases)
    assert any(a == 0 for a in bases)
    flattened = edge_pair_values(MathOperation.SfpuElwpow, F16B, F16B, dest_acc=D_NO)
    assert not any(negative_zero(b) for _, b in flattened)


def test_dest_truncation_masks_match_the_mantissa_width_table():
    assert dest_truncation_mask(F16) == 0xFFFFE000
    assert dest_truncation_mask(F16B) == 0xFFFF0000


# ── The exp family's two ceilings ─────────────────────────────────────────────


def test_exp_family_ceiling_follows_the_approximation_mode():
    """The registry holds the range bound; only ApproximationMode.Yes narrows to the
    approximation's accuracy ceiling, exp_with_base's is double exp's in both modes, and
    a narrow format's branch already under the ceiling is left alone."""
    for op, range_high, approx_high in (
        (MathOperation.Exp, 80.0, 16.0),
        (MathOperation.Exp2, 100.0, 23.0),
        (MathOperation.ExpWithBase, 160.0, 32.0),
    ):
        assert for_op(op, F32).spec_A.high == range_high
        assert for_op(op, F32, approx_mode=NO).spec_A.high == range_high
        assert for_op(op, F32, approx_mode=YES).spec_A.high == approx_high
    for mode in (NO, YES):
        exp = for_op(MathOperation.Exp, F32, approx_mode=mode).spec_A.high
        base = for_op(MathOperation.ExpWithBase, F32, approx_mode=mode).spec_A.high
        assert base == 2 * exp
    for fmt, high in (
        (DataFormat.MxFp8P, 5.0),
        (DataFormat.Fp8_e4m3, 5.0),
        (F16, 10.0),
    ):
        for mode in (None, NO, YES):
            assert for_op(MathOperation.Exp, fmt, approx_mode=mode).spec_A.high == high


# ── Golden-generator rules ────────────────────────────────────────────────────

# The goldens' model of the datapath (Dest format, NaN sign, the SFPU total order) has
# no host test file of its own; these sit beside the pipeline rules they mirror.


def test_binary_golden_dest_format_matches_the_domains_rule():
    """BinarySFPUGolden._dest_format and nan_survives_to_l1 must derive the same Dest.

    Three places state this rule -- the unary golden, the binary golden, and sfpu_domains'
    internal derivation -- and a silent disagreement would put the golden's NaN substitution on
    a different set of cells than the gate that decides where the probe may be asserted. Checked
    rather than commented, since the two live in different modules.
    """
    for input_format, output_format, dest_acc in _EDGE_SWEEP_CELLS:
        dst = BinarySFPUGolden._dest_format(input_format, output_format, dest_acc)
        preserves = (dst, output_format) in {
            (DataFormat.Float16, DataFormat.Float16),
            (DataFormat.Float32, DataFormat.Float16),
            (DataFormat.Float32, DataFormat.Float32),
        }
        assert preserves == nan_survives_to_l1(input_format, output_format, dest_acc), (
            f"{input_format.name}->{output_format.name} dest_acc={dest_acc}: the golden "
            f"derives Dest={dst.name}, which disagrees with nan_survives_to_l1"
        )


def test_binary_golden_requires_dest_acc_and_output_format_together():
    """Supplying one without the other is rejected rather than half-modelled.

    Half the contract is worse than none of it: the Dest width would come from dest_acc while
    the pack decision silently defaulted, giving a golden wrong in a new way rather than in the
    documented old one.
    """
    golden = BinarySFPUGolden()
    tile_pair = torch.zeros(2048, dtype=torch.float32)
    for missing in ("output_format", "dest_acc"):
        kwargs = {
            "dest_acc": DestAccumulation.No,
            "output_format": DataFormat.Float32,
        }
        del kwargs[missing]
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            ValueError, match="must be supplied together"
        ):
            golden(
                MathOperation.SfpuElwadd,
                tile_pair.clone(),
                0,
                1,
                0,
                32,
                [64, 32],
                DataFormat.Float32,
                **kwargs,
            )


def test_binary_comparison_family_splits_on_the_kernel_nan_guard():
    """max/min follow the SFPU total order; the six comparisons do not. Pinned both ways.

    This split is not derivable from the ISA, which is why it is pinned rather than commented.
    All eight ops route through SFPSWAP, whose page specifies SignMagIsSmaller() and the total
    order -- but the six comparison kernels wrap the swap in an explicit NaN rejection
    ("rejects NaN", `SFPIADD(inf, |a|+|b|, CC_GTE0)`), so a NaN operand never reaches the compare
    and the IEEE unordered answer stands. binary_max_min is a bare TTI_SFPSWAP with no such
    guard, so for those two the order does reach the result.

    Measured on a Wormhole n150: modelling all eight on the total order failed the six
    comparisons on 4 cells each and passed max/min everywhere.

    Both NaN signs are probed for max/min, because torch.maximum agrees with the total order on
    +NaN by coincidence and disagrees on -NaN -- a one-sided probe certifies a wrong golden.
    """
    golden = BinarySFPUGolden()
    nan, inf = float("nan"), float("inf")

    # The six: IEEE unordered, because the kernel rejects a NaN operand outright.
    ieee_probes = [(nan, 1.0), (1.0, nan), (nan, nan), (nan, inf), (-inf, nan)]
    for op, expected in {
        MathOperation.SfpuElwLt: 0.0,
        MathOperation.SfpuElwGt: 0.0,
        MathOperation.SfpuElwLe: 0.0,
        MathOperation.SfpuElwGe: 0.0,
        MathOperation.SfpuElwEq: 0.0,
        MathOperation.SfpuElwNe: 1.0,
    }.items():
        for a, b in ieee_probes:
            got = float(golden.ops[op](torch.tensor(a), torch.tensor(b)))
            assert got == expected, (
                f"{op.name}({a}, {b}) = {got}, expected {expected}. These kernels reject a NaN "
                "operand before the compare, so the answer is IEEE's unordered one and NOT the "
                "SFPU total order. Do not 'fix' this from the SFPSWAP ISA page -- read the guard "
                "in calculate_binary_comp_fp32_* first."
            )

    # max/min: the total order, both NaN signs.
    for op, reference in (
        (MathOperation.SfpuBinaryMax, sfpu_max),
        (MathOperation.SfpuBinaryMin, sfpu_min),
    ):
        for a, b in [(nan, 1.0), (-nan, 1.0), (nan, inf), (-inf, nan), (nan, nan)]:
            got = float(golden.ops[op](torch.tensor(a), torch.tensor(b)))
            want = float(reference(a, b))
            assert got == want or (got != got and want != want), (
                f"{op.name}({a}, {b}) = {got}, but the total order gives {want}. "
                "binary_max_min is a bare SFPSWAP(VEC_MIN_MAX) with no NaN guard."
            )

    # The integer axis, which shares the same six entries plus max/min, must be unaffected.
    for op, (a, b, want) in {
        MathOperation.SfpuElwLt: (-5, 3, 1.0),
        MathOperation.SfpuElwGe: (-5, 3, 0.0),
        MathOperation.SfpuElwLe: (7, 7, 1.0),
        MathOperation.SfpuElwGt: (7, 7, 0.0),
        MathOperation.SfpuMaxInt32: (-5, 3, 3.0),
        MathOperation.SfpuMinInt32: (-5, 3, -5.0),
    }.items():
        got = float(
            golden.ops[op](
                torch.tensor(a, dtype=torch.int32), torch.tensor(b, dtype=torch.int32)
            )
        )
        assert got == want, (
            f"{op.name}({a}, {b}) = {got} on the Int32 axis, expected {want}: "
            "calculate_binary_comp_int32 is a plain two's-complement compare"
        )


def _binary_op_enumerators(arch_dir: str) -> set:
    """The BinaryOp enumerator names one architecture declares, parsed from its
    ckernel_defs.h: a copy of the enum here would keep agreeing with itself after the
    header changed, which is the failure this guard exists to catch."""
    header = (
        Path(__file__).resolve().parents[3] / arch_dir / "common/inc/ckernel_defs.h"
    )
    assert header.is_file(), (
        f"{header} not found. This guard pins the WH/BH BinaryOp enum; if the layout "
        "moved, repoint it rather than deleting it."
    )
    body = re.search(
        r"enum class BinaryOp\s*:\s*[^{]*\{(.*?)\}", header.read_text(), re.S
    )
    assert body, f"no `enum class BinaryOp` found in {header}"
    return set(re.findall(r"^\s*([A-Z][A-Z0-9_]*)\s*(?:=|,)", body.group(1), re.M))


# These carry MathOpType.SFPU_BINARY_INT, which only the Quasar dispatch implements; on
# WH/BH the same kernels are reached through the SFPU_BINARY members they map to, at
# DataFormat.Int32. The int multiply is MUL on Quasar (_mul_int32_) and MUL_INT32 on
# WH/BH, which SfpuMulInt32 drives.
_QUASAR_INT_BINARY_ALIASES = {
    MathOperation.SfpuGtInt: MathOperation.SfpuElwGt,
    MathOperation.SfpuLtInt: MathOperation.SfpuElwLt,
    MathOperation.SfpuLeInt: MathOperation.SfpuElwLe,
    MathOperation.SfpuGeInt: MathOperation.SfpuElwGe,
    MathOperation.SfpuElwmulInt: MathOperation.SfpuMulInt32,
}


@pytest.mark.parametrize("arch_dir", ["tt_llk_wormhole_b0", "tt_llk_blackhole"])
def test_quasar_int_binary_members_alias_covered_kernels(arch_dir):
    """The SFPU_BINARY_INT members are unreachable on WH/BH; their kernels are not. A
    member that became dispatchable needs a test of its own; an alias that left the
    header leaves the kernel uncovered."""
    declared = _binary_op_enumerators(arch_dir)
    for member, alias in _QUASAR_INT_BINARY_ALIASES.items():
        spec = member.value
        if member is MathOperation.SfpuElwmulInt:
            # "MUL" is declared as the float multiply: dispatch follows MathOpType.
            assert spec.operation_type.name == "SFPU_BINARY_INT"
        else:
            assert spec.cpp_enum_value not in declared, (
                f"{member.name} names BinaryOp::{spec.cpp_enum_value}, now declared in "
                f"{arch_dir}: it is reachable there and needs a test of its own"
            )
        assert alias.value.cpp_enum_value in declared, (
            f"{member.name}'s kernel is covered on WH/BH only through {alias.name}, whose "
            f"BinaryOp::{alias.value.cpp_enum_value} is no longer declared in {arch_dir}"
        )


def test_int_comparison_ops_are_the_ordered_elw_compares():
    driven = set(test_eltwise_binary_sfpu._INT_COMPARISON_OPS)
    assert driven == {
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwGe,
    }
    assert (
        set(_QUASAR_INT_BINARY_ALIASES.values()) - {MathOperation.SfpuMulInt32}
        == driven
    ), "the alias table and the driven set disagree"


def test_total_order_key_matches_the_isa_remap():
    """Both order keys must reproduce `SignMagIsSmaller()`'s remap, signed zeros included.

    The ISA spells the remap as an xor with the sign bit smeared down over the magnitude -- an
    arithmetic shift right by 30, then a logical shift right by 1 -- which is a mask of 0x7FFFFFFF
    where the sign bit is set and 0 where it is not. So a negative with magnitude m ranks at
    -1 - m, putting -0.0 at -1 and strictly below +0.0 at 0: the `-0 < +0` the documented chain
    shows.

    Ranking a negative at -m instead is order-isomorphic *everywhere except the two zeros*, where
    it ties them -- which is why this is worth pinning rather than leaving to the NaN probes
    above. A tie makes sfpu_min/sfpu_max return whichever operand came first, and no device test
    can catch it: passed_test() cannot see a zero's sign.
    """

    def isa_remap(value: float) -> int:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        mask = ((bits >> 31) & 1) * 0x7FFFFFFF
        return struct.unpack("<i", struct.pack("<I", bits ^ mask))[0]

    probes = [
        -float("nan"),
        -float("inf"),
        -3.5,
        -1.0,
        -1.4e-45,
        -0.0,
        0.0,
        1.4e-45,
        1.0,
        3.5,
        float("inf"),
        float("nan"),
    ]

    for value in probes:
        assert sfpu_total_order_key(value) == isa_remap(value), (
            f"sfpu_total_order_key({value!r}) disagrees with SignMagIsSmaller()'s remap. The "
            "mask is 0x7FFFFFFF when the sign bit is set, so a negative ranks at -1 - magnitude."
        )

    vectorised = sfpu_order_key_elementwise(torch.tensor(probes, dtype=torch.float32))
    assert vectorised.tolist() == [isa_remap(v) for v in probes], (
        "sfpu_order_key_elementwise disagrees with the scalar key. The binary and reduce goldens "
        "use the vectorised one, so a drift here splits the two families' semantics."
    )

    assert sfpu_total_order_key(-0.0) < sfpu_total_order_key(0.0), (
        "-0.0 must rank strictly below +0.0. Tying them makes min/max operand-order-dependent, "
        "and passed_test() cannot see a zero's sign, so no device variant would fail."
    )

    # Order-independent in both directions, which the tie is what would break.
    for a, b in ((-0.0, 0.0), (0.0, -0.0)):
        assert math.copysign(1.0, sfpu_min(a, b)) < 0.0, (
            f"sfpu_min({a!r}, {b!r}) must be -0.0 under the total order, whatever the operand "
            "order"
        )
        assert math.copysign(1.0, sfpu_max(a, b)) > 0.0, (
            f"sfpu_max({a!r}, {b!r}) must be +0.0 under the total order, whatever the operand "
            "order"
        )


def test_hardtanh_golden_matches_the_clamp_golden():
    """Hardtanh's golden must stay Clamp's golden -- pin the identity.

    Both ops bind metal kernels that are the same SFPSWAP max-then-min composition
    (calculate_clamp's unary_max_min chain; calculate_hardtanh's sfpi::clamp), so one
    golden -- sfpu_clamp -- models both. If either golden is ever remodelled
    independently, the divergence surfaces here.
    """
    golden = UnarySFPUGolden()
    low, high = -1.0, 1.0
    probes = [
        -float("inf"),
        -2.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        2.0,
        float("inf"),
        float("nan"),
    ]
    for x in probes:
        want = float(golden._clamp(x, low, high))
        got = float(golden._hardtanh(x, low, high))
        assert got == want or (got != got and want != want), (
            f"hardtanh({x}) golden gives {got} but the clamp golden gives {want}; "
            "the two goldens must move together while both ops bind the same composition."
        )


@pytest.mark.parametrize(
    "output_format,dest_acc",
    [
        (DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DestAccumulation.No),
        (DataFormat.Float32, DestAccumulation.Yes),
    ],
)
def test_fmod_golden_preserves_dividend_sign_through_pack(output_format, dest_acc):
    probes = torch.tensor([float("inf"), -float("inf"), 5.0, -5.0])
    result = UnarySFPUGolden()(
        MathOperation.Fmod,
        probes.repeat(256),
        output_format,
        dest_acc,
        DataFormat.Float32,
        dimensions=(32, 32),
    )

    if output_format == DataFormat.Float32 and dest_acc == DestAccumulation.Yes:
        assert torch.isnan(result[:2]).all()
    else:
        assert torch.equal(result[:2].float(), probes[:2])
    assert torch.equal(torch.signbit(result[:4]), torch.signbit(probes))
    expected = torch.fmod(probes[2:], UnarySFPUGolden._FMOD_DIVISOR)
    assert torch.equal(result[2:4].float(), expected)


def test_reduce_extremum_follows_the_total_order_on_floats_only():
    """Reduce MAX/MIN fold under the SFPU total order; Sum/Average stay IEEE; ints stay torch.

    ckernel_sfpu_reduce.h reduces MAX/MIN with a bare TTI_SFPSWAP(VEC_MIN_MAX) and no NaN guard, so
    unlike the six binary comparisons the order does reach the result. MIN is the load-bearing case:
    a column holding one +NaN must reduce to the *finite* minimum, where torch.min propagates.

    Integers are checked in the same test because ReduceColumn/ReduceRow are deliberately *not*
    routed through _call_integer, so an Int32 reduce reaches the same helper -- and there the model
    is _emit_int32_signed_cswap_, i.e. plain two's complement.
    """
    nan, inf = float("nan"), float("inf")
    fold = UnarySFPUGolden._reduce_extremum

    column = torch.tensor([[1.0], [nan], [-2.0], [inf]])
    assert math.isnan(
        float(fold(column, dim=0, want_max=True)[0])
    ), "max over a column containing +NaN must be NaN: +NaN is the total order's maximum"
    assert float(fold(column, dim=0, want_max=False)[0]) == -2.0, (
        "min over a column containing +NaN must be the finite minimum (-2.0), not NaN. "
        "torch.min propagates the NaN, which is IEEE and not what SFPSWAP does."
    )

    negative_nan = torch.tensor([[1.0], [-nan], [2.0]])
    assert math.isnan(
        float(fold(negative_nan, dim=0, want_max=False)[0])
    ), "-NaN is the total order's minimum, so min must return it"
    assert (
        float(fold(negative_nan, dim=0, want_max=True)[0]) == 2.0
    ), "-NaN must not win a max; this is the direction torch.maximum gets wrong"

    ints = torch.tensor([[5], [-7], [3]], dtype=torch.int32)
    assert int(fold(ints, dim=0, want_max=False)[0]) == -7
    assert int(fold(ints, dim=0, want_max=True)[0]) == 5


def test_binary_nan_sign_is_relaxed_only_where_sfpmad_emits_it():
    """An emitted NaN loses its sign; a selected one keeps it. Per op, not per lane.

    `SFPMAD.md` scopes its NaN wording to "if a NaN is emitted" and draws no line between a NaN it
    computed and one that arrived on an input -- Blackhole gives the canonical 0x7fc00000 either
    way, Wormhole "might or might not" set the sign either way. So for the four arithmetic ops the
    sign is never the operand's, including at `sub(NaN, 1)`.

    binary_max_min is the counter-case and the reason this is not a blanket relaxation: a bare
    SFPSWAP(VEC_MIN_MAX) *selects* an operand, so the NaN it returns is the datum it was handed and
    its sign is real on both arches.

    A genuine infinity is never relaxed on either path -- `0 - (-inf)` is `+inf` by IEEE, and that
    is the assertion the per-lane mask exists to keep.

    The exclusion list is deliberately the *selecting* ops rather than an allowlist of the
    arithmetic ones: every op that is not max/min computes its result, including the compositions
    whose NaN arrives via a reciprocal or an exp.
    """
    nan = float("nan")
    canonicalise = BinarySFPUGolden._canonicalise_emitted_nan

    # The composition ops are in this list deliberately. Excluding them is what broke the
    # Wormhole gate for div/fmod/remainder once, on a Blackhole-green branch: their NaN is as
    # computed as add's, and the xfails that used to cover them were deleted on the premise that
    # the gate would excuse the sign.
    for op in (
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwrsub,
        MathOperation.SfpuElwdiv,
        MathOperation.SfpuBinaryFmod,
        MathOperation.SfpuBinaryRemainder,
        MathOperation.SfpuXlogy,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuAtan2,
    ):
        assert op not in BinarySFPUGolden._NAN_SIGN_SELECTED_OPS, (
            f"{op.name} computes its NaN through the datapath, so its sign is the ISA's to "
            "choose and must be relaxed"
        )
        # A forwarded NaN is relaxed too, which is the whole of this fix: a negative one must not
        # come back out of the golden carrying the operand's sign.
        row = torch.tensor([-nan, nan, float("inf"), -float("inf"), 1.0])
        out, mask = canonicalise(op, row)
        assert mask.tolist() == [True, True, False, False, False], (
            f"{op.name}: the mask must be every NaN lane and no infinity lane, got "
            f"{mask.tolist()}"
        )
        assert not math.copysign(1.0, float(out[0])) < 0.0, (
            f"{op.name}: a -NaN must be canonicalised, or the golden exports a sign the ISA "
            "declines to promise"
        )
        assert float(out[2]) == float("inf") and float(out[3]) == -float("inf"), (
            f"{op.name}: infinities must pass through untouched -- their sign is IEEE's and "
            "stays asserted"
        )

    for op in (MathOperation.SfpuBinaryMax, MathOperation.SfpuBinaryMin):
        assert op in BinarySFPUGolden._NAN_SIGN_SELECTED_OPS, (
            f"{op.name} is a bare SFPSWAP that selects an operand, so its NaN sign is a real "
            "datum -- see _reduce_extremum for the same argument on the reduce path"
        )
        row = torch.tensor([-nan, nan, 1.0])
        out, mask = canonicalise(op, row)
        assert not mask.any(), f"{op.name} must relax nothing"
        assert (
            math.copysign(1.0, float(out[0])) < 0.0
        ), f"{op.name}: a selected -NaN must keep its sign"


def test_reduce_nan_sign_is_relaxed_only_for_the_accumulating_pools():
    """Sum and Average emit their NaN through SFPMAD; Max and Min select a lane.

    The golden canonicalises only the first pair, and `test_float_reduce_specials` relaxes the
    comparison only for the same pair. Both read `_SFPMAD_REDUCE_POOLS`, so they cannot drift.

    The case that motivated it: `torch.sum` over a column holding `+inf` and `-inf` returns a
    *negatively* signed NaN, which the pack substitution turns into `-inf` -- where Blackhole's
    SFPMAD emits the canonical 0x7fc00000 and packs `+inf`.
    """
    pools = UnarySFPUGolden._SFPMAD_REDUCE_POOLS
    assert set(pools) == {ReducePool.Sum, ReducePool.Average}
    assert ReducePool.Max not in pools and ReducePool.Min not in pools, (
        "Max/Min are a bare SFPSWAP(VEC_MIN_MAX) that selects a lane, so the NaN they return "
        "is the datum they picked and its sign stays asserted"
    )

    # torch really does hand back a negative NaN here, which is what makes this load-bearing.
    column = torch.tensor([[float("inf")], [-float("inf")]])
    host_nan = torch.sum(column)
    assert math.copysign(1.0, float(host_nan)) < 0.0, (
        "torch.sum over (+inf, -inf) no longer returns a negative NaN; the canonicalisation is "
        "still correct but this test no longer demonstrates why"
    )

    model = UnarySFPUGolden._model_reduce_dest_and_pack
    for pool, want_positive in (
        (ReducePool.Sum, True),
        (ReducePool.Average, True),
        (ReducePool.Max, False),
        (ReducePool.Min, False),
    ):
        out = model(
            torch.tensor([-float("nan")]),
            DataFormat.Float32,
            DataFormat.Float32,
            DestAccumulation.Yes,
            pool,
        )
        got_positive = not math.copysign(1.0, float(out[0])) < 0.0
        assert got_positive == want_positive, (
            f"{pool}: expected the sign to be "
            f"{'canonicalised' if want_positive else 'preserved'}"
        )
