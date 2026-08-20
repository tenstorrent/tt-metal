# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the sfpu_domains gates that decide what a sweep may inject.

No kernel, no device: these are pure-Python assertions about metadata. They are here
because specials_safe() is a *measured* matrix — 250 hardware variants reduced to a
handful of rules (see the section comment in sfpu_domains) — and until now nothing
executed it. Both production callers short-circuit on SPECIALS_READY_OPS, which is empty,
so the rules and the enum-normalisation trap underneath them could be rewritten without a
single test changing outcome. The measurement is expensive to redo and cheap to pin, so it
is pinned here.

The second half guards probe *spacing*, which has the same shape of problem: a probe that
is silently quantized back onto the boundary it was meant to straddle still reads as
coverage, and no hardware run reports it — the variant passes, having tested the boundary
twice.
"""

import math
import struct

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    ReducePool,
)
from helpers.sfpu_domains import (
    GENERATED_NAN_SIGN_OPS,
    Operand,
    edge_values,
    for_op,
    generated_nan_sign_is_asserted,
    nan_sign_is_unspecified,
    nan_survives_to_l1,
    ops_with_singularity,
    probe_spacing_format,
    sfpu_unary_ops,
    specials_safe,
    specials_safe_formats,
)

# The formats the measurement covered: the 5x5 matrix driven over the isinf / isposinf /
# isneginf / isnan / isfinite predicates on Wormhole n150. Integer and Fp8/MX *output*
# formats are deliberately absent — the measurement never covered them and the gate makes
# no claim there, so pinning a verdict for them would invent one.
_MEASURED_FORMATS = [
    DataFormat.Float32,
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
]

# The whole accepted set, written out. Seven cells of fifty, and each one is a rule:
#
#   * A Float32 input carries specials at either dest_acc, into any non-block output —
#     Float16 included, but only with the 32-bit dest (breaker 1, output side).
#   * A Float16_b input carries them only at dest_acc=No; the bf16 -> fp32 dest unpack
#     loses -inf and NaN (breaker 2), and cannot reach a Float16 output at all.
#   * A Float16 input never carries them, at any dest_acc, into any output (breaker 1).
#   * A block-float input cannot carry them in the first place, and a block-float output
#     cannot express the result.
_ACCEPTED = frozenset(
    {
        (DataFormat.Float32, DataFormat.Float32, False),
        (DataFormat.Float32, DataFormat.Float16_b, False),
        (DataFormat.Float32, DataFormat.Float32, True),
        (DataFormat.Float32, DataFormat.Float16, True),
        (DataFormat.Float32, DataFormat.Float16_b, True),
        (DataFormat.Float16_b, DataFormat.Float32, False),
        (DataFormat.Float16_b, DataFormat.Float16_b, False),
    }
)

_MEASURED_MATRIX = [
    (inp, out, dest_acc)
    for inp in _MEASURED_FORMATS
    for out in _MEASURED_FORMATS
    for dest_acc in (False, True)
]


@pytest.mark.parametrize(
    "input_format,output_format,dest_acc",
    _MEASURED_MATRIX,
    ids=[
        f"{inp.name}-{out.name}-dest_acc_{'Yes' if d else 'No'}"
        for inp, out, d in _MEASURED_MATRIX
    ],
)
def test_specials_safe_matches_measured_matrix(input_format, output_format, dest_acc):
    """Every cell of the measured matrix, against the checked-in verdict."""
    expected = (input_format, output_format, dest_acc) in _ACCEPTED
    assert specials_safe(input_format, output_format, dest_acc) is expected, (
        f"specials_safe({input_format.name}, {output_format.name}, dest_acc={dest_acc}) "
        f"changed: expected {expected}. Either a rule moved or the measurement did — if "
        f"the latter, re-measure with the predicate sweep described in sfpu_domains and "
        f"update _ACCEPTED with the new verdict."
    )


def test_specials_safe_accepts_nothing_outside_the_measured_matrix():
    """The gate defaults to deny, so an unlisted input format is False rather than a
    wall of failures with one root cause."""
    for unlisted in (
        DataFormat.Tf32,
        DataFormat.Int32,
        DataFormat.UInt32,
        DataFormat.Fp8_e4m3,
        DataFormat.MxFp8P,
        DataFormat.Bfp2_b,
    ):
        for dest_acc in (False, True):
            assert not specials_safe(unlisted, DataFormat.Float32, dest_acc)


def test_specials_safe_rejects_mx_outputs():
    """MX outputs are excluded statically, not by measurement: an inf/NaN inside a block
    whose shared exponent is finite is not a value the format can express."""
    for mx_output in (DataFormat.MxFp8P, DataFormat.MxFp8R, DataFormat.MxFp4):
        assert not specials_safe(DataFormat.Float32, mx_output, True)


# The {Float16_b, Float32} matrix the two edge sweeps parametrize over, as (input, output).
_EDGE_SWEEP_PAIRS = [
    (DataFormat.Float16_b, DataFormat.Float16_b),
    (DataFormat.Float16_b, DataFormat.Float32),
    (DataFormat.Float32, DataFormat.Float16_b),
    (DataFormat.Float32, DataFormat.Float32),
]

_EDGE_SWEEP_CELLS = [
    (inp, out, dest_acc)
    for inp, out in _EDGE_SWEEP_PAIRS
    for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
]


def test_nan_survives_only_into_a_32_bit_dest_and_a_32_bit_pack():
    """Both legs have to stay 32-bit, and on this matrix exactly one cell manages it.

    Pinned because the gate's whole scope follows from it: five of the six triples
    specials_safe() accepts narrow a NaN somewhere, and those five are precisely where a
    generated NaN's sign becomes an observable +/-inf.
    """
    carrying = [c for c in _EDGE_SWEEP_CELLS if specials_safe(*c)]
    assert len(carrying) == 6, "specials_safe's verdict on this matrix moved"

    survives = [c for c in carrying if nan_survives_to_l1(*c)]
    assert survives == [
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes)
    ], (
        "Float32->Float32 at dest_acc=Yes is the only cell on this matrix that carries a "
        f"NaN to L1 as a NaN; got {[(i.name, o.name, str(d)) for i, o, d in survives]}. "
        "If this moved, the Wormhole NaN-sign skip's scope moved with it."
    )


# The two suites do not share a format axis, so the gate's reach is counted per suite.
# ScalarRsub is the scalar suite's only member of the set; the other ten are unary.
_SCALAR_NAN_SIGN_OPS = frozenset({MathOperation.ScalarRsub})
_UNARY_NAN_SIGN_OPS = GENERATED_NAN_SIGN_OPS - _SCALAR_NAN_SIGN_OPS

# test_sfpu_binop_scalar's axis after _skip_unsupported: a Float32 tensor needs the 32-bit
# dest and a Float16_b tensor cannot use one, so two of its eight cells survive.
_SCALAR_SUITE_CELLS = [
    (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
]


def test_nan_sign_gate_matches_the_measured_wormhole_failures():
    """Pinned against the Wormhole n300 run rather than against the rule that generated it.

    Unary: 50 (op, cell) pairs — the 49 recorded failures plus the one Rsqrt cell that was
    already xfailed for the unrelated -0.0 divergence, and which this gate now skips first.
    Scalar: 1, which is the run's single `ScalarRsub` failure.
    """
    unary_gated = [
        (op, cell)
        for op in _UNARY_NAN_SIGN_OPS
        for cell in _EDGE_SWEEP_CELLS
        if specials_safe(*cell) and nan_sign_is_unspecified(op, *cell)
    ]
    assert len(_UNARY_NAN_SIGN_OPS) == 10
    assert (
        len(unary_gated) == 50
    ), f"expected 50 gated unary (op, cell) pairs, got {len(unary_gated)}"

    scalar_gated = [
        (op, cell)
        for op in _SCALAR_NAN_SIGN_OPS
        for cell in _SCALAR_SUITE_CELLS
        if specials_safe(*cell) and nan_sign_is_unspecified(op, *cell)
    ]
    assert scalar_gated == [
        (MathOperation.ScalarRsub, _SCALAR_SUITE_CELLS[0])
    ], f"expected only ScalarRsub's Float16_b cell, got {scalar_gated}"

    # Every enrolled op keeps the one cell where the sign is readable, so none of them is
    # skipped out of its sweep entirely.
    for op in GENERATED_NAN_SIGN_OPS:
        assert not nan_sign_is_unspecified(
            op, DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes
        ), f"{op.name} lost its Float32->Float32 dest_acc=Yes assertion"


def test_generated_nan_sign_gate_is_the_narrowing_cells_on_wormhole_only():
    """The binary family's gate: no op argument, and exactly the cells that narrow.

    Where the unary gate needs a measured op set (a kernel either invents its NaN or forwards
    one, and nothing about the format axis predicts which), the binary edge sweep already knows
    per class -- so this gate is the pipeline half alone. Pinned so that stays true: an op
    argument appearing here would mean the two gates had been conflated.
    """
    for cell in _EDGE_SWEEP_CELLS:
        if not specials_safe(*cell):
            continue
        narrows = not nan_survives_to_l1(*cell)
        assert generated_nan_sign_is_asserted(*cell, on_wormhole=True) == narrows, (
            f"{cell} disagrees with nan_survives_to_l1 on Wormhole -- the gate must be "
            "exactly the narrowing cells"
        )
        assert not generated_nan_sign_is_asserted(*cell, on_wormhole=False), (
            f"{cell} is gated on Blackhole, where SFPMAD promises the canonical 0x7fc00000 "
            "and the sign is therefore assertable"
        )


def test_binary_golden_dest_format_matches_the_domains_rule():
    """BinarySFPUGolden._dest_format and nan_survives_to_l1 must derive the same Dest.

    Three places state this rule -- the unary golden, the binary golden, and sfpu_domains'
    internal derivation -- and a silent disagreement would put the golden's NaN substitution on
    a different set of cells than the gate that decides where the probe may be asserted. Checked
    rather than commented, since the two live in different modules.
    """
    from helpers.golden_generators import BinarySFPUGolden

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
    import torch
    from helpers.golden_generators import BinarySFPUGolden

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
    import torch
    from helpers.golden_generators import BinarySFPUGolden, sfpu_max, sfpu_min

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
    """The BinaryOp enumerator names declared for one architecture.

    Parsed from ckernel_defs.h rather than mirrored here, because a copy of the enum in this file
    would keep agreeing with itself after the header changed -- which is the exact failure this
    guard exists to prevent.
    """
    import re
    from pathlib import Path

    header = (
        Path(__file__).resolve().parents[2]
        / arch_dir
        / "common"
        / "inc"
        / "ckernel_defs.h"
    )
    assert header.is_file(), (
        f"{header} not found. This guard pins coverage-audit section 4.5 against the arch "
        "headers; if the layout moved, repoint it rather than deleting it."
    )
    text = header.read_text()
    body = re.search(r"enum class BinaryOp\s*:\s*[^{]*\{(.*?)\}", text, re.S)
    assert body, f"no `enum class BinaryOp` found in {header}"
    return set(re.findall(r"^\s*([A-Z][A-Z0-9_]*)\s*(?:=|,)", body.group(1), re.M))


# Listed in the coverage audit as driven by "none (WH/BH)", which is true of the enum *members*
# and false of the *kernels*: these carry MathOpType.SFPU_BINARY_INT, which only the Quasar
# dispatch header implements, while the same kernels are reached on WH/BH through the SFPU_BINARY
# members below at DataFormat.Int32. Same aliasing hazard the audit records for SfpuWhere/TTNNWhere
# and LogicalNot/LogicalNotUnary, one enum apart.
_QUASAR_INT_BINARY_ALIASES = {
    MathOperation.SfpuGtInt: MathOperation.SfpuElwGt,
    MathOperation.SfpuLtInt: MathOperation.SfpuElwLt,
    MathOperation.SfpuLeInt: MathOperation.SfpuElwLe,
    MathOperation.SfpuGeInt: MathOperation.SfpuElwGe,
    # The int multiply is spelled MUL on Quasar and reaches _mul_int32_; on WH/BH the same kernel
    # is MUL_INT32, which SfpuMulInt32 drives (test_sfpu_binary_int_uniform).
    MathOperation.SfpuElwmulInt: MathOperation.SfpuMulInt32,
}


@pytest.mark.parametrize("arch_dir", ["tt_llk_wormhole_b0", "tt_llk_blackhole"])
def test_quasar_int_binary_members_alias_covered_kernels(arch_dir):
    """The five SFPU_BINARY_INT members are unreachable on WH/BH; their kernels are not.

    Two halves, and both matter. If the first fails, one of these members became dispatchable and
    is now genuinely untested -- give it a test. If the second fails, the alias it was relying on
    stopped being driven, and the kernel lost its only WH/BH coverage while the audit still
    recorded it as covered by proxy. Either way the audit's section 4.5 needs rewriting, which is
    why this asserts the shape rather than describing it.
    """
    declared = _binary_op_enumerators(arch_dir)

    for member, alias in _QUASAR_INT_BINARY_ALIASES.items():
        spec = member.value
        if member is MathOperation.SfpuElwmulInt:
            # "MUL" *is* declared -- it is the float multiply. What matters is that this member
            # cannot be dispatched here, which its MathOpType decides, not its spelling.
            assert spec.operation_type.name == "SFPU_BINARY_INT"
        else:
            assert spec.cpp_enum_value not in declared, (
                f"{member.name} names BinaryOp::{spec.cpp_enum_value}, which is now declared "
                f"in {arch_dir}. It is reachable on this arch and needs a test of its own; the "
                "audit's section 4.5 alias note no longer covers it."
            )

        assert alias.value.cpp_enum_value in declared, (
            f"{member.name}'s kernel is covered on WH/BH only through {alias.name}, whose "
            f"BinaryOp::{alias.value.cpp_enum_value} is no longer declared in {arch_dir}"
        )


def test_int_comparison_aliases_are_driven_at_int32():
    """The aliasing claim above is only worth anything while the alias is actually driven at Int32.

    Checked against the test module's own list so the two cannot drift: if the ordered comparisons
    stop being driven on an integer format, the four Quasar members lose their proxy coverage
    silently.
    """
    import test_sfpu_binary

    driven = set(test_sfpu_binary._INT_COMPARISON_OPS)
    expected = {
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwGe,
    }
    assert driven == expected, (
        "_INT_COMPARISON_OPS no longer holds the four ordered comparisons, so the Int32 "
        f"comparison kernel's coverage moved: {driven ^ expected}"
    )
    assert (
        set(_QUASAR_INT_BINARY_ALIASES.values()) - {MathOperation.SfpuMulInt32}
        == driven
    ), "the alias table and the driven set disagree"


def test_every_float_binary_op_is_classified_for_cat_b():
    """Enrolled or recorded-as-not-ready, for every float op the binary sweep can drive.

    Totality, in the same spirit as test_sfpu_binary's three stimulus-source sets: an op that is in
    neither dict keeps cat B switched off while looking, to a reader, as though it had been
    considered. The count is not pinned -- only the partition -- so adding a binary op is a
    one-line decision rather than a test edit.
    """
    import test_sfpu_binary
    from helpers.sfpu_domains import (
        _BINARY_SPECIALS_NOT_READY,
        BINARY_SPECIALS_READY_OPS,
    )

    candidates = (
        test_sfpu_binary._CLASSIFIED_STIMULI_OPS
        - test_sfpu_binary._INT_DRIVEN_BINARY_OPS
    )
    classified = set(BINARY_SPECIALS_READY_OPS) | set(_BINARY_SPECIALS_NOT_READY)
    unclassified = sorted(op.name for op in candidates - classified)
    assert not unclassified, (
        "these float binary ops reach sfpu_binary() but appear in neither "
        "BINARY_SPECIALS_READY_OPS nor _BINARY_SPECIALS_NOT_READY, so nothing records whether "
        f"cat B is off for them by decision or by omission: {unclassified}"
    )
    stale = sorted(op.name for op in classified - candidates)
    assert (
        not stale
    ), f"these ops carry a cat-B verdict but no longer reach the binary driver: {stale}"

    for op, reason in BINARY_SPECIALS_READY_OPS.items():
        assert len(reason) > 20, f"{op.name}'s cat-B reason is too short to be a claim"


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
    import torch
    from helpers.golden_generators import (
        sfpu_max,
        sfpu_min,
        sfpu_order_key_elementwise,
        sfpu_total_order_key,
    )

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


def test_dest_truncation_masks_match_the_mantissa_width_table():
    """The 32 -> 16-bit Dest masks must be derived, not written twice.

    Both goldens truncate a 32-bit operand on the way into a 16-bit Dest, and the mask used to be
    an unnamed literal at each site. `dest_truncation_mask` derives it from
    `_FORMAT_MANTISSA_BITS`; these are the values those literals had.
    """
    from helpers.sfpu_domains import dest_truncation_mask

    assert dest_truncation_mask(DataFormat.Float16) == 0xFFFFE000
    assert dest_truncation_mask(DataFormat.Float16_b) == 0xFFFF0000


def test_exp_with_base_ceiling_is_currently_unreachable():
    """ExpWithBase's `_APPROX_ACCURACY_MAX` entry is correct but never fires today.

    The op sits in STANDARD_SWEEP_OPS, which drives ApproximationMode.No only, so the swept domain
    is the range bound (high=160, exp argument 80) and the 32.0 ceiling is inert. The entry is kept
    so that enrolling the op in BROAD_SWEEP_OPS cannot silently hand the approximation an argument
    of 80 -- ten times the ~8 where its overshoot starts.

    If this fails because ExpWithBase joined the broad profile, that is the good outcome: the
    ceiling now fires, and what wants re-checking is the accurate path's own domain.
    """
    import test_sfpu_unary

    assert MathOperation.ExpWithBase in test_sfpu_unary.STANDARD_SWEEP_OPS
    assert MathOperation.ExpWithBase not in test_sfpu_unary.BROAD_SWEEP_OPS
    accurate = for_op(
        MathOperation.ExpWithBase,
        DataFormat.Float32,
        approx_mode=ApproximationMode.No,
    ).spec_A
    assert accurate.high == 160.0, (
        "the accurate path's ceiling moved; it is measured green at 160 on a Wormhole n300 and "
        "has no custom tolerance, so a change here wants a re-run"
    )


def test_hardtanh_golden_matches_the_hardtanh_kernel_chain():
    """_hardtanh models a clamp, but its kernel is not one -- pin the agreement.

    `SfpuType::hardtanh` dispatches to `_calculate_hardtanh_`, three adds with two clamps-at-zero
    and bf16 constants, where Clamp dispatches to `_calculate_clamp_` with fp16 min/max. They
    agree on the finite range and at every special this op is enrolled for, but by arithmetic
    rather than by sharing code, so the golden's use of sfpu_clamp is only sound while that holds.
    """
    from helpers.golden_generators import UnarySFPUGolden, sfpu_total_order_key

    def kernel_chain(x: float, low: float, high: float) -> float:
        # val += p0; v_if (val < 0) val = 0; val += p1; v_if (val >= 0) val = 0; val += p2
        p0, p1, p2 = -low, -(high - low), high
        val = x + p0
        if sfpu_total_order_key(val) < 0:
            val = 0.0
        val = val + p1
        if sfpu_total_order_key(val) >= 0:
            val = 0.0
        return val + p2

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
        want = kernel_chain(x, low, high)
        got = float(golden._hardtanh(x, low, high))
        assert got == want or (got != got and want != want), (
            f"hardtanh({x}) golden gives {got} but the kernel chain gives {want}. The golden "
            "models _calculate_clamp_; if the two have stopped agreeing, model the chain."
        )


def test_reduce_extremum_follows_the_total_order_on_floats_only():
    """Reduce MAX/MIN fold under the SFPU total order; Sum/Average stay IEEE; ints stay torch.

    ckernel_sfpu_reduce.h reduces MAX/MIN with a bare TTI_SFPSWAP(VEC_MIN_MAX) and no NaN guard, so
    unlike the six binary comparisons the order does reach the result. MIN is the load-bearing case:
    a column holding one +NaN must reduce to the *finite* minimum, where torch.min propagates.

    Integers are checked in the same test because ReduceColumn/ReduceRow are deliberately *not*
    routed through _call_integer, so an Int32 reduce reaches the same helper -- and there the model
    is _emit_int32_signed_cswap_, i.e. plain two's complement.
    """
    import torch
    from helpers.golden_generators import UnarySFPUGolden

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


def test_nan_sign_gate_ignores_ops_that_forward_a_nan():
    """Neg, Abs and Identity move the sign bit rather than inventing one, so their NaN sign
    is a real datum and stays asserted on every cell. They are UnarySFPUGolden's
    _NAN_SIGN_TRANSPARENT_OPS, and this gate must never overlap that set."""
    for op in (MathOperation.Neg, MathOperation.Abs, MathOperation.Identity):
        assert op not in GENERATED_NAN_SIGN_OPS
        for cell in _EDGE_SWEEP_CELLS:
            assert not nan_sign_is_unspecified(op, *cell)


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
    import torch
    from helpers.golden_generators import BinarySFPUGolden

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
    import torch
    from helpers.golden_generators import UnarySFPUGolden

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


@pytest.mark.parametrize(
    "dest_acc,flag",
    [
        (True, True),
        (False, False),
        (DestAccumulation.Yes, True),
        (DestAccumulation.No, False),
    ],
    ids=["bool_True", "bool_False", "DestAccumulation_Yes", "DestAccumulation_No"],
)
def test_dest_acc_normalisation(dest_acc, flag):
    """A DestAccumulation member must read as its value, not as its truthiness.

    Both members are truthy objects, so a `bool(dest_acc)` implementation would evaluate
    DestAccumulation.No as the 32-bit-dest case and silently flip whole rows of the
    matrix. Float16_b -> Float16_b is the probe because its verdict inverts with the flag:
    accepted at dest_acc=No, rejected at dest_acc=Yes (breaker 2). So a member that
    normalised to the wrong bool would fail here rather than pass by coincidence.
    """
    assert specials_safe(DataFormat.Float16_b, DataFormat.Float16_b, dest_acc) is (
        not flag
    )


@pytest.mark.parametrize(
    "bad", [1, 0, "Yes", None, 1.0], ids=["int_1", "int_0", "str", "None", "float"]
)
def test_dest_acc_rejects_non_flags(bad):
    """Anything that is neither a bool nor a DestAccumulation raises rather than being
    guessed at. 1 and 0 included: they would work by accident and hide a caller bug."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        specials_safe(DataFormat.Float32, DataFormat.Float32, bad)


def test_specials_safe_formats_filters_to_the_accepted_rows():
    formats = [
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float16),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float32),
    ]

    kept = specials_safe_formats(formats, DestAccumulation.No)
    assert [(f.input_format, f.output_format) for f in kept] == [
        (DataFormat.Float32, DataFormat.Float32),
        (DataFormat.Float16_b, DataFormat.Float16_b),
    ]

    kept = specials_safe_formats(formats, DestAccumulation.Yes)
    assert [(f.input_format, f.output_format) for f in kept] == [
        (DataFormat.Float32, DataFormat.Float32),
        (DataFormat.Float32, DataFormat.Float16),
    ]


def test_specials_safe_formats_validates_dest_acc_on_an_empty_list():
    """Normalisation happens once up front, so a bad flag raises even with nothing to
    filter — otherwise the error surfaces only for callers that happen to pass formats.
    """
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        specials_safe_formats([], "Yes")


# ─────────────────────────────────────────────────────────────────────────────
# Probe spacing: every edge probe has to still be distinct once the datapath has it
#
# The failure this pins is not a wrong answer, it is a probe that stops probing. With
# dest_acc=No the DEST holds 16 bits whatever the input format is, so an fp32 probe one fp32
# ULP above a pole of 1.0 (0x3F800002) is truncated straight back to 1.0 and the variant
# tests the pole twice while reading as "pole plus one above it". Acosh was the op it hit.
#
# The truncation is modelled here as an independent `& 0xFFFF0000` rather than by importing
# sfpu_domains' own helper, so these assert the *property* and not the implementation.
# ─────────────────────────────────────────────────────────────────────────────

# The format axis test_eltwise_unary_sfpu_edges actually collects.
_EDGE_SWEEP_FORMATS = [
    (inp, out)
    for inp in (DataFormat.Float16_b, DataFormat.Float32)
    for out in (DataFormat.Float16_b, DataFormat.Float32)
]

# Everything edge_values() can be asked about: the ops with a registry entry (which is what
# sfpu_unary_ops() gates the sweep on) plus the ops carrying a singularity on either operand.
_PROBED_OPS = sorted(
    set(sfpu_unary_ops()) | set(ops_with_singularity()), key=lambda op: op.name
)


def _as_16bit_dest(value: float) -> float:
    """*value* as a 16-bit DEST holds it: fp32 with the low mantissa half dropped."""
    if value != value or value in (float("inf"), float("-inf")):
        return value
    raw = struct.unpack("<I", struct.pack("<f", value))[0] & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def _distinct_key(value: float):
    """Identity of *value* as delivered, keeping the two zeros apart.

    -0.0 == +0.0 and they are zero ULPs apart, so a plain set() would collapse them — and
    for signbit / sign / heaviside the difference between them is the entire probe.
    """
    delivered = _as_16bit_dest(value)
    if delivered == 0.0:
        return ("zero", struct.pack("<f", delivered)[-1] & 0x80)
    return ("value", delivered)


@pytest.mark.parametrize(
    "input_format,output_format",
    _EDGE_SWEEP_FORMATS,
    ids=[f"{i.name}-{o.name}" for i, o in _EDGE_SWEEP_FORMATS],
)
@pytest.mark.parametrize(
    "operand", [Operand.A, Operand.B], ids=["operand_A", "operand_B"]
)
def test_edge_probes_stay_distinct_in_a_16bit_dest(
    input_format, output_format, operand
):
    """No op's probe list may contain two values a 16-bit DEST cannot tell apart.

    Parametrized over the formats and swept over every op rather than pinning Acosh, so an
    op that *gains* a nonzero singularity later is covered without touching this test —
    which is the failure mode that produced the original: the collapse was a property of
    (boundary magnitude, format pair, dest_acc), not of Acosh.
    """
    for op in _PROBED_OPS:
        probes = edge_values(
            op, input_format, output_format, operand, dest_acc=DestAccumulation.No
        )
        delivered = [_distinct_key(v) for v in probes]
        assert len(set(delivered)) == len(probes), (
            f"{op.name} probes {probes} collapse in a 16-bit DEST: they arrive as "
            f"{[_as_16bit_dest(v) for v in probes]}. A probe that quantizes onto its own "
            f"boundary tests the boundary twice while reading as coverage — widen the step "
            f"via probe_beside()/probe_spacing_format()."
        )


@pytest.mark.parametrize(
    "fmt,dest_acc,expected",
    [
        (DataFormat.Float32, DestAccumulation.No, DataFormat.Float16_b),
        (DataFormat.Float32, DestAccumulation.Yes, DataFormat.Float32),
        (DataFormat.Float16_b, DestAccumulation.No, DataFormat.Float16_b),
        (DataFormat.Float16_b, DestAccumulation.Yes, DataFormat.Float16_b),
        (DataFormat.Float16, DestAccumulation.No, DataFormat.Float16),
        (DataFormat.Bfp4_b, DestAccumulation.No, DataFormat.Bfp4_b),
        (DataFormat.Float32, None, DataFormat.Float32),
        # Integer formats are 32-bit but mantissa truncation is not what their DEST does.
        (DataFormat.Int32, DestAccumulation.No, DataFormat.Int32),
        (DataFormat.UInt32, DestAccumulation.No, DataFormat.UInt32),
    ],
    ids=[
        "fp32_destaccNo_coarsens",
        "fp32_destaccYes_keeps",
        "bf16_destaccNo_keeps",
        "bf16_destaccYes_keeps",
        "fp16_destaccNo_keeps",
        "bfp4_destaccNo_keeps",
        "no_dest_acc_keeps",
        "int32_destaccNo_keeps",
        "uint32_destaccNo_keeps",
    ],
)
def test_probe_spacing_format(fmt, dest_acc, expected):
    """Only a 32-bit format is coarsened, and only at dest_acc=No.

    A format that is already 16-bit or narrower keeps its own spacing: the DEST it lands in
    is no coarser than it is, so coarsening would loosen the probe for nothing. dest_acc=None
    keeps the format-only behaviour for callers with no flag to pass.
    """
    assert probe_spacing_format(fmt, dest_acc) is expected


def test_probe_spacing_format_rejects_non_flags():
    """Same normalisation trap as specials_safe(): a bare int would work by accident."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        probe_spacing_format(DataFormat.Float32, 1)


def test_acosh_above_pole_probe_is_widened_at_dest_acc_no():
    """The measured case, pinned with its actual values.

    Acosh is (1.0, ABOVE) and (Float32, Float16_b) ties on the bfloat16 ceiling and resolves
    to Float32, so the fp32 step is 2*2**-23 and the probe is 0x3F800002 — which arrives as
    1.0. The bfloat16 step is 2*2**-7, so 1.015625 arrives intact.
    """
    for output_format in (DataFormat.Float32, DataFormat.Float16_b):
        assert edge_values(
            MathOperation.Acosh,
            DataFormat.Float32,
            output_format,
            dest_acc=DestAccumulation.No,
        ) == [1.0, 1.015625]

        # dest_acc=Yes keeps the full fp32 dest, so the tight probe stands.
        assert edge_values(
            MathOperation.Acosh,
            DataFormat.Float32,
            output_format,
            dest_acc=DestAccumulation.Yes,
        ) == [1.0, 1.0000002384185791]


@pytest.mark.parametrize(
    "op",
    [MathOperation.Asin, MathOperation.Acos, MathOperation.Atanh, MathOperation.Erfinv],
    ids=lambda op: op.name,
)
def test_below_boundary_probes_are_not_loosened(op):
    """The other ±1 ops keep their tight below-1.0 probes at dest_acc=No.

    Stepping *down* from 1.0 crosses into the next binade (0x3F7FFFFF -> 0.99609375), so it
    survives the same truncation that destroys the step upward. Coarsening per boundary
    rather than per side would loosen these four for no gain in distinctness, which is why
    probe_beside() decides one side at a time.
    """
    probes = edge_values(
        op, DataFormat.Float32, DataFormat.Float32, dest_acc=DestAccumulation.No
    )
    # Two fp32 ULPs at 1.0, i.e. 0.9999997615814209 — not the two bfloat16 ULPs (0.984375)
    # a per-boundary coarsening would have produced.
    assert 1.0 - 2 * 2**-23 in probes
    assert -(1.0 - 2 * 2**-23) in probes


@pytest.mark.parametrize(
    "op",
    [MathOperation.Sqrt, MathOperation.Log, MathOperation.Reciprocal],
    ids=lambda op: op.name,
)
def test_zero_pole_probes_are_not_loosened(op):
    """Zero-poles keep the fp32 step too, and for a different reason than the ±1 ops.

    Truncation drops mantissa bits and keeps the exponent, and bfloat16 carries fp32's full
    8-bit exponent range, so 2**-23 is representable in a 16-bit DEST and stays a visible
    distance from zero. 17 ops sit on a zero pole; a blanket coarsening would loosen all of
    them.
    """
    probes = edge_values(
        op, DataFormat.Float32, DataFormat.Float32, dest_acc=DestAccumulation.No
    )
    assert min(abs(v) for v in probes if v != 0.0) == 2 * 2**-23


# ─────────────────────────────────────────────────────────────────────────────
# The exp family's two ceilings: range on the registry, accuracy behind approx mode
#
# One registry entry is consumed by both approximation modes, so an accuracy limit parked
# on it silently narrows the *accurate* path too. Exp lost (16, 80] that way — with it the
# exponent-overflow region and all large-exp saturation — and ExpWithBase lost 32..160 for a
# limit it never executes, since STANDARD_SWEEP_OPS runs ApproximationMode.No only.
#
# Nothing executed the split before these tests: the sweep asserts a tolerance, not a
# domain, so a bound could move in either direction without a single test changing outcome.
# ─────────────────────────────────────────────────────────────────────────────

# (op, range-bound high, approximation-accuracy high). The accuracy column is what the
# merged branch had on the shared entry; the range column is what the accurate path gets
# back. Wormhole-measured for the approximation (see _APPROX_EXP_ACCURACY_XFAIL).
_EXP_FAMILY_BOUNDS = [
    (MathOperation.Exp, 80.0, 16.0),
    (MathOperation.Exp2, 100.0, 23.0),
    (MathOperation.ExpWithBase, 160.0, 32.0),
]


@pytest.mark.parametrize(
    "op,range_high,approx_high", _EXP_FAMILY_BOUNDS, ids=lambda v: getattr(v, "name", v)
)
def test_exp_family_ceiling_depends_on_approximation_mode(op, range_high, approx_high):
    """The accurate path keeps the range bound; only the approximating path is narrowed."""
    assert (
        for_op(op, DataFormat.Float32, approx_mode=ApproximationMode.No).spec_A.high
        == range_high
    )
    assert (
        for_op(op, DataFormat.Float32, approx_mode=ApproximationMode.Yes).spec_A.high
        == approx_high
    )


@pytest.mark.parametrize(
    "op,range_high,approx_high", _EXP_FAMILY_BOUNDS, ids=lambda v: getattr(v, "name", v)
)
def test_exp_family_default_is_the_unnarrowed_domain(op, range_high, approx_high):
    """Omitting approx_mode applies no narrowing.

    That is the right default for a caller that measures error instead of asserting a
    tolerance (the accuracy harness), and it keeps the registry readable as "what the op and
    format can hold" rather than "what one mode happens to tolerate".
    """
    assert for_op(op, DataFormat.Float32).spec_A.high == range_high


def test_exp_with_base_argument_ceiling_matches_exp_in_both_modes():
    """exp_with_base computes exp(0.5*x), so its bound has to be exactly double exp's.

    This is the invariant its docstring claims, and the one that broke when the accuracy
    limit landed on the shared entry: exp went to 16 and exp_with_base to 32, but exp's
    *range* bound stayed 80 while exp_with_base's became 32 — an argument of 16 against
    exp's 80.
    """
    for mode in (ApproximationMode.No, ApproximationMode.Yes):
        exp_high = for_op(
            MathOperation.Exp, DataFormat.Float32, approx_mode=mode
        ).spec_A.high
        base_high = for_op(
            MathOperation.ExpWithBase, DataFormat.Float32, approx_mode=mode
        ).spec_A.high
        assert base_high == 2 * exp_high, (
            f"exp_with_base's argument ceiling is {0.5 * base_high} against exp's "
            f"{exp_high} in ApproximationMode.{mode.name}"
        )


@pytest.mark.parametrize(
    "fmt,high",
    [(DataFormat.MxFp8P, 5.0), (DataFormat.Fp8_e4m3, 5.0), (DataFormat.Float16, 10.0)],
    ids=lambda v: getattr(v, "name", v),
)
def test_narrow_format_exp_domains_are_not_widened_by_the_ceiling(fmt, high):
    """The ceiling only ever narrows, so a format branch already tighter is untouched."""
    for mode in (ApproximationMode.No, ApproximationMode.Yes, None):
        assert for_op(MathOperation.Exp, fmt, approx_mode=mode).spec_A.high == high


@pytest.mark.parametrize(
    "op",
    [
        MathOperation.Sqrt,
        MathOperation.Log,
        MathOperation.Reciprocal,
        MathOperation.Gelu,
    ],
    ids=lambda op: op.name,
)
def test_approx_mode_does_not_touch_ops_outside_the_table(op):
    """Only _APPROX_ACCURACY_MAX members react to the mode."""
    no = for_op(op, DataFormat.Float32, approx_mode=ApproximationMode.No).spec_A
    yes = for_op(op, DataFormat.Float32, approx_mode=ApproximationMode.Yes).spec_A
    assert (no.low, no.high, no.intervals) == (yes.low, yes.high, yes.intervals)


@pytest.mark.parametrize(
    "bad", [1, 0, "Yes", 1.0], ids=["int_1", "int_0", "str", "float"]
)
def test_for_op_rejects_non_flag_approx_mode(bad):
    """Same truthiness trap as dest_acc: ApproximationMode.No is a truthy object, so a
    bare int would work by accident on one branch and silently narrow on the other."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        for_op(MathOperation.Exp, DataFormat.Float32, approx_mode=bad)


def test_two_state_flag_rejects_the_other_two_state_enum():
    """A bool check alone cannot tell the two-state enums apart.

    DestAccumulation and ApproximationMode both wrap True/False, so passing one where the
    other is expected satisfies any `.value is a bool` test and selects a valid but
    unintended branch. That is the swap the guard exists to catch, and it is invisible in a
    result -- both arguments are legal, the answer is just quietly the wrong one.
    """
    for wrong, call in (
        (
            DestAccumulation.Yes,
            lambda v: for_op(MathOperation.Exp, DataFormat.Float32, approx_mode=v),
        ),
        (
            ApproximationMode.Yes,
            lambda v: specials_safe(DataFormat.Float32, DataFormat.Float32, v),
        ),
    ):
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            TypeError
        ):
            call(wrong)

    # The right enum, and a bare bool, still pass.
    for_op(MathOperation.Exp, DataFormat.Float32, approx_mode=ApproximationMode.Yes)
    specials_safe(DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes)
    specials_safe(DataFormat.Float32, DataFormat.Float32, True)
