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
    format_dict,
)
from helpers.sfpu_domains import (
    GENERATED_NAN_SIGN_OPS,
    Operand,
    _is_negative_zero,
    edge_pair_values,
    edge_values,
    extremes_safe,
    for_op,
    generated_nan_sign_is_asserted,
    nan_sign_is_unspecified,
    nan_survives_to_l1,
    narrowest_range_format,
    op_edge_points,
    ops_with_singularity,
    probe_spacing_format,
    sfpu_unary_ops,
    specials_safe,
    specials_safe_formats,
)
from helpers.stimuli_generator import StimuliSpec

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
    # is MUL_INT32, which SfpuMulInt32 drives (test_eltwise_binary_sfpu_int_uniform).
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
    import test_eltwise_binary_sfpu

    driven = set(test_eltwise_binary_sfpu._INT_COMPARISON_OPS)
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

    Totality, in the same spirit as test_eltwise_binary_sfpu's three stimulus-source sets: an op that is in
    neither dict keeps cat B switched off while looking, to a reader, as though it had been
    considered. The count is not pinned -- only the partition -- so adding a binary op is a
    one-line decision rather than a test edit.
    """
    import test_eltwise_binary_sfpu
    from helpers.sfpu_domains import (
        _BINARY_SPECIALS_NOT_READY,
        _SFPU_BINARY_OPS,
        BINARY_SPECIALS_READY_OPS,
    )

    # _SFPU_BINARY_OPS as well as the ops reaching sfpu_binary(): SfpuAddTopRow is a registered
    # float binary op that the shared driver does not carry, so it was escaping this check by
    # being in neither set. The int-driven ops still come out -- an integer op has no IEEE
    # specials to have a verdict about.
    candidates = (
        test_eltwise_binary_sfpu._CLASSIFIED_STIMULI_OPS | _SFPU_BINARY_OPS
    ) - test_eltwise_binary_sfpu._INT_DRIVEN_BINARY_OPS
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


def test_exact_at_zero_ops_all_contain_zero_in_their_domain():
    """Zero must be inside every enrolled op's registered domain, or the probe is out of bounds.

    This is what keeps the gamma family out without a second list. Digamma, Lgamma and Polygamma
    have poles at zero and domains starting at 0.1, 1.0 and 0.5, so adding one here would drive
    a value the kernel never promised anything about -- the same mistake the note above
    _OP_SINGULARITIES warns against for their poles. Asserted rather than trusted, because the
    membership tuple is hand-written and the domains are not.
    """
    from helpers.sfpu_domains import _EXACT_AT_ZERO_OPS, for_op

    out_of_domain = []
    for op in _EXACT_AT_ZERO_OPS:
        spec = for_op(op, DataFormat.Float32).spec_A
        if spec.intervals is not None or not (spec.low <= 0.0 <= spec.high):
            out_of_domain.append(
                f"{op.name} (domain {spec.intervals or [spec.low, spec.high]})"
            )
    assert not out_of_domain, (
        "these ops are enrolled for the exact-value-at-zero probe but zero is outside their "
        f"registered domain: {out_of_domain}"
    )


def test_exact_at_zero_probe_reaches_the_edge_sweep():
    """The enrolment has to actually produce a probe, not just sit in a tuple.

    op_edge_points() is what the edge sweep reads, and an op joins by appearing in
    _OP_EDGE_POINTS -- this checks the two are wired together rather than the tuple being
    declared and never merged in.
    """
    from helpers.sfpu_domains import _EXACT_AT_ZERO_OPS

    for op in _EXACT_AT_ZERO_OPS:
        knees = op_edge_points(op)
        assert 0.0 in knees, f"{op.name} has no zero probe: {knees}"


def test_every_sweepable_op_is_classified_for_cat_f():
    """Enrolled or recorded-as-not-ready, for every op a cat-F sweep can actually reach.

    The last of the four totality checks, and the one that took two tranches: the first was
    hand-picked as the ops that cannot overflow, the second measured the remaining 74 and split
    55 / 19. Scoped to what a sweep can reach, because an op with no sweep to be enrolled into
    is a different problem from an undecided one.
    """
    import test_eltwise_binary_sfpu as binary
    import test_eltwise_unary_sfpu as unary
    from helpers.sfpu_domains import _EXTREMES_NOT_READY, EXTREMES_READY_OPS

    # The saturation ops are covered by their own sweep rather than by EXTREMES_READY_OPS --
    # cat F has two halves, and a check that knew only about the enrolment half would demand a
    # verdict for ops that already have one.
    candidates = set(unary._EDGE_SWEEP_OPS)
    classified = (
        set(EXTREMES_READY_OPS)
        | set(_EXTREMES_NOT_READY)
        | set(unary._SATURATION_PROBES)
        | set(binary._BINARY_SATURATION_PAIRS)
    )
    unclassified = sorted(op.name for op in candidates - classified)
    assert not unclassified, (
        "these ops are reachable by the cat-F sweep but appear in neither EXTREMES_READY_OPS "
        f"nor _EXTREMES_NOT_READY: {unclassified}"
    )
    for op, reason in _EXTREMES_NOT_READY.items():
        assert len(reason) > 20, f"{op.name}'s cat-F reason is too short to be a claim"


def test_every_unary_op_is_classified_for_cat_b():
    """Enrolled or recorded-as-not-ready, for every unary op the sweep drives.

    The unary family had only the first half of this partition for as long as cat B has
    existed: 67 ops enrolled, 28 outside it, and nothing saying whether that was a decision or
    an omission. The binary and ternary families have had the check since they were written --
    this is the same one, and it is what stops a newly registered unary op quietly defaulting
    to "no specials" while looking, to a reader, as though it had been considered.
    """
    from helpers.sfpu_domains import (
        _UNARY_OPS_NOT_SWEPT,
        _UNARY_SPECIALS_NOT_READY,
        SPECIALS_READY_OPS,
        sfpu_unary_ops,
    )

    candidates = sfpu_unary_ops() - set(_UNARY_OPS_NOT_SWEPT)
    classified = set(SPECIALS_READY_OPS) | set(_UNARY_SPECIALS_NOT_READY)
    unclassified = sorted(op.name for op in candidates - classified)
    assert not unclassified, (
        "these unary ops are swept but appear in neither SPECIALS_READY_OPS nor "
        "_UNARY_SPECIALS_NOT_READY, so nothing records whether cat B is off for them by "
        f"decision or by omission: {unclassified}"
    )
    stale = sorted(op.name for op in set(_UNARY_SPECIALS_NOT_READY) - candidates)
    assert (
        not stale
    ), f"these ops carry a not-ready verdict but are no longer swept: {stale}"

    for op, reason in _UNARY_SPECIALS_NOT_READY.items():
        assert len(reason) > 20, f"{op.name}'s cat-B reason is too short to be a claim"


def test_every_ternary_op_is_classified_for_cat_b():
    """Enrolled or recorded-as-not-ready, for every op in the ternary family.

    Totality, the same check test_every_float_binary_op_is_classified_for_cat_b makes: an op in
    neither dict keeps cat B switched off while looking, to a reader, as though it had been
    considered. The candidate set is _SFPU_TERNARY_OPS itself rather than a list here, so adding
    a ternary op is a one-line decision and not a test edit.
    """
    from helpers.sfpu_domains import (
        _SFPU_TERNARY_OPS,
        _TERNARY_SPECIALS_NOT_READY,
        TERNARY_SPECIALS_READY_OPS,
    )

    classified = set(TERNARY_SPECIALS_READY_OPS) | set(_TERNARY_SPECIALS_NOT_READY)
    unclassified = sorted(op.name for op in _SFPU_TERNARY_OPS - classified)
    assert not unclassified, (
        "these ternary ops appear in neither TERNARY_SPECIALS_READY_OPS nor "
        "_TERNARY_SPECIALS_NOT_READY, so nothing records whether cat B is off for them by "
        f"decision or by omission: {unclassified}"
    )
    stale = sorted(op.name for op in classified - _SFPU_TERNARY_OPS)
    assert (
        not stale
    ), f"these ops carry a cat-B verdict but are not in the ternary family: {stale}"

    for op, reason in TERNARY_SPECIALS_READY_OPS.items():
        assert len(reason) > 20, f"{op.name}'s cat-B reason is too short to be a claim"


def test_ternary_golden_dest_format_matches_the_domains_rule():
    """The shared sfpu_dest_format() and nan_survives_to_l1 must derive the same Dest.

    The ternary half of test_binary_golden_dest_format_matches_the_domains_rule. Both goldens
    now delegate to one function, so this pins the *function* rather than a third copy of the
    rule -- and it is worth pinning separately because the ternary suite passes it the input
    format where the binary one passes data_format, and a transposed pair would put the NaN
    substitution on the wrong cells without failing anything else.
    """
    from helpers.golden_generators import sfpu_dest_format

    for input_format, output_format, dest_acc in _EDGE_SWEEP_CELLS:
        dst = sfpu_dest_format(input_format, output_format, dest_acc)
        preserves = (dst, output_format) in {
            (DataFormat.Float16, DataFormat.Float16),
            (DataFormat.Float32, DataFormat.Float16),
            (DataFormat.Float32, DataFormat.Float32),
        }
        assert preserves == nan_survives_to_l1(input_format, output_format, dest_acc), (
            f"{input_format.name}->{output_format.name} dest_acc={dest_acc}: "
            f"sfpu_dest_format derives Dest={dst.name}, which disagrees with "
            "nan_survives_to_l1"
        )


def test_ternary_goldens_require_their_format_arguments_together():
    """Supplying part of the Dest/pack contract is rejected rather than half-modelled.

    Same reasoning as test_binary_golden_requires_dest_acc_and_output_format_together: half the
    contract is worse than none of it, because the golden would then be wrong in a new way
    rather than in the documented old one.
    """
    import torch
    from helpers.golden_generators import TernarySFPUGolden, WhereGolden

    tile = torch.zeros(1024, dtype=torch.float32)

    for missing in ("input_format", "dest_acc"):
        kwargs = {
            "input_format": DataFormat.Float32,
            "dest_acc": DestAccumulation.Yes,
        }
        del kwargs[missing]
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            ValueError, match="must be supplied together"
        ):
            TernarySFPUGolden()(
                MathOperation.SfpuAddcmul,
                tile.clone(),
                tile.clone(),
                tile.clone(),
                0,
                DataFormat.Float32,
                **kwargs,
            )

    for missing in ("input_format", "output_format", "dest_acc"):
        kwargs = {
            "input_format": DataFormat.Float32,
            "output_format": DataFormat.Float32,
            "dest_acc": DestAccumulation.Yes,
        }
        del kwargs[missing]
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            ValueError, match="must be supplied together"
        ):
            WhereGolden()(tile.clone(), tile.clone(), tile.clone(), **kwargs)


def test_ternary_golden_substitutes_an_infinity_only_where_the_pack_narrows():
    """The NaN a ternary op emits reaches L1 as a NaN, or as an infinity, per the gate.

    This is the modelling that made the specials_in class assertable: without it, a probe on a
    narrowing pipeline read the packer's substituted infinity as the kernel having computed one.
    Pinned on both sides so a change to either the golden or nan_survives_to_l1 fails here
    rather than turning a green cell into a wall of xfails.

    lerp(inf, b, 1) is `inf + 1*(b - inf)` = inf - inf = NaN for a finite b, so this drives an
    emitted NaN rather than one that arrived on an operand.
    """
    import torch
    from helpers.golden_generators import TernarySFPUGolden

    a = torch.tensor([float("inf")] * 1024, dtype=torch.float32)
    b = torch.zeros(1024, dtype=torch.float32)
    c = torch.ones(1024, dtype=torch.float32)

    for input_format, output_format, dest_acc in _EDGE_SWEEP_CELLS:
        out = TernarySFPUGolden()(
            MathOperation.SfpuLerp,
            a,
            b,
            c,
            0,
            output_format,
            input_format=input_format,
            dest_acc=dest_acc,
        )
        survives = nan_survives_to_l1(input_format, output_format, dest_acc)
        got_nan = bool(torch.isnan(out[0]))
        assert got_nan == survives, (
            f"{input_format.name}->{output_format.name} dest_acc={dest_acc}: golden returns "
            f"{'NaN' if got_nan else float(out[0])}, but nan_survives_to_l1 says "
            f"{survives}"
        )
        if not survives:
            # The substituted infinity is positive: the golden clears the sign of every NaN it
            # emits, because SFPMAD promises the canonical 0x7fc00000 on Blackhole and leaves
            # the sign open on Wormhole -- so exporting one would be inventing it.
            assert float(out[0]) == float("inf"), (
                f"{input_format.name}->{output_format.name} dest_acc={dest_acc}: an emitted "
                f"NaN packed to {float(out[0])}, not +inf"
            )


@pytest.mark.parametrize(
    "op,operand",
    [
        (MathOperation.SfpuElwdiv, Operand.B),
        (MathOperation.SfpuAtan2, Operand.B),
        (MathOperation.Reciprocal, Operand.A),
        (MathOperation.SfpuAddcdiv, Operand.C),
        (MathOperation.SfpuSnakeBeta, Operand.C),
        (MathOperation.SfpuBinaryFmod, Operand.B),
    ],
)
def test_zero_poles_are_probed_with_both_signs(op, operand):
    """A zero pole is two probes, not one: 1/+0 and 1/-0 differ in the result's sign.

    The whole of cat G. Before this, boundary_probes() emitted only the +0.0 the table records,
    and the other zero arrived only through FLOAT_SPECIALS -- which is gated on
    *_SPECIALS_READY_OPS, and every op with a zero pole was outside it. So `div(x, -0.0)`, which
    must be the opposite sign from `div(x, +0.0)`, was driven nowhere in the suite.
    """
    values = edge_values(
        op,
        DataFormat.Float32,
        DataFormat.Float32,
        operand=operand,
        dest_acc=DestAccumulation.Yes,
    )
    signs = {math.copysign(1.0, v) for v in values if v == 0.0}
    assert signs == {1.0, -1.0}, f"{op.name} operand {operand.name} probes {values}"


def test_negative_zero_pole_probe_is_dropped_where_it_cannot_be_delivered():
    """Not sent on the datacopy path -- the LREG holds +0.0 there, so the probe is vacuous.

    The same gate cat B and cat D go through, and the reason Signbit's six former xfails were
    retired rather than kept: an xfail on a pipeline that flattens the datum blames the kernel
    for something it never received, and no kernel change could ever clear it.
    """
    values = edge_values(
        MathOperation.SfpuElwdiv,
        DataFormat.Float16_b,
        DataFormat.Float32,
        operand=Operand.B,
        dest_acc=DestAccumulation.Yes,
    )
    assert not any(_is_negative_zero(v) for v in values), values


def test_signed_zero_probe_survives_the_dedup():
    """_dedup_representable keys zeros by sign, so both survive a list that holds them.

    Load-bearing and easy to break: the two zeros are numerically equal and zero ULPs apart, so
    any dedup written on `==` or on a distance threshold would silently drop one of them and
    take the whole of cat G with it.
    """
    from helpers.sfpu_domains import _dedup_representable

    kept = _dedup_representable([0.0, -0.0, 1.0], DataFormat.Float32)
    signs = {math.copysign(1.0, v) for v in kept if v == 0.0}
    assert signs == {1.0, -1.0}, kept


# ─────────────────────────────────────────────────────────────────────────────
# The coverage ratchet
# ─────────────────────────────────────────────────────────────────────────────

# Floors, not exact counts: gaining coverage must not need a test edit, losing it must not pass
# silently. A, B, D, F and G come from sfpu_domains' own tables; C, E and the saturation half of
# cat F are lists in the suites, so they are floored here against those lists directly -- which
# also catches the rename that would otherwise empty one of them without a failure.
#
# Raising a floor is the last step of enrolling an op. `python -m helpers.sfpu_domains` prints
# the five it derives.
_COVERAGE_FLOORS = {
    "A": 23,
    "B": 92,
    "D": 66,
    "F": 69,
    "G": 17,
}
_SUITE_FLOORS = {
    "C integer extremes": 30,
    "E operand parameters": 7,
    "F saturation sweeps": 9,
}


def _suite_coverage_counts():
    """The three classes whose delivery machinery lives in the test modules, not in helpers."""
    import test_eltwise_binary_sfpu as binary
    import test_eltwise_unary_sfpu as unary
    import test_sfpu_ternary as ternary

    return {
        "C integer extremes": len(
            set(binary._INT_EXTREME_OPS)
            | set(binary._SHIFT_EDGE_OPS)
            | set(binary._UINT32_BINARY_OPS)
            | set(unary._INT_UNARY_OPS)
        ),
        "E operand parameters": len(
            set(unary._UNARY_SHIFT_OPS)
            | set(binary._SHIFT_EDGE_OPS)
            | set(ternary._SCALAR_OPS)
        ),
        "F saturation sweeps": len(
            set(unary._SATURATION_PROBES) | set(binary._BINARY_SATURATION_PAIRS)
        ),
    }


def test_coverage_does_not_regress():
    """Per class, at least as many ops are driven at it as the last time it was measured.

    The one thing the sweeps cannot do for themselves: coverage loss here is silent. Drop an op
    from an enrolment table and no test fails -- the sweep collects fewer variants and the run is
    green. Measured, removing three ops from the cat-F enrolment took 24 device variants out of
    the unary sweep without a single failure.
    """
    from helpers.sfpu_domains import coverage_counts

    counts = {**coverage_counts(), **_suite_coverage_counts()}
    floors = {**_COVERAGE_FLOORS, **_SUITE_FLOORS}
    shortfalls = {
        name: (counts[name], floor)
        for name, floor in floors.items()
        if counts[name] < floor
    }
    assert not shortfalls, (
        "coverage went backwards (class: got, floor): "
        f"{shortfalls}. Either an op lost its enrolment or a table entry was dropped."
    )


def test_shift_ops_really_do_drive_the_integer_extremes():
    """The shift sweeps are counted for cat C, so their value list has to contain an extreme.

    Counting them is only honest while the values are still there, so the claim is checked
    against the list rather than against the memory of having checked it.
    """
    import test_eltwise_binary_sfpu as binary
    from helpers.sfpu_domains import integer_specials

    extremes = set(integer_specials(DataFormat.Int32))
    driven = set(binary._SHIFT_EDGE_VALUES)
    assert extremes & driven, (
        "_SHIFT_EDGE_VALUES no longer contains an int32 extreme, so the shift ops must come "
        "out of the cat-C count"
    )
    # INT32_MAX specifically: INT32_MIN is filtered per-op (sign-magnitude Dst cannot hold it)
    # and has its own xfail, so it is present in the list but never actually driven.
    assert 2**31 - 1 in driven


def test_every_int_binary_op_drives_zero_or_records_why_not():
    """Zero reaches every int binary op, on both operands, or the exclusion is written down.

    `_INT_BINARY_STIMULI` gives each of these a single positive uniform range -- all but max/min
    starting at 1 -- so before the zero probe existed, gcd(0, x) = x and lcm(0, x) = 0 were
    simply never driven, and nothing recorded whether that was a decision or an omission. This
    converts "nobody got to it" into "here is why not".
    """
    import test_eltwise_binary_sfpu as binary

    for op in binary._INT_BINARY_STIMULI:
        pairs = binary._int_zero_pairs(op)
        assert any(a == 0 for a, _ in pairs), (
            f"{op.name}: a zero dividend is never driven, and 0 op x is defined for every op "
            "in this table"
        )
        drives_zero_divisor = any(b == 0 for _, b in pairs)
        excluded = op in binary._INT_ZERO_UNDEFINED_DIVISOR
        assert drives_zero_divisor != excluded, (
            f"{op.name}: a zero divisor is neither driven nor recorded in "
            "_INT_ZERO_UNDEFINED_DIVISOR (or it is both)"
        )
        if excluded:
            assert (
                len(binary._INT_ZERO_UNDEFINED_DIVISOR[op]) > 20
            ), f"{op.name}'s zero-divisor exclusion reason is too short to be a claim"


def test_every_int_binary_op_is_driven_at_the_extremes_or_recorded():
    """Cat C totality: driven at the integer extremes, or the exclusion is written down.

    The same shape as the zero-operand check above, and it exists for the same reason: reading
    _INT_BINARY_STIMULI's sub-range comments as "cannot be driven at an extreme" is how twelve
    of these thirteen ops came to look excluded when measurement showed them fine. Those ranges
    are accuracy bounds for the random sweep; whether an extreme *value* round-trips is a
    different question, and this is what makes the suite answer it per op.
    """
    import test_eltwise_binary_sfpu as binary

    driven = set(binary._INT_EXTREME_OPS) | set(binary._UINT32_BINARY_OPS)
    for op in binary._INT_BINARY_STIMULI:
        excluded = op in binary._INT_EXTREMES_OUT_OF_RANGE
        assert (op in driven) != excluded, (
            f"{op.name}: an int binary op must be either driven at the extremes or recorded "
            "in _INT_EXTREMES_OUT_OF_RANGE with a reason — not both, and not neither"
        )
        if excluded:
            assert (
                len(binary._INT_EXTREMES_OUT_OF_RANGE[op]) > 20
            ), f"{op.name}'s out-of-range reason is too short to be a claim"
    stale = sorted(
        op.name
        for op in set(binary._INT_EXTREMES_OUT_OF_RANGE)
        - set(binary._INT_BINARY_STIMULI)
    )
    assert (
        not stale
    ), f"these ops carry an out-of-range verdict but are not swept: {stale}"


def test_uint32_probe_reaches_the_actual_ceiling():
    """The uint32 sweep must drive integer_specials(UInt32), not merely approach it.

    It stopped at 2**32 - 2 for a while -- one value short -- so those ops were driven near the
    top of the range and never at it. All bits set is the pattern a sign-magnitude Dst is most
    likely to misread, so it is the one value in the list worth being exact about, and the
    ledger counts these ops for cat C on the strength of it.
    """
    import test_eltwise_binary_sfpu as binary
    from helpers.sfpu_domains import integer_specials

    driven = {v for pair in binary._UINT32_HIGH_PAIRS for v in pair}
    missing = sorted(set(integer_specials(DataFormat.UInt32)) - driven)
    assert not missing, f"the uint32 sweep never drives {missing}"


def test_signed_division_probe_separates_trunc_from_floor():
    """The signed-division pairs must contain inputs where the two conventions disagree.

    Truncating and flooring division differ only when the operands have opposite signs, so a
    probe list that happened to be all same-sign would drive both ops on stimuli that cannot
    tell them apart -- which is exactly the state the positive-only table left them in, and a
    green variant would look like it had fixed it.
    """
    import test_eltwise_binary_sfpu as binary
    import torch
    from helpers.golden_generators import BinarySFPUGolden

    golden = BinarySFPUGolden()
    separating = [
        (a, b)
        for a, b in binary._SIGNED_DIVISION_PAIRS
        if int(golden.ops[MathOperation.SfpuDivInt32](torch.tensor(a), torch.tensor(b)))
        != int(
            golden.ops[MathOperation.SfpuDivInt32Floor](
                torch.tensor(a), torch.tensor(b)
            )
        )
    ]
    assert separating, (
        "no pair in _SIGNED_DIVISION_PAIRS has trunc != floor, so the two ops are still "
        "driven on stimuli that cannot distinguish them"
    )


def test_uint32_probe_reaches_above_the_signed_boundary_on_both_sides():
    """The uint32 pairs must cross 2**31, and must order a large operand against a small one.

    Below 2**31 an unsigned op and its signed twin agree on every input, so this list is the
    only thing that can tell MaxUint32 from MaxInt32. Both halves of the claim are asserted:
    that large values are present at all, and that they are *paired against* small ones -- a
    random spec over two intervals gets the first and silently loses the second, because
    interval selection is proportional to length and the upper half is ~2000x longer.

    2**31 itself is excluded: it is the sign-magnitude "negative zero" pattern that cannot
    round-trip, which has its own xfail.
    """
    import test_eltwise_binary_sfpu as binary

    pairs = binary._UINT32_HIGH_PAIRS
    boundary = 2**31

    def as_signed(v):
        return v - 2**32 if v >= boundary else v

    assert any(
        a >= boundary or b >= boundary for a, b in pairs
    ), "no pair reaches above 2**31, where the unsigned ops differ from the signed ones"
    crossed = [(a, b) for a, b in pairs if (a >= boundary) != (b >= boundary)]
    assert crossed, (
        "every pair sits on one side of 2**31, so no comparison ever orders a large operand "
        "against a small one -- the case a signed reading of the bits gets backwards"
    )
    separating = [
        (a, b) for a, b in pairs if max(as_signed(a), as_signed(b)) != max(a, b)
    ]
    assert len(separating) > len(pairs) // 4, (
        f"only {len(separating)} of {len(pairs)} pairs distinguish an unsigned maximum from a "
        "signed one"
    )
    assert boundary not in {v for pair in pairs for v in pair}, (
        "2**31 is the sign-magnitude negative-zero pattern and cannot round-trip; use "
        "2**31 + 1 as INT32_MIN + 1 stands in on the signed side"
    )


def test_logsigmoid_exp_branch_is_a_logsigmoid():
    """-exp(-x) has to *be* logsigmoid(x) above the threshold, or modelling it proves nothing.

    BinarySFPUGolden._logsigmoid returns -t2 above 4.0, which makes the device test assert that
    the kernel selected the right branch and used the operand it was handed -- and stops it
    asserting anything about the mathematics. This is the other half: that the kernel's
    approximation is a good one on the interval it is used on, and in particular that the
    threshold sits where the approximation has already become good.

    The bound is loose on purpose. It is not a tolerance the device test uses; it is a guard
    that fails if the branch threshold ever moves down to where -exp(-x) is not a logsigmoid,
    which is the change that would make the modelled golden quietly vacuous.
    """
    import torch
    from helpers.golden_generators import BinarySFPUGolden

    threshold = BinarySFPUGolden._LOGSIGMOID_EXP_BRANCH
    x = torch.linspace(threshold, 40.0, 512, dtype=torch.float64)
    approximation = -torch.exp(-x)
    exact = torch.nn.functional.logsigmoid(x)
    worst = float(((approximation - exact).abs() / exact.abs()).max())
    assert worst < 0.02, (
        f"-exp(-x) is {worst:.2%} off logsigmoid(x) at its worst on [{threshold}, 40]; the "
        "golden models the kernel's branch verbatim, so this is the only thing asserting that "
        "the branch computes a logsigmoid at all"
    )
    # And that the error is worst at the threshold and decays, which is what makes a single
    # bound meaningful over an unbounded interval.
    at_threshold = float(((approximation[0] - exact[0]) / exact[0]).abs())
    assert at_threshold == pytest.approx(worst, rel=1e-6)


@pytest.mark.parametrize(
    "fmt", [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Int32]
)
def test_where_mixed_condition_is_mixed_on_every_format(fmt):
    """Both branches of the `mixed` where variant must be reachable, on every format.

    The in-test assertion catches this too, but only on a lane with hardware. This is the
    failure mode that motivated the whole check: a condition that is all-true passes against a
    golden that is also all-true, so the variant reports green while testing half of what it
    claims. `uniform(0.0, 1.0)` produced 0 exact zeros in 4096 on Float32 and 20 on Float16_b;
    only Int32's integer narrowing made it look mixed.

    Pinned per format because the two ways to get this wrong are format-specific: a float
    format rounds nothing to zero, and an integer one quantizes a fractional non-zero *to*
    zero -- so a spread chosen for one silently degenerates on the other.
    """
    import test_sfpu_ternary
    import torch
    from helpers.stimuli_generator import generate_stimuli

    spec = StimuliSpec(distribution=test_sfpu_ternary._where_mixed_condition, seed=0)
    src, _, _, _ = generate_stimuli(
        stimuli_format_A=fmt,
        input_dimensions_A=[64, 64],
        stimuli_format_B=fmt,
        input_dimensions_B=[64, 64],
        spec_A=spec,
        spec_B=spec,
    )
    values = src.flatten().to(torch.float32)
    frac_true = float((values != 0.0).float().mean())
    assert 0.2 < frac_true < 0.8, f"{fmt.name}: condition is {frac_true:.1%} true"
    # Both signs on the non-zero half, so the variant asserts that `cond != 0` and not
    # `cond > 0` is what selects the true branch.
    assert bool((values > 0).any()) and bool((values < 0).any()), (
        f"{fmt.name}: the non-zero half of the condition is single-signed, so a kernel "
        "selecting on cond > 0 would pass"
    )


# ─────────────────────────────────────────────────────────────────────────────
# How the probe fills the face
#
# Not metadata, but the mechanism that delivers it, and the failure mode is the same shape as
# everything else here: a probe that reaches only lanes 0-3 still reads as coverage, and no
# hardware run reports it -- the variant passes, having tested four lanes and 252 zeros.
# ─────────────────────────────────────────────────────────────────────────────


def test_edge_spec_cycles_probes_across_the_whole_face():
    """Edge probes must fill the face rather than leaving a zero tail.

    With a four-value median list against a 256-element face, a zero-filled tail makes the
    tolerance verdict a statement about 0.0: PCC and every aggregate are dominated by the
    filler, and the probes never leave the first vector operation of each face.
    """
    from helpers.sfpu_domains import edge_spec

    spec = edge_spec(
        MathOperation.Reciprocal,
        DataFormat.Float32,
        DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
    )
    assert spec.cycle, (
        "edge probes must fill the face; a zero-filled tail makes the verdict a statement "
        "about 0.0, not about the probe"
    )


def test_edge_spec_lets_a_caller_opt_out_of_cycling():
    """cycle=True is a default, not a fixed policy.

    The int comparison sweep builds its own spec around a zero tail on purpose -- the tail is
    its below-threshold probe -- so the knob has to stay reachable, or that sweep would have to
    stop using this builder to keep its stimulus.
    """
    from helpers.sfpu_domains import edge_spec

    spec = edge_spec(
        MathOperation.Reciprocal,
        DataFormat.Float32,
        DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
        cycle=False,
    )
    assert not spec.cycle


def test_cycled_custom_face_holds_only_probe_values():
    """A cycled face contains the probe values and nothing else -- no filler, in any lane.

    The strategy-level half of the same claim: `test_edge_spec_cycles_probes_across_the_whole_face`
    pins the flag, this pins what the flag does. A four-value list must produce a face with no
    zeros in it unless 0.0 is one of the four, and every lane must hold a value the caller asked
    for.
    """
    from helpers.stimuli_generator.strategies.structured import CustomStrategy

    size = 256
    values = [-2.5, -1.5, 1.5, 2.5]
    face = CustomStrategy().generate_face(
        StimuliSpec.custom(values=values, cycle=True),
        DataFormat.Float32,
        16,
        size,
        None,
    )
    assert face.numel() == size
    assert not bool((face == 0.0).any()), "a cycled face must have no filler lanes"
    assert set(face.tolist()) == set(values)
    # Tiled in order and truncated at the end, so a face that is not a multiple of the list
    # length still starts every repeat at values[0].
    assert face[: len(values)].tolist() == values
    assert face[len(values) : 2 * len(values)].tolist() == values


def test_uncycled_custom_face_still_zero_fills():
    """The default is unchanged, so a caller that relied on the tail still gets it."""
    from helpers.stimuli_generator.strategies.structured import CustomStrategy

    face = CustomStrategy().generate_face(
        StimuliSpec.custom(values=[1.0, 2.0]),
        DataFormat.Float32,
        16,
        256,
        None,
    )
    assert face[:2].tolist() == [1.0, 2.0]
    assert int((face == 0.0).sum()) == 254


def test_custom_rejects_an_over_long_list_only_when_it_would_drop_values():
    """Longer than a face is an error when writing at the head, and fine when tiling.

    Writing 300 values at the head of a 256-element face silently drops 44 of them, which is
    the worst failure mode an edge list can have. Tiling truncates at a value boundary instead,
    and every element is still one the caller asked for.
    """
    from helpers.stimuli_generator.strategies.structured import CustomStrategy

    long_list = [float(i) for i in range(300)]
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="cycle=True"
    ):
        CustomStrategy().generate_face(
            StimuliSpec.custom(values=long_list), DataFormat.Float32, 16, 256, None
        )

    face = CustomStrategy().generate_face(
        StimuliSpec.custom(values=long_list, cycle=True),
        DataFormat.Float32,
        16,
        256,
        None,
    )
    assert face.tolist() == long_list[:256]


# ─────────────────────────────────────────────────────────────────────────────
# Cat F — format extremes and the subnormal band
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fmt", [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float16]
)
def test_format_extremes_straddle_the_ftz_cliff(fmt):
    """The subnormal probe must be below the smallest normal and the min-normal probe at it.

    A probe placed on the wrong side of the flush-to-zero cliff tests nothing and looks like it
    tests everything: the pair exists to put one value where the hardware keeps it and one where
    the hardware (or the golden's _apply_ftz) does not, and if both land on the same side the
    variant still passes.

    Checked against golden_generators._FTZ_THRESHOLD rather than against a literal, because that
    is the number the goldens actually flush by -- the two are built from the same torch.finfo
    call and this is what keeps them so.
    """
    from helpers.golden_generators import _FTZ_THRESHOLD
    from helpers.sfpu_domains import _FORMAT_MIN_NORMAL, format_extremes

    magnitudes = sorted({abs(v) for v in format_extremes(fmt)})
    subnormal, min_normal = magnitudes[0], magnitudes[1]

    assert min_normal == _FORMAT_MIN_NORMAL[fmt]
    assert subnormal < min_normal, (
        f"{fmt.name}: the subnormal probe {subnormal} is not below the smallest normal "
        f"{min_normal}, so the pair asserts nothing about the subnormal band"
    )

    threshold = _FTZ_THRESHOLD[fmt]
    if fmt is DataFormat.Float16:
        # Float16 keeps subnormals, so its cliff is below the smallest *subnormal* and this
        # probe is a real value the hardware holds rather than one it flushes.
        assert subnormal > threshold, (
            f"{fmt.name} keeps subnormals, so the probe {subnormal} must be above the FTZ "
            f"threshold {threshold} -- otherwise it is a flushed value, not a subnormal one"
        )
    else:
        # Float32 and Float16_b flush the whole subnormal band, so the cliff sits at the
        # smallest normal and the probe must be on the flushed side of it.
        assert threshold == min_normal
        assert subnormal < threshold, (
            f"{fmt.name} flushes subnormals below {threshold}, and the probe {subnormal} is "
            "not below it"
        )


@pytest.mark.parametrize(
    "fmt", [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float16]
)
def test_format_extremes_are_exactly_representable(fmt):
    """Every cat-F probe must survive a round trip through the format it is a probe for.

    Same trap the saturation probes are written to avoid, and it applies just as hard here: a
    value written near a threshold gets pinned to a value other than the one it names. _FORMAT_MAX_MAGNITUDE's bfloat16 fallback is a decimal literal that sits a
    hair *above* the true bfloat16 maximum, so an unrounded ceiling probe would quantize on the
    way in and stop being the ceiling.
    """
    import torch
    from helpers.sfpu_domains import format_extremes

    dtype = format_dict[fmt]
    for value in format_extremes(fmt):
        round_tripped = float(
            torch.tensor([value], dtype=torch.float32).to(dtype).float()
        )
        assert round_tripped == value, (
            f"{fmt.name}: the probe {value!r} arrives as {round_tripped!r}, so it pins a "
            "value other than the one it names"
        )


def test_format_extremes_are_never_clipped_away():
    """clip_to_format() must keep every probe format_extremes() emits, on every pipeline.

    The two read the same table, and this is what keeps them reading it the same way: a ceiling
    derived from the format's own maximum but clipped against the *pipeline's* would silently
    drop the one probe the sweep exists to drive, leaving a variant that passes on six values
    instead of eight and says nothing about it.
    """
    from helpers.sfpu_domains import clip_to_format, format_extremes

    for input_format, output_format, dest_acc in _EDGE_SWEEP_CELLS:
        if not extremes_safe(input_format, output_format, dest_acc):
            continue
        range_fmt = narrowest_range_format(input_format, output_format)
        emitted = list(format_extremes(range_fmt))
        assert clip_to_format(emitted, range_fmt) == emitted, (
            f"{input_format.name}->{output_format.name}: clip_to_format drops "
            f"{sorted(set(emitted) - set(clip_to_format(emitted, range_fmt)))}"
        )


@pytest.mark.parametrize("dest_acc", [DestAccumulation.No, DestAccumulation.Yes])
@pytest.mark.parametrize("input_format", [DataFormat.Float32, DataFormat.Float16_b])
def test_subnormal_probe_is_sent_only_where_it_can_be_delivered(input_format, dest_acc):
    """The subnormal is dropped off the unpack-to-dest path; the other three probes are not.

    The measured half of cat F (see subnormal_delivered): Ceil, Floor, Sign and Signbit all
    answered as though the input were +0.0 on every pipeline but Float32 at dest_acc=Yes.
    Dropping the probe rather than xfailing it is the same decision Signbit's six retired
    entries record -- an xfail there blames the kernel for a datum it never received.
    """
    from helpers.sfpu_domains import _FORMAT_MIN_NORMAL, extreme_values

    values = extreme_values(input_format, input_format, dest_acc)
    subnormals = [
        v for v in values if v != 0.0 and abs(v) < _FORMAT_MIN_NORMAL[input_format]
    ]
    delivered = input_format.is_32_bit() and dest_acc == DestAccumulation.Yes

    assert bool(subnormals) == delivered, (
        f"{input_format.name} dest_acc={dest_acc}: subnormal probes {subnormals}, but "
        f"subnormal_delivered says {delivered}"
    )
    # The ceiling, its neighbour and the smallest normal are unaffected either way, so the
    # gate must never be doing more than it claims.
    assert len(values) == (8 if delivered else 6), (
        f"{input_format.name} dest_acc={dest_acc}: {len(values)} probes, expected the four "
        "magnitudes in both signs minus the subnormal pair where it is not delivered"
    )


def test_extremes_gate_is_not_the_specials_gate():
    """extremes_safe() and specials_safe() must not collapse into each other.

    They answer different questions -- whether a finite datum with an extreme exponent arrives,
    against whether a non-finite one does -- and the whole reason cat F has its own flag is that
    specials_safe()'s two measured breakers (a Float16 anywhere, a 16-bit input into a 32-bit
    Dest) are about non-finites and do not apply to a subnormal or a ceiling. If someone
    "simplifies" one into the other, this fails rather than silently taking cat F off the cells
    where it is the whole point.
    """
    from helpers.sfpu_domains import extremes_safe

    differ = [
        cell
        for cell in _EDGE_SWEEP_CELLS
        if extremes_safe(*cell) != specials_safe(*cell)
    ]
    assert differ, (
        "extremes_safe() now agrees with specials_safe() on every cell the edge sweep "
        "reaches, which means one of them has been redefined in terms of the other"
    )
    # Every cell specials_safe accepts, extremes_safe must accept too: a pipeline that carries
    # a NaN certainly carries a large finite number. The converse is what differs.
    for cell in _EDGE_SWEEP_CELLS:
        if specials_safe(*cell):
            assert extremes_safe(
                *cell
            ), f"{cell} carries specials but not extremes, which cannot be right"


def test_format_extremes_rejects_formats_with_no_per_element_small_end():
    """Integer and block-float formats raise rather than returning a plausible number.

    Bfp8_b's smallest element is set by the exponent shared across its 16-element block, so any
    single number returned for it would be wrong for every block but one -- and would look
    entirely reasonable in a probe list.
    """
    from helpers.sfpu_domains import format_extremes

    for fmt in (DataFormat.Int32, DataFormat.UInt32):
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            ValueError, match="cat C"
        ):
            format_extremes(fmt)

    for fmt in (DataFormat.Bfp8_b, DataFormat.Bfp4_b):
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            ValueError, match="shared across a block"
        ):
            format_extremes(fmt)


def test_every_extremes_enrolment_carries_a_claim():
    """A cat-F reason string has to say something, as the cat-B ones do."""
    from helpers.sfpu_domains import EXTREMES_READY_OPS

    assert EXTREMES_READY_OPS, "the cat-F tranche is empty, so nothing is ever driven"
    for op, reason in EXTREMES_READY_OPS.items():
        assert len(reason) > 20, f"{op.name}'s cat-F reason is too short to be a claim"


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
    import test_eltwise_unary_sfpu

    assert MathOperation.ExpWithBase in test_eltwise_unary_sfpu.STANDARD_SWEEP_OPS
    assert MathOperation.ExpWithBase not in test_eltwise_unary_sfpu.BROAD_SWEEP_OPS
    accurate = for_op(
        MathOperation.ExpWithBase,
        DataFormat.Float32,
        approx_mode=ApproximationMode.No,
    ).spec_A
    assert accurate.high == 160.0, (
        "the accurate path's ceiling moved; it is measured green at 160 on a Wormhole n300 and "
        "has no custom tolerance, so a change here wants a re-run"
    )


def test_hardtanh_golden_matches_the_clamp_golden():
    """Hardtanh's golden must stay Clamp's golden -- pin the identity.

    Both ops bind metal kernels that are the same SFPSWAP max-then-min composition
    (calculate_clamp's unary_max_min chain; calculate_hardtanh's sfpi::clamp), so one
    golden -- sfpu_clamp -- models both. If either golden is ever remodelled
    independently, the divergence surfaces here.
    """
    from helpers.golden_generators import UnarySFPUGolden

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


def test_pow_edge_pairs_include_negative_zero_exponent():
    """The setsgn(pow, 0) guard is only as good as the pairs that would fail without it.

    The hardware sweep's both_zero class keys on a == 0 and b == 0, which +0.0 satisfies
    on its own, so a sweep that never generated exponent -0.0 would stay green after
    dropping setsgn. Pin the stimulus here, host-side: Float32 dest_acc=Yes is the
    pipeline that actually delivers a signed zero, and the cartesian product must
    include a positive base, a negative base, and a zero base against that exponent.
    """
    pairs = edge_pair_values(
        MathOperation.SfpuElwpow,
        DataFormat.Float32,
        DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
    )
    neg_zero_bases = [a for a, b in pairs if _is_negative_zero(b)]
    assert neg_zero_bases, (
        "SfpuElwpow edge pairs must include exponent -0.0 on the pipeline that "
        "delivers a signed zero; otherwise test_sfpu_binary_edges cannot catch a "
        "regression that drops setsgn(pow, 0)"
    )
    assert any(a > 0.0 for a in neg_zero_bases), neg_zero_bases
    assert any(a < 0.0 for a in neg_zero_bases), neg_zero_bases
    assert any(a == 0.0 for a in neg_zero_bases), neg_zero_bases

    # The datacopy path flattens -0.0, so claiming the pair there would be false coverage.
    flattened = edge_pair_values(
        MathOperation.SfpuElwpow,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        dest_acc=DestAccumulation.No,
    )
    assert not any(_is_negative_zero(b) for _, b in flattened)


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
