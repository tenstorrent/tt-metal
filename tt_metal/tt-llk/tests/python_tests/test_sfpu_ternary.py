# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import struct

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TernarySFPUGolden,
    WhereGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize, runtime
from helpers.sfpu_domains import (
    _OP_DOMAIN_REGISTRY,
    BLOCK_SPREAD_DECADES,
    TERNARY_SPECIALS_READY_OPS,
    Operand,
    block_spread_spec,
    edge_values,
    exclude_undefined_pair,
    for_op,
    generated_nan_sign_is_asserted,
    nan_survives_to_l1,
    negative_zero_delivered,
    specials_safe,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    DISABLE_SRC_ZERO_FLAG,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    SFPU_TERNARY_OP,
    SFPU_TERNARY_SCALAR,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.utils import passed_test


def _scalar_bits(value: float) -> int:
    """*value* as the raw fp32 bit pattern the kernel receives in SFPU_TERNARY_SCALAR.

    The kernel decodes it with Converter::as_float / SFPLOADI, and TernarySFPUGolden decodes
    it the same way, so this is the one place a scalar becomes a template argument.
    """
    return struct.unpack("<I", struct.pack("<f", value))[0]


_SCALAR_VALUE = 2.0
_SCALAR_VALUE_BITS = _scalar_bits(_SCALAR_VALUE)


# Helper check function
def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


def _ternary_default_specs(mathop, input_format):
    """Per-operand defaults for *mathop*: its registered domain, else the built-in one.

    No ternary op has an _OP_DOMAIN_REGISTRY entry, so every op currently takes the
    built-in branch. This is the single place a registered domain would take effect, and
    callers of _run_sfpu_ternary can override any operand to reach an edge the defaults
    exclude (e.g. the c -> 0 pole that addcdiv and snake_beta pin away from).

    The registry branch reads spec_C rather than reusing spec_B for it: that reuse was correct
    only while OperandSpecs had two operands, and keeping it would silently drop a registered C
    domain on the one code path that exists to honour it.
    """
    if mathop in _OP_DOMAIN_REGISTRY:
        specs = exclude_undefined_pair(mathop, for_op(mathop, input_format))
        return specs.spec_A, specs.spec_B, specs.spec_C

    # addcdiv and snake_beta divide by c, so c is held away from zero.
    divide_by_c = mathop in (MathOperation.SfpuAddcdiv, MathOperation.SfpuSnakeBeta)
    spec_ab = StimuliSpec.uniform(low=-1.0, high=1.0)
    spec_c = (
        StimuliSpec.uniform(low=1.0, high=2.0)
        if divide_by_c
        else StimuliSpec.uniform(low=-1.0, high=1.0)
    )
    return spec_ab, spec_ab, spec_c


def _run_sfpu_ternary(
    formats,
    dest_acc,
    mathop,
    input_dimensions=[64, 64],
    spec_A=None,
    spec_B=None,
    spec_C=None,
    unspecified_nonfinite_sign=False,
    scalar_bits=_SCALAR_VALUE_BITS,
):
    """Drive one ternary variant; returns (src_A, golden_tensor, res_tensor) for extra checks.

    *scalar_bits* is the addc multiplier as a raw fp32 bit pattern, reaching the kernel as a
    `constexpr std::uint32_t SFPU_TERNARY_SCALAR` -- a compile-time axis, passed as an argument
    so the templates list and the golden call cannot disagree about which value was driven.

    *unspecified_nonfinite_sign* compares a non-finite result by magnitude only, for the one case
    where the sign genuinely is not specified: a NaN the kernel emitted, packed as a signed
    infinity through a pipeline too narrow to hold it, on Wormhole, where `SFPMAD.md` says that
    sign "might or might not be set". Better than withdrawing the variant, since magnitude,
    finiteness and every finite lane stay checked.

    Scoped per lane, from the mask the golden records while the NaN is still a NaN, because one
    tensor holds both kinds of non-finite: `inf + (-inf)`, whose sign the ISA leaves open, and
    `lerp(-inf, b, 0) = -inf`, which IEEE fully specifies.
    """
    # The specs below carry no seed, so seed here: an unseeded redraw makes a variant
    # sitting near its tolerance pass or fail by luck. Same as the binary driver.
    torch.manual_seed(0)

    default_A, default_B, default_C = _ternary_default_specs(
        mathop, formats.input_format
    )
    spec_a = spec_A if spec_A is not None else default_A
    spec_b = spec_B if spec_B is not None else default_B
    spec_c = spec_C if spec_C is not None else default_C

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_a,
        spec_B=spec_b,
    )

    src_C, tile_cnt_C, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_c,
        spec_B=spec_c,
    )

    # input_format and dest_acc turn on the Dest-width and pack-path modelling: the SFPU
    # evaluates in fp32 and stores into a Dest whose width dest_acc selects, and the packer
    # substitutes a signed infinity for a NaN a narrower pipeline cannot hold. Both steps are
    # sub-ULP on a finite value and decisive on a non-finite one, so they are what makes the
    # specials_in class assertable rather than a wall of "returns inf where IEEE says nan".
    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden = generate_golden(
        mathop,
        src_A,
        src_B,
        src_C,
        scalar_bits,
        formats.output_format,
        input_format=formats.input_format,
        dest_acc=dest_acc,
        collect_generated_nan=unspecified_nonfinite_sign,
    )
    # Asked of the return value rather than of unspecified_nonfinite_sign, the way the binary
    # driver asks it: DummyGoldenGenerator stands in for the golden under --compile-producer and
    # returns a bare tensor whatever it is asked for, so keying the unpack off the flag would
    # raise there and starve the shared ELF instead of building it. That phase skips inside
    # run() below, before any comparison, so a None mask costs it nothing.
    emitted_nan = None
    if isinstance(golden, tuple):
        golden, emitted_nan = golden

    configuration = TestConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        templates=[
            SFPU_TERNARY_OP(mathop),
            SFPU_TERNARY_SCALAR(scalar_bits),
            APPROX_MODE(ApproximationMode.No),
            DISABLE_SRC_ZERO_FLAG(True),
            DEST_SYNC(),
        ],
        runtimes=[NUM_BLOCKS(tile_cnt_A), NUM_TILES_IN_BLOCK(1)],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            buffer_C=src_C.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_cnt_C,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[: len(golden)]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    golden_tensor = torch.tensor(golden, dtype=torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    if emitted_nan is not None:
        # Clear the sign only on the lanes that held an emitted NaN *and* where both sides are
        # non-finite. A golden +inf against a hardware 5.0 still compares +inf vs 5.0 and still
        # fails; a golden +inf against a hardware NaN likewise, because abs() leaves a NaN a NaN
        # and passed_test's both-NaN clause needs both. So this excuses one bit on the lanes the
        # ISA declines to pin, and nothing else anywhere.
        unspecified = (
            emitted_nan[: len(golden_tensor)]
            & ~torch.isfinite(golden_tensor)
            & ~torch.isfinite(res_tensor)
        )
        golden_tensor = torch.where(unspecified, golden_tensor.abs(), golden_tensor)
        res_tensor = torch.where(unspecified, res_tensor.abs(), res_tensor)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"

    return src_A, golden_tensor, res_tensor


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=[
        MathOperation.SfpuAddcmul,
        MathOperation.SfpuAddcdiv,
        MathOperation.SfpuLerp,
        MathOperation.SfpuSnakeBeta,
    ],
)
def test_sfpu_ternary(formats, dest_acc, mathop):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Bfp8_b
        and mathop != MathOperation.SfpuAddcmul
    ):
        pytest.skip("Bfp8_b is only supported for addcmul")

    _run_sfpu_ternary(formats, dest_acc, mathop)


# ─────────────────────────────────────────────────────────────────────────────
# Deliberate edge values, per operand and per failure class
#
# The random sweep holds c in uniform(1, 2) for addcdiv and snake_beta because both divide by it,
# so the pole is unreachable by construction. This drives it, and drives the two operands the
# sweep never touched. `edge_values(op, ..., operand=X)` resolves it through the usual metadata
# -- addcdiv and snake_beta from _OP_SINGULARITIES, lerp from _OP_OPERAND_EDGE_POINTS, addcmul
# from nothing, a multiply having no pole and no knee -- so an op joins by gaining a table entry.
#
# Two runtime axes, and each is separate for its own reason.
#
# *operand* says which of a, b, c carries the probe; the other two keep their random domains, so
# every probe meets a fresh random pair in each of the sixteen faces StimuliSpec.custom fills.
# Pinning two operands would pair their lists *index-wise* rather than crossing them --
# generate_stimuli fills each independently and tilize pairs by position -- so five probes
# against five would test five combinations, not twenty-five, and would spend the rest of every
# face on the (0, 0, 0) the zero-fill leaves behind.
#
# *edge_class* splits the finite edges (cat A's singularity straddled by a ULP step, cat D's
# knees, the signed zeros) from the non-finite ones (cat B, gated on TERNARY_SPECIALS_READY_OPS
# and specials_safe()), classified by `math.isfinite` on the probe. One class per variant: a
# non-finite operand is a different question from a pole, and without the split addcdiv's c = 0
# and c = inf would share one xfail and a regression in either would hide behind the other.
# ─────────────────────────────────────────────────────────────────────────────

_TERNARY_EDGE_OPS = [
    MathOperation.SfpuAddcdiv,
    MathOperation.SfpuAddcmul,
    MathOperation.SfpuLerp,
    MathOperation.SfpuSnakeBeta,
]

_TERNARY_EDGE_CLASS_POLE = "pole"
_TERNARY_EDGE_CLASS_SPECIALS = "specials_in"

# Order is documentation, not mechanism -- but the first entry is the one that builds the
# shared ELF under --compile-producer (conftest._collapse_runtime_only_variants keeps one item
# per compile key), so the class most likely to be non-empty goes first. See the PRODUCE guard
# in the test body for what happens when it is not.
_TERNARY_EDGE_CLASSES = (_TERNARY_EDGE_CLASS_POLE, _TERNARY_EDGE_CLASS_SPECIALS)

_TERNARY_OPERANDS = (Operand.A, Operand.B, Operand.C)

# Ops that divide by c, and therefore need a numerator held away from zero.
#
# c = 0 with an unconstrained numerator mixes two questions: the pole with a nonzero numerator,
# where every element should be ±inf, and 0/0, the indeterminate form already recorded against
# div, fmod, remainder and xlogy in the binary suite. Measured on Blackhole, unconstrained
# addcdiv and snake_beta fail only where the golden is NaN and agree on every ±inf, so holding
# the numerator off zero turns a tolerated xfail into a real assertion about the pole. Driving
# 0/0 here would want its own variant and xfail, as the binary suite splits classes.
_TERNARY_DIVIDES_BY_C = frozenset(
    {MathOperation.SfpuAddcdiv, MathOperation.SfpuSnakeBeta}
)

# |x| >= 0.5 on both a and b. addcdiv's numerator is value * b, so b alone decides it;
# snake_beta's is sin(b*a)^2, which vanishes only when b*a is an exact multiple of pi, and
# holding both off zero keeps it clear of that too (|b*a| <= 1 < pi).
#
# Two specs differing only in seed: the seed is per-spec, so one spec shared by both operands
# makes them bit-identical and every variant runs a == b -- which still reaches the pole on c,
# but degenerates snake_beta from sin(b*a) to sin(a^2) and hides a kernel reading the wrong
# operand. Seeded rather than defaulted so the streams stay reproducible while differing.
_TERNARY_NONZERO_A = StimuliSpec.uniform(intervals=[(-1.0, -0.5), (0.5, 1.0)], seed=0)
_TERNARY_NONZERO_B = StimuliSpec.uniform(intervals=[(-1.0, -0.5), (0.5, 1.0)], seed=1)


def _ternary_cat_b_enabled(mathop, formats, dest_acc):
    """Two independent gates, and both must pass.

    TERNARY_SPECIALS_READY_OPS says this op's *golden* defines an answer at a non-finite
    operand; specials_safe() says this *pipeline* delivers one intact. Neither implies the
    other, so both are asked -- the same shape the unary and binary edge sweeps use.
    """
    return mathop in TERNARY_SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )


def _ternary_edge_class_values(
    mathop, formats, operand, edge_class, dest_acc, specials
):
    """The probe values of *edge_class* for (*mathop*, *operand*) on this pipeline.

    One edge_values() call partitioned by finiteness rather than two calls with different
    `specials`, so the two classes cannot come to disagree about which value belongs where.
    """
    vals = edge_values(
        mathop,
        formats.input_format,
        formats.output_format,
        operand=operand,
        specials=specials,
        dest_acc=dest_acc,
    )
    if edge_class == _TERNARY_EDGE_CLASS_SPECIALS:
        return [v for v in vals if not math.isfinite(v)]
    return [v for v in vals if math.isfinite(v)]


def _producer_probe_values(mathop, formats, dest_acc, specials):
    """Any non-empty probe list for this compile key, or [] if the op has no edge at all here.

    For the compile-producer pass only. `operand` and `edge_class` are both runtime() axes, so
    _collapse_runtime_only_variants keeps one item per compile key -- operand A, class pole --
    and that item builds the ELF the other five share; a skip there leaves them running against
    a binary that was never built, which presents as TENSIX TIMED OUT.

    So the fallback drops *both* axes. Dropping the class alone is not enough, and Float16_b /
    dest_acc=Yes is why: cat B is off there, leaving operands A and B with no edge of either
    class while operand C still has its pole -- 3 values for addcdiv and snake_beta, 4 for lerp
    -- which the consumer will execute.

    Any non-empty list compiles the right kernel, since the ELF depends on (op, formats,
    dest_acc) and never on which values go in which tensor, and the producer builds then skips
    before the stimulus reaches a device. The consumer still partitions and still skips. An op
    with nothing on any operand -- addcmul on that cell -- returns [] and skips correctly.
    """
    for candidate in _TERNARY_OPERANDS:
        vals = edge_values(
            mathop,
            formats.input_format,
            formats.output_format,
            operand=candidate,
            specials=specials,
            dest_acc=dest_acc,
        )
        if vals:
            return vals
    return []


# The cells this sweep's format axis can reach, so the divergence sets below are derived from
# the same gates the stimulus is, rather than transcribed.
_TERNARY_EDGE_CELLS = tuple(
    (fmt.input_format, fmt.output_format, dest_acc)
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32], same=True
    )
    for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
)


def _cat_b_cells(applies=lambda _in_fmt, _out_fmt, _dest_acc: True):
    """The specials-carrying cells of this sweep for which *applies* is true."""
    return tuple(
        cell for cell in _TERNARY_EDGE_CELLS if specials_safe(*cell) and applies(*cell)
    )


# What driving the ternary specials found on a Blackhole p150, once both goldens modelled the Dest
# write and the pack (which accounted for 10 cells on its own). Keyed by (op, operand), scoped to
# the specials_in class -- every entry is a non-finite *operand*, and the pole class agreed
# everywhere. Non-strict, so each case still executes and reports XPASS if behaviour changes, and
# derived from the delivery gates rather than listed so a cell drifting in or out shows up.
#
# TWO CAUSES, NOT FOUR:
#
#   c = NaN through the reciprocal (addcdiv and snake_beta, operand C). Both build the divide on
#   SFPARECIP, which returns +0 for 1/NaN instead of propagating, so the result is `a` where the
#   golden says NaN -- the divergence unary Reciprocal already carries, through the same
#   primitive. On every specials-carrying cell, being arithmetic rather than a delivery fact.
#   Measured: c = +/-inf agrees on both cells, which is what makes this the NaN probe alone.
#
#   A non-finite reaching the sin (snake_beta, operands A and B). sin(b*a) then has a square of
#   +inf against a golden NaN, and SFPLUTFP32 documents no NaN/inf handling, so what the
#   polynomial does there is an LLK decision with no ISA ruling. Scoped to the cells where a NaN
#   *survives to L1*, from nan_survives_to_l1() rather than transcribed: where it does not, the
#   golden's NaN packs to the same +inf and the two agree by substitution.
_TERNARY_EDGE_KNOWN_DIVERGENCES = {
    (MathOperation.SfpuAddcdiv, Operand.C): _cat_b_cells(),
    (MathOperation.SfpuSnakeBeta, Operand.C): _cat_b_cells(),
    (MathOperation.SfpuSnakeBeta, Operand.A): _cat_b_cells(nan_survives_to_l1),
    (MathOperation.SfpuSnakeBeta, Operand.B): _cat_b_cells(nan_survives_to_l1),
}

_RECIPROCAL_NAN_NOTE = (
    "the kernel's reciprocal returns +0 for 1/NaN instead of propagating, so the quotient "
    "vanishes and the result is `a`; the same divergence unary Reciprocal carries, through the "
    "same SFPARECIP composition. c = +/-inf agrees, so this is the NaN probe alone"
)

_TERNARY_EDGE_REASON = {
    (MathOperation.SfpuAddcdiv, Operand.C): f"addcdiv(a, b, NaN) returns a, not NaN "
    f"({_RECIPROCAL_NAN_NOTE}).",
    (
        MathOperation.SfpuSnakeBeta,
        Operand.C,
    ): f"snake_beta(a, b, NaN) returns a, not NaN "
    f"({_RECIPROCAL_NAN_NOTE}).",
    (
        MathOperation.SfpuSnakeBeta,
        Operand.A,
    ): "sin(b*a) with a non-finite gives the kernel a "
    "value whose square is +inf, where torch gives NaN, so the result is +inf against a golden "
    "NaN. SFPLUTFP32 documents no NaN/inf handling, so this is an LLK decision with no ISA "
    "ruling. Only on the cells where a NaN survives to L1 -- elsewhere the pack substitutes the "
    "same +inf and the two agree.",
    (
        MathOperation.SfpuSnakeBeta,
        Operand.B,
    ): "As operand A: the non-finite reaches the same "
    "sin through the b*a product.",
}

assert set(_TERNARY_EDGE_REASON) == set(_TERNARY_EDGE_KNOWN_DIVERGENCES), (
    "_TERNARY_EDGE_REASON and _TERNARY_EDGE_KNOWN_DIVERGENCES disagree on which (op, operand) "
    f"pairs diverge: {set(_TERNARY_EDGE_REASON) ^ set(_TERNARY_EDGE_KNOWN_DIVERGENCES)}"
)
assert all(
    cells for cells in _TERNARY_EDGE_KNOWN_DIVERGENCES.values()
), "an (op, operand) claiming a divergence with no cell to apply it to is a dead xfail"


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_TERNARY_EDGE_OPS,
    # runtime(): both axes select which values go into which operand tensor and nothing
    # else, so all six share the one ELF the (op, formats, dest_acc) triple decides.
    operand=runtime(list(_TERNARY_OPERANDS)),
    edge_class=runtime(list(_TERNARY_EDGE_CLASSES)),
)
def test_sfpu_ternary_operand_edges(
    request, formats, dest_acc, mathop, operand, edge_class
):
    """Drive one class of one ternary operand's edges against random values on the other two."""
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    specials = _ternary_cat_b_enabled(mathop, formats, dest_acc)

    # Marked before the stimulus is built, but only for the specials_in class: every recorded
    # divergence is a non-finite operand, and the pole class shares neither the cause nor the
    # cells. Where the class is empty the variant skips below and the marker never fires.
    reason = _TERNARY_EDGE_REASON.get((mathop, operand))
    if (
        reason is not None
        and edge_class == _TERNARY_EDGE_CLASS_SPECIALS
        and (formats.input_format, formats.output_format, dest_acc)
        in _TERNARY_EDGE_KNOWN_DIVERGENCES[(mathop, operand)]
    ):
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=False))

    vals = _ternary_edge_class_values(
        mathop, formats, operand, edge_class, dest_acc, specials
    )

    if not vals and TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        # The compile-producer pass must not skip on a runtime-only axis: it would starve the
        # shared ELF. Unpartitioned and un-operanded, for the reasons in the helper.
        vals = _producer_probe_values(mathop, formats, dest_acc, specials)

    if not vals:
        pytest.skip(
            reason=f"{mathop.name} operand {operand.name} has no {edge_class} edge for "
            "this pipeline"
            + (
                ""
                if edge_class != _TERNARY_EDGE_CLASS_SPECIALS or specials
                else " (cat B is off for this op or this pipeline)"
            )
        )

    # Keep the numerator off zero when the probed operand is the divisor, so the variant
    # asserts the pole (and c = ±inf, which gives an exact zero quotient) rather than the 0/0
    # indeterminate form. See _TERNARY_DIVIDES_BY_C. Probing a or b instead leaves c on its
    # uniform(1, 2) default, which is already off the pole, so no guard is needed there.
    guard = operand == Operand.C and mathop in _TERNARY_DIVIDES_BY_C
    specs = {
        Operand.A: _TERNARY_NONZERO_A if guard else None,
        Operand.B: _TERNARY_NONZERO_B if guard else None,
        Operand.C: None,
    }
    # cycle=True: the probed operand fills its face instead of leaving a zero tail, so the
    # probe meets a fresh random pair in every lane rather than in the first few, and the
    # verdict is not dominated by the (0, random, random) triples the tail would create.
    specs[operand] = StimuliSpec.custom(values=vals, seed=0, cycle=True)

    # Where the golden's answer is a NaN the op emitted, a narrowing pipeline turns its sign
    # into the observable result, and Wormhole's SFPMAD leaves that sign unspecified -- so
    # assert the magnitude there rather than withdrawing the variant. Blackhole specifies the
    # canonical NaN and keeps the full assertion. Pipeline and arch only: which lanes hold an
    # emitted NaN is the golden's own mask, since one tensor carries `inf + (-inf)` alongside
    # `lerp(-inf, b, 0) = -inf`.
    unspecified_sign = generated_nan_sign_is_asserted(
        formats.input_format,
        formats.output_format,
        dest_acc,
        on_wormhole=TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE,
    )

    _run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        spec_A=specs[Operand.A],
        spec_B=specs[Operand.B],
        spec_C=specs[Operand.C],
        unspecified_nonfinite_sign=unspecified_sign,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The addc multiplier
#
# `value` reaches the kernel as a `constexpr std::uint32_t SFPU_TERNARY_SCALAR`, so varying it is
# a compile-time axis -- and it was not an axis at all before: 2.0 everywhere. Three probes are
# worth the six ELFs. At 0.0 both ops collapse to the identity in `a`, which is a cheap and very
# strong check that neither kernel reads the wrong Dst tile -- one returning `b` or `c` would pass
# every other variant here, all three operands carrying plausible random values. 1.0 removes the
# multiply; -2.0 flips a sign the golden and the kernel have to agree on.
#
# One format column and one dest_acc value, deliberately: the scalar is orthogonal to both, and
# crossing it with the full profile would multiply ELFs for no new question.
#
# Scoped to the ops' *ordinary* domains, so the identity at value = 0 is a clean assertion. Its
# interaction with addcdiv's pole is measured rather than guessed -- at value = 0 and c = 0 the
# kernel returns NaN on Blackhole, matching the golden's IEEE reading, and addcmul returns `a` on
# all 4096 lanes -- but recorded here rather than driven, since one variant holding both the pole
# and the scalar is what the edge_class split exists to prevent.
# ─────────────────────────────────────────────────────────────────────────────

_SCALAR_PROBES = (0.0, 1.0, -2.0)

# The two ops that read the scalar at all. lerp's weight is operand C and snake_beta has no
# multiplier, so SFPU_TERNARY_SCALAR is dead template argument for both.
_SCALAR_OPS = [MathOperation.SfpuAddcmul, MathOperation.SfpuAddcdiv]


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.Yes],
    mathop=_SCALAR_OPS,
    scalar=list(_SCALAR_PROBES),
)
def test_sfpu_ternary_scalar(formats, dest_acc, mathop, scalar):
    """Drive addcmul and addcdiv at a multiplier other than the hardcoded 2.0."""
    src_A, _, res_tensor = _run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        scalar_bits=_scalar_bits(scalar),
    )

    if scalar != 0.0:
        return

    # value = 0 makes both ops the identity in `a`, and it has to hold *exactly* rather than
    # within a tolerance: `a + 0*x` is `a` for every finite x, with no rounding anywhere to
    # excuse a difference. passed_test() has already compared against the golden, which says
    # the same thing -- this asserts it against the stimulus instead, so a golden that made
    # the same mistake as the kernel could not hide it.
    expected = src_A.flatten().to(format_dict[formats.output_format])[: len(res_tensor)]
    mismatched = int((res_tensor != expected).sum())
    assert mismatched == 0, (
        f"{mathop.name} with value = 0 must return `a` bit for bit, but {mismatched} of "
        f"{len(res_tensor)} lanes differ — the kernel is reading an operand it should be "
        "multiplying away"
    )


# ─────────────────────────────────────────────────────────────────────────────
# addcmul's cancellation edge
#
# addcmul is the one ternary op with nothing in _OP_SINGULARITIES or _OP_OPERAND_EDGE_POINTS --
# `a + value * b * c` is smooth in all three operands -- so the sweep above gives it only cat B.
# What it does have is exact cancellation: at a = -value * b * c the result must be zero, and the
# *sign* of that zero is a real hardware question, SFPMAD flushing to positive zero on Wormhole
# against sign-preserved zero on Blackhole -- the same split that arch-gates
# _EDGE_CLASS_NEGATIVE_ZERO in the binary suite.
#
# An explicit triple rather than a StimuliSpec, because the relation is *between* the operands
# and no per-operand domain can express it. Every b and c is a power of two and value is 2.0, so
# the product is exact in every format here and the cancellation is not a rounding artifact.
#
# The variant asserts the result is *zero*, which is the part that can fail: passed_test() judges
# by torch.isclose, a both-NaN clause and PCC, and -0.0 == +0.0 under all three. The sign would
# need a bitwise comparator, which is a suite-wide change.
# ─────────────────────────────────────────────────────────────────────────────

# (b, c) pairs; a is derived as -_SCALAR_VALUE * b * c. Powers of two, both signs, three decades
# of magnitude, so the product is exact and the cancellation is tested across the exponent range.
_ADDCMUL_CANCELLATION_BC = (
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
    (0.5, 0.5),
    (-0.25, 8.0),
    (64.0, 0.125),
    (-16.0, -4.0),
    (
        0.0,
        1.0,
    ),  # a = -0.0 exactly: 0 + value*0*1, the signed-zero case of the same relation
    (1.0, 0.0),
)


def _addcmul_cancellation_specs():
    """(spec_A, spec_B, spec_C) whose lanes satisfy a + value*b*c == 0 exactly."""
    b = [b for b, _ in _ADDCMUL_CANCELLATION_BC]
    c = [c for _, c in _ADDCMUL_CANCELLATION_BC]
    a = [-_SCALAR_VALUE * bv * cv for bv, cv in _ADDCMUL_CANCELLATION_BC]
    # cycle=True on all three so the relation a == -value*b*c holds in every lane. A zero
    # tail would leave 0 + value*0*0 == 0 across ~96% of the tensor, which is a true
    # statement about zero and not the cancellation this variant exists to drive.
    return (
        StimuliSpec.custom(values=a, seed=0, cycle=True),
        StimuliSpec.custom(values=b, seed=0, cycle=True),
        StimuliSpec.custom(values=c, seed=0, cycle=True),
    )


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.SfpuAddcmul,
)
def test_sfpu_addcmul_cancellation(formats, dest_acc, mathop):
    """a + value*b*c with a chosen to cancel the product exactly: the result must be zero."""
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    spec_A, spec_B, spec_C = _addcmul_cancellation_specs()
    _, _, res_tensor = _run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        spec_A=spec_A,
        spec_B=spec_B,
        spec_C=spec_C,
    )

    # passed_test() cannot make this assertion: the golden is all zeros, so its magnitude sits
    # under PCC_SIGNAL_FLOOR and the verdict falls back to the per-element tolerance, which is
    # atol=0.05 on both formats here -- a lane returning 0.01 would pass. So assert the zero
    # directly. It is exact by construction and not by tolerance: every b and c is a power of
    # two and value is 2.0, so value*b*c is representable in both formats and a = -value*b*c
    # cancels it bit for bit, leaving a zero as the only admissible result.
    #
    # `!= 0` and not a bitwise test, because -0.0 == 0.0: the sign of the cancelled zero is the
    # arch split this variant deliberately does not judge -- SFPMAD flushes to +0 on Wormhole
    # where Blackhole preserves it -- and pinning it needs the bitwise comparator #52938 tracks.
    nonzero = int((res_tensor != 0).sum())
    assert nonzero == 0, (
        f"a + value*b*c with a = -value*b*c must cancel to zero, but {nonzero} of "
        f"{len(res_tensor)} lanes are non-zero (largest magnitude "
        f"{float(res_tensor.abs().max())}) — the product or the add is losing bits the "
        "operands were chosen to keep exact"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-magnitude block-float blocks, ternary side
#
# The stimulus is sfpu_domains.block_spread_spec(), the same builder and the same decades the
# unary half drives; the reasoning for its shape is there and in that half's header. The three
# operands differ only in *seed*, which makes the specs distinguishable objects -- the pattern
# is identical across them, which is what keeps a + value*b*c exactly reproducible.
#
# addcmul is the only ternary op the suite drives on Bfp8_b and a good subject: no pole and no
# knee, so a mixed block is the only thing the variant asks about, and its three operands each
# carry their own shared exponent -- which is why the spread is driven on all three. Quantization
# is per operand, so pinning two to a narrow range would leave two thirds of the question
# untested, and there is no second failure class here to keep separate.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Bfp8_b], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.SfpuAddcmul,
    decades=runtime(list(BLOCK_SPREAD_DECADES)),
)
def test_sfpu_ternary_block_spread(formats, dest_acc, mathop, decades):
    """addcmul on three Bfp8_b operands whose blocks span the shared exponent."""
    _run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        spec_A=block_spread_spec(decades, seed=0),
        spec_B=block_spread_spec(decades, seed=1),
        spec_C=block_spread_spec(decades, seed=2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TTNNWhere
#
# where(cond, t, f) is a select, not an arithmetic op: the result is one of the two data
# verbatim. That makes the driver below shared rather than copied -- the three variants differ
# only in the three tensors they hand it, and a third transcription of the TestConfig block is
# how they would come to disagree about unpack_to_dest or the comparator.
# ─────────────────────────────────────────────────────────────────────────────


def _skip_unsupported_where(formats, dest_acc):
    """The two (format, dest_acc) pairs the where kernel does not support."""
    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")


def _run_ttnn_where(formats, dest_acc, mathop, cond, true_value, false_value):
    """Drive the where kernel on three prepared tensors and assert against WhereGolden.

    The formats are handed to the golden as well as to the kernel, which is what turns on the
    pack-path modelling: a NaN selected into a pipeline too narrow to hold one arrives as a
    signed infinity. Nothing changes for finite data, where the substitution never fires.
    """
    tile_count = cond.numel() // (DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM)

    golden_generator = get_golden_generator(WhereGolden)
    golden = golden_generator(
        cond,
        true_value,
        false_value,
        input_format=formats.input_format,
        output_format=formats.output_format,
        dest_acc=dest_acc,
    )

    configuration = TestConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        templates=[
            SFPU_TERNARY_OP(mathop),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(ApproximationMode.No),
            DISABLE_SRC_ZERO_FLAG(True),
            DEST_SYNC(),
        ],
        runtimes=[NUM_BLOCKS(tile_count), NUM_TILES_IN_BLOCK(1)],
        variant_stimuli=StimuliConfig(
            cond.flatten(),
            formats.input_format,
            true_value.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
            buffer_C=false_value.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_count,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[: len(golden)]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    # Int32 is compared as bfloat16: the where kernel moves raw bits, so the comparison only
    # has to be exact and reinterpreting both sides the same way keeps it so.
    dtype = (
        format_dict[formats.output_format]
        if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
        else torch.bfloat16
    )
    golden_tensor = torch.tensor(golden, dtype=dtype).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=dtype).flatten()

    assert torch_equal_nan(golden_tensor, res_tensor), "Assert against golden failed"


# The condition for the `mixed` variant, mixed *by construction* on every format.
#
# It used to be `uniform(0.0, 1.0)`, which produced 0 exact zeros in 4096 on Float32 and 20 on
# Float16_b (and those only because bf16 rounds small draws down) -- so `mixed` was bit-for-bit
# `all_ones` there, and the false branch was never taken by the one variant meant to take it.
# Int32's integer narrowing gave a genuine ~50/50 and hid it. `uniform(intervals=[(0.0, 0.0),
# (0.5, 1.0)])` looks like the fix and is not: an interval is chosen by *length*, so a
# zero-length one is never chosen. Hence a callable.
#
# The non-zero half is a spread of small magnitudes and both signs, not a constant 1.0: `where`
# selects on `cond != 0`, so -1 and 2 must both take the true branch and a constant asserts
# neither. Small integers because the same tensor runs on Int32, where anything in (0, 1) would
# quantize to zero and turn the spread back into the bug it replaces.
_WHERE_MIXED_NONZERO = (1.0, 2.0, -1.0, -2.0)


def _where_mixed_condition(size, dtype, generator):
    """Half zeros exactly, the rest a signed spread of small non-zero magnitudes."""
    # Half the slots are built as zero and half as the spread, then shuffled: the callable is
    # invoked once per face, so this makes the split exactly half on *every* face. Drawing each
    # element independently -- bucketing one uniform draw, which is what this used to do -- puts
    # the split at half only in expectation, and a face is then free to come out any ratio at
    # all, up to and including the all-true tensor this variant exists to avoid.
    half = size // 2
    spread = torch.tensor(_WHERE_MIXED_NONZERO, dtype=torch.float32)
    values = torch.zeros(size, dtype=torch.float32)
    values[half:] = spread.repeat((size - half + len(spread) - 1) // len(spread))[
        : size - half
    ]
    return values[torch.randperm(size, generator=generator)].to(dtype)


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.TTNNWhere,
    test_case=["mixed", "all_ones", "all_zeros"],
)
def test_ttnn_where(
    formats,
    dest_acc,
    mathop,
    test_case,
):
    _skip_unsupported_where(formats, dest_acc)

    # 64x64 = 2x2 tiles: exercises the multi-tile block loop in sfpu_ternary_test.cpp.
    input_dimensions = [64, 64]
    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    cond_spec = (
        StimuliSpec(distribution=_where_mixed_condition, seed=0)
        if test_case == "mixed"
        else sfpu_false_spec
    )
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=cond_spec,
        spec_B=sfpu_false_spec,
    )

    src_C, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    # Modify the condition tensor based on test case
    if test_case == "all_ones":
        src_A = torch.ones_like(src_A)
    elif test_case == "all_zeros":
        src_A = torch.zeros_like(src_A)
    else:
        # The failure this variant was in is silent -- an all-true condition passes against an
        # all-true golden -- so assert the stimulus rather than trusting the spec to have
        # produced it. _where_mixed_condition splits every face exactly in half and none of the
        # three formats perturb its values, so the bound is the exact one: anything else means
        # the spec stopped reaching the tensor.
        frac_true = float((src_A.flatten().to(torch.float32) != 0.0).float().mean())
        assert frac_true == 0.5, (
            f"the 'mixed' condition is {frac_true:.1%} true, not the half it is built to be "
            "-- this variant is drifting towards a duplicate of all_ones/all_zeros"
        )

    _run_ttnn_where(formats, dest_acc, mathop, src_A, src_B, src_C)


# MCW test with dynamic format sweeping like main test
# Use same input/output format - no mixing
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.TTNNWhere,
)
def test_ttnn_where_mcw(
    formats,
    dest_acc,
    mathop,
):
    # Multi-tile tensor dimensions (2x2 tiles of 32x32).
    height = 64
    width = 64

    _skip_unsupported_where(formats, dest_acc)

    # Create alternating pattern for condition (0, 1, 0, 1, ...)
    pattern = torch.arange(height * width) % 2
    C = pattern.view(height, width).to(format_dict[formats.input_format])

    # Set specific values for true and false tensors
    T = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 2
    F = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 11

    _run_ttnn_where(formats, dest_acc, mathop, C, T, F)


# ─────────────────────────────────────────────────────────────────────────────
# IEEE specials through where, one operand at a time
#
# where selects rather than computes, so its three operands ask two different questions. On the
# *condition*: is `cond != 0` still right at +/-inf, NaN or -0.0? The predicate is built on
# SFPSETCC, whose contract holds only "provided that VC is neither negative zero nor any kind of
# NaN" -- both outside what the primitive promises, which is why driving them is the point. On
# *true/false*: does a special survive being selected and packed? A raw LO16 move should carry
# anything, and the interesting case is a NaN through a pipeline that substitutes an infinity.
#
# The non-probed operands are constants and the condition is pinned to the branch under test, so
# every probe value is definitely the one selected -- an alternating condition would leave half
# the list unobserved and still pass.
#
# Values come from edge_values() like every other cat-B sweep, so where enrols through
# TERNARY_SPECIALS_READY_OPS and the -0.0 probe is dropped by negative_zero_delivered() on the
# pipelines that flatten it.
#
# Two variants, not one. A -0.0 *condition* is the only where probe that diverges, and the
# driver asserts a whole tile at once, so it gets its own variant -- otherwise its xfail would
# stand for every special in the same tile and hide a regression on any of them.
# ─────────────────────────────────────────────────────────────────────────────

# Constants for the two operands not under test: exact in every format here, and distinct from
# each other so the result says which branch was taken.
_WHERE_TRUE_CONST = 2.0
_WHERE_FALSE_CONST = 11.0


def _is_negative_zero(value):
    return value == 0.0 and math.copysign(1.0, value) < 0.0


def _where_const_tile(value, fmt, dimensions):
    return torch.full(dimensions, value, dtype=format_dict[fmt])


def _where_probe_tile(values, fmt, dimensions):
    """*values* tiled across the whole tensor.

    Tiled rather than written at the head with a filler tail, for the reason edge_spec() gives:
    a probe confined to the first few lanes hides any lane-position-dependent defect, and the
    verdict is then mostly a statement about the filler. Tiling also removes the question of
    what the filler should be -- a zero in the *condition* operand would silently add the
    false-branch case to a variant driving the true one.
    """
    total = dimensions[0] * dimensions[1]
    reps = -(-total // len(values))
    flat = torch.tensor((list(values) * reps)[:total], dtype=torch.float32).to(
        format_dict[fmt]
    )
    return flat.view(*dimensions)


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.TTNNWhere,
    # runtime(): the operand axis changes the three tensors and nothing about the kernel.
    operand=runtime(list(_TERNARY_OPERANDS)),
)
def test_ttnn_where_specials(formats, dest_acc, mathop, operand):
    """Drive IEEE specials into one where operand, with the other two held at constants."""
    _skip_unsupported_where(formats, dest_acc)

    specials = _ternary_cat_b_enabled(mathop, formats, dest_acc)
    # Nothing is filtered: where has no pole and no knee, so edge_values() is cat B alone here
    # and every special it returns -- the NaN, both infinities and both zeros -- is a probe this
    # variant wants. Stated because the finite/non-finite partition _ternary_edge_class_values()
    # uses looks like it belongs here too, and applying it would silently drop the NaN and the
    # infinities from a variant whose whole subject is them.
    vals = edge_values(
        mathop,
        formats.input_format,
        formats.output_format,
        operand=operand,
        specials=specials,
        dest_acc=dest_acc,
    )

    if operand == Operand.A:
        # The -0.0 condition is driven by test_ttnn_where_negative_zero_condition instead, and
        # has to leave here: _run_ttnn_where makes one aggregate torch_equal_nan assert over the
        # whole tile, so the xfail that divergence needs would absorb a future regression on the
        # NaN or either infinity into the same expected failure. One failure class per variant,
        # the way the binary suite splits on (op, edge_class). Dropped unconditionally rather
        # than only on the delivering cells, so this variant drives the same condition probe
        # everywhere and its verdict does not change shape with the pipeline.
        #
        # Before the producer guard below, not after: the guard's job is to leave the compile
        # producer with a non-empty list, and a filter applied downstream of it could empty the
        # list again and skip the build the other two variants share.
        vals = [v for v in vals if not _is_negative_zero(v)]

    if not vals and TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        # `operand` is runtime() here too, so the same starvation is available. It does not
        # happen today: cat B is where's only source of probes and is a per-pipeline fact, so
        # every operand empties on the same cells and the skip is uniform. The guard is here so
        # that a per-operand registry entry later -- a knee on the condition, say -- gains
        # coverage rather than a timeout.
        vals = _producer_probe_values(mathop, formats, dest_acc, specials)

    if not vals:
        pytest.skip(
            reason=f"cat B is off for {mathop.name} on this pipeline, and where has no "
            "other edge"
        )

    dimensions = (64, 64)
    tiles = {
        Operand.A: _where_const_tile(1.0, formats.input_format, dimensions),
        Operand.B: _where_const_tile(
            _WHERE_TRUE_CONST, formats.input_format, dimensions
        ),
        Operand.C: _where_const_tile(
            _WHERE_FALSE_CONST, formats.input_format, dimensions
        ),
    }
    if operand == Operand.C:
        # Probing the false branch, so the condition has to select it.
        tiles[Operand.A] = _where_const_tile(0.0, formats.input_format, dimensions)
    tiles[operand] = _where_probe_tile(vals, formats.input_format, dimensions)

    _run_ttnn_where(
        formats,
        dest_acc,
        mathop,
        tiles[Operand.A],
        tiles[Operand.B],
        tiles[Operand.C],
    )


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.TTNNWhere,
)
def test_ttnn_where_negative_zero_condition(request, formats, dest_acc, mathop):
    """A -0.0 condition on its own, because it is the one where probe that diverges.

    Split out of test_ttnn_where_specials rather than xfailed inside it: _run_ttnn_where makes
    one aggregate comparison over the whole tile, so an xfail covering this divergence would
    also cover a future NaN or infinity regression on the same condition operand and report it
    as the expected failure. Same reason the binary edge sweep keys its xfails on
    (op, edge_class) instead of on op.

    Runs on every cell, including the ones that flatten the probe to +0.0, where it passes
    vacuously -- the unary suite's Sign and Heaviside entries are scoped the same way. That
    keeps the xfail derived from negative_zero_delivered() rather than from a listed set of
    cells, so a cell drifting in or out of delivery reports a behaviour change instead of
    leaving a stale table behind.
    """
    _skip_unsupported_where(formats, dest_acc)

    # A -0.0 *condition* selects the true branch on the unpack-to-dest path, where a real -0.0
    # reaches the LREG; `-0.0 == 0` makes it the false branch. Outside the documented contract
    # rather than a hardware fault: SFPSETCC is specified only for inputs that are not negative
    # zero (tt-isa-documentation WormholeB0/.../VectorUnit.md, identically on Blackhole), which
    # is the same caveat that scopes Sign's and Heaviside's divergences in the unary suite -- and
    # to the same set of cells, since negative_zero_delivered() is what decides both. Measured on
    # a Blackhole p150: the only divergent lane, on the only cell that delivers the probe.
    if negative_zero_delivered(formats.input_format, dest_acc):
        request.node.add_marker(
            pytest.mark.xfail(
                reason="where(-0.0, t, f) returns t; -0.0 == 0 makes it f. Outside the "
                "documented contract: SFPSETCC is specified only for inputs that are not "
                "negative zero. Same caveat and same unpack-to-dest scoping as Sign and "
                "Heaviside.",
                strict=False,
            )
        )

    dimensions = (64, 64)
    _run_ttnn_where(
        formats,
        dest_acc,
        mathop,
        _where_probe_tile([-0.0], formats.input_format, dimensions),
        _where_const_tile(_WHERE_TRUE_CONST, formats.input_format, dimensions),
        _where_const_tile(_WHERE_FALSE_CONST, formats.input_format, dimensions),
    )
