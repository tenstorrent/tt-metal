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
    """*value* as the raw fp32 bit pattern SFPU_TERNARY_SCALAR carries."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


_SCALAR_VALUE = 2.0
_SCALAR_VALUE_BITS = _scalar_bits(_SCALAR_VALUE)


# Helper check function
def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


def _ternary_default_specs(mathop, input_format):
    """Per-operand defaults for *mathop*: its registered domain, else the built-in one.

    Callers of _run_sfpu_ternary override any operand to reach an edge the defaults
    exclude (e.g. the c -> 0 pole that addcdiv and snake_beta pin away from).
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
    """Drive one ternary variant; returns (src_A, golden_tensor, res_tensor).

    *unspecified_nonfinite_sign* compares a non-finite result by magnitude only, for the one
    case where the sign is genuinely unspecified: a NaN the kernel emitted, packed as a signed
    infinity through a narrowing pipeline, on Wormhole. Per lane, from the golden's mask.
    """
    # The specs below carry no seed; seed here so a near-tolerance variant cannot pass by luck.
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

    # input_format and dest_acc turn on the unpack/Dest/pack modelling: sub-ULP on a finite
    # value, decisive on a non-finite one.
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
    # Asked of the return value, not the flag: under --compile-producer the golden stub returns
    # a bare tensor whatever it is asked for, so keying off the flag would raise there.
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
        # Only on lanes that held an emitted NaN *and* where both sides are non-finite, so
        # this excuses one bit on the lanes the ISA declines to pin and nothing else.
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
# The random sweep holds c away from zero for the ops that divide by it, so the pole is
# unreachable by construction. This drives it, and the two operands the sweep never touched.
# *operand* says which of a, b, c carries the probe, the other two keeping their random
# domains, where pinning two would pair their lists *index-wise* rather than crossing them.
# *edge_class* gives one failure class per variant, _run_sfpu_ternary asserting a whole tile
# at once: unsplit, addcdiv's c = 0 shares a marker with c = inf, and its c = NaN divergence
# covers the measured-green c = +/-inf lanes.
# ─────────────────────────────────────────────────────────────────────────────

_TERNARY_EDGE_OPS = [
    MathOperation.SfpuAddcdiv,
    MathOperation.SfpuAddcmul,
    MathOperation.SfpuLerp,
    MathOperation.SfpuSnakeBeta,
]

# Ops that divide by c, and therefore need a numerator held away from zero: c = 0 with an
# unconstrained numerator would mix the pole (every element ±inf) with the 0/0 indeterminate
# form, which the binary suite covers as a class of its own.
_TERNARY_DIVIDES_BY_C = frozenset(
    {MathOperation.SfpuAddcdiv, MathOperation.SfpuSnakeBeta}
)

# |x| >= 0.5 on both a and b, keeping both numerators off zero -- addcdiv's is value * b,
# snake_beta's sin(b*a)^2. Two specs differing only in seed: one spec shared by both operands
# would make every variant run a == b, hiding a kernel that reads the wrong operand.
_TERNARY_NONZERO_A = StimuliSpec.uniform(intervals=[(-1.0, -0.5), (0.5, 1.0)], seed=0)
_TERNARY_NONZERO_B = StimuliSpec.uniform(intervals=[(-1.0, -0.5), (0.5, 1.0)], seed=1)

_TERNARY_EDGE_CLASS_POLE = "pole"
_TERNARY_EDGE_CLASS_NAN = "nan_in"
_TERNARY_EDGE_CLASS_INF = "inf_in"

#: The two cat-B classes, for the gates that apply to a non-finite operand however it is shaped.
_TERNARY_SPECIALS_CLASSES = (_TERNARY_EDGE_CLASS_NAN, _TERNARY_EDGE_CLASS_INF)

# Order matters only for --compile-producer, where the first entry builds the ELF all classes
# share, so the class most likely to be non-empty goes first.
_TERNARY_EDGE_CLASSES = (_TERNARY_EDGE_CLASS_POLE,) + _TERNARY_SPECIALS_CLASSES

_TERNARY_OPERANDS = (Operand.A, Operand.B, Operand.C)


def _ternary_cat_b_enabled(mathop, formats, dest_acc):
    """Two gates: the *golden* defines an answer, and the *pipeline* delivers the stimulus
    intact. Neither implies the other."""
    return mathop in TERNARY_SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )


def _ternary_edge_class_values(
    mathop, formats, operand, edge_class, dest_acc, specials
):
    """The probe values of *edge_class* for (*mathop*, *operand*) on this pipeline.

    One edge_values() call partitioned three ways, so the classes cannot disagree about which
    value belongs where, and the partition is total -- no probe silently leaves the sweep.
    """
    vals = edge_values(
        mathop,
        formats.input_format,
        formats.output_format,
        operand=operand,
        specials=specials,
        dest_acc=dest_acc,
    )
    if edge_class == _TERNARY_EDGE_CLASS_NAN:
        return [v for v in vals if math.isnan(v)]
    if edge_class == _TERNARY_EDGE_CLASS_INF:
        return [v for v in vals if math.isinf(v)]
    return [v for v in vals if math.isfinite(v)]


def _producer_probe_values(mathop, formats, dest_acc, specials):
    """Any non-empty probe list for this compile key, or [] if the op has no edge at all here.

    For --compile-producer, where one item per compile key builds the ELF every other variant
    shares -- a skip there presents as TENSIX TIMED OUT. Drops *both* axes: on Float16_b /
    dest_acc=Yes cat B is off, leaving A and B with no edge while C still has its pole.
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


# The cells this sweep's format axis reaches, so the divergence sets below derive from the
# same gates the stimulus does rather than being transcribed.
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


def _software_rne_path(_in_fmt, _out_fmt, dest_acc):
    """True where calculate_lerp() takes its `if constexpr (!is_fp32_dest_acc_en)` branch.

    Its only caller of float32_to_bf16_rne, where the divergence lives. Measured on an n300.
    """
    return dest_acc == DestAccumulation.No


# What driving the ternary specials found, once both goldens modelled the Dest write and the
# pack (10 cells on its own). Non-strict, and derived from the delivery gates so a cell
# drifting in or out shows up. The class is in the key because a NaN operand and an infinite
# one are separate verdicts on every op -- three diverge at the NaN, two at the infinity --
# and there are THREE CAUSES, NOT FIVE:
#
#   c = NaN through the reciprocal (addcdiv, snake_beta, operand C). SFPARECIP returns +0 for
#   1/NaN instead of propagating, so the result is `a` where the golden says NaN -- the
#   divergence unary Reciprocal already carries, on every specials-carrying cell.
#
#   An *infinity* reaching the sin (snake_beta, operands A and B): sin(b*a) has a square of
#   +inf against a golden NaN, and SFPLUTFP32 documents no NaN/inf handling. Scoped to the
#   cells where a NaN survives to L1, since where the pack substitutes an infinity the two
#   agree; a NaN argument comes back out of the polynomial as a NaN, so only inf_in is here.
#
#   a = NaN wrapping to an exact zero in the shared software RNE (lerp, operand A, Wormhole
#   only) -- in a *shared* conversion helper rather than in lerp. The wrap needs the NaN's
#   mantissa bits, so a = +/-inf agrees.
_TERNARY_EDGE_KNOWN_DIVERGENCES = {
    (MathOperation.SfpuAddcdiv, Operand.C, _TERNARY_EDGE_CLASS_NAN): _cat_b_cells(),
    (MathOperation.SfpuSnakeBeta, Operand.C, _TERNARY_EDGE_CLASS_NAN): _cat_b_cells(),
    (
        MathOperation.SfpuSnakeBeta,
        Operand.A,
        _TERNARY_EDGE_CLASS_INF,
    ): _cat_b_cells(nan_survives_to_l1),
    (
        MathOperation.SfpuSnakeBeta,
        Operand.B,
        _TERNARY_EDGE_CLASS_INF,
    ): _cat_b_cells(nan_survives_to_l1),
    (
        MathOperation.SfpuLerp,
        Operand.A,
        _TERNARY_EDGE_CLASS_NAN,
    ): _cat_b_cells(_software_rne_path),
}

# The arch each divergence was measured on; absent means every arch. The lerp wrap is
# Wormhole-only and *asserted* on Blackhole rather than assumed: float32_to_bf16_rne leaves
# the canonical 0x7fc00000 unchanged, so a Blackhole hit is a fresh failure, not an xfail.
_TERNARY_EDGE_ARCH_GATE = {
    (MathOperation.SfpuLerp, Operand.A, _TERNARY_EDGE_CLASS_NAN): (
        ChipArchitecture.WORMHOLE,
    ),
}

_LERP_RNE_WRAP_NOTE = (
    "float32_to_bf16_rne() has no non-finite guard: it rounds by adding 0x7fff + lsb to the "
    "raw fp32 bits, and on a NaN whose pattern is 0xffff8000 or above that add carries out of "
    "bit 31 and the mask leaves 0x00000000, so the NaN becomes an exact +0. The canonical "
    "0x7fc00000 is untouched by the same add, which is why this is Wormhole-only"
)

_RECIPROCAL_NAN_NOTE = (
    "the kernel's reciprocal returns +0 for 1/NaN instead of propagating, so the quotient "
    "vanishes and the result is `a`; the same divergence unary Reciprocal carries, through the "
    "same SFPARECIP composition. c = +/-inf agrees, so this is the NaN probe alone"
)

_SNAKE_BETA_SIN_NOTE = (
    "sin(b*a) at an infinite argument gives the kernel a value whose square is +inf, where "
    "torch gives NaN, so the result is +inf against a golden NaN. SFPLUTFP32 documents no "
    "NaN/inf handling, so this is an LLK decision with no ISA ruling. Only on the cells where "
    "a NaN survives to L1 -- the divergence is against the *golden's* NaN, so where the pack "
    "substitutes an infinity for it the two agree. A NaN operand does not take this path: it "
    "comes back out of the polynomial as a NaN and the golden agrees, measured"
)

_TERNARY_EDGE_REASON = {
    (
        MathOperation.SfpuAddcdiv,
        Operand.C,
        _TERNARY_EDGE_CLASS_NAN,
    ): f"addcdiv(a, b, NaN) returns a, not NaN ({_RECIPROCAL_NAN_NOTE}).",
    (
        MathOperation.SfpuSnakeBeta,
        Operand.C,
        _TERNARY_EDGE_CLASS_NAN,
    ): f"snake_beta(a, b, NaN) returns a, not NaN ({_RECIPROCAL_NAN_NOTE}).",
    (
        MathOperation.SfpuSnakeBeta,
        Operand.A,
        _TERNARY_EDGE_CLASS_INF,
    ): f"snake_beta(+/-inf, b, c): {_SNAKE_BETA_SIN_NOTE}.",
    (
        MathOperation.SfpuSnakeBeta,
        Operand.B,
        _TERNARY_EDGE_CLASS_INF,
    ): "As operand A: the infinity reaches the same sin through the b*a product.",
    (
        MathOperation.SfpuLerp,
        Operand.A,
        _TERNARY_EDGE_CLASS_NAN,
    ): "lerp(NaN, b, c) returns an exact 0 instead of a non-finite when c is a power of two "
    f"of magnitude >= 0.5, on the dest_acc=No path only ({_LERP_RNE_WRAP_NOTE}). Measured on "
    "an n300 with a = NaN, b = 1.0: c in 0.5, 1, 2, 4, 8, 16 all give 0, while every "
    "non-power-of-two c and every c below 0.5 give the packed infinity the golden expects, "
    "and a = +/-inf agrees throughout. Not lerp's defect: the wrap is in a conversion helper "
    "the binary ops share, so a fix belongs there.",
}

assert set(_TERNARY_EDGE_REASON) == set(_TERNARY_EDGE_KNOWN_DIVERGENCES), (
    "_TERNARY_EDGE_REASON and _TERNARY_EDGE_KNOWN_DIVERGENCES disagree on which "
    "(op, operand, edge_class) keys diverge: "
    f"{set(_TERNARY_EDGE_REASON) ^ set(_TERNARY_EDGE_KNOWN_DIVERGENCES)}"
)
assert all(
    edge_class in _TERNARY_EDGE_CLASSES
    for _op, _operand, edge_class in _TERNARY_EDGE_KNOWN_DIVERGENCES
), "a divergence keyed on an edge_class no variant runs is a dead xfail"
assert all(cells for cells in _TERNARY_EDGE_KNOWN_DIVERGENCES.values()), (
    "an (op, operand, edge_class) claiming a divergence with no cell to apply it to is a dead "
    "xfail"
)
assert set(_TERNARY_EDGE_ARCH_GATE) <= set(_TERNARY_EDGE_KNOWN_DIVERGENCES), (
    "_TERNARY_EDGE_ARCH_GATE names keys with no divergence to scope: "
    f"{set(_TERNARY_EDGE_ARCH_GATE) - set(_TERNARY_EDGE_KNOWN_DIVERGENCES)}"
)


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_TERNARY_EDGE_OPS,
    # runtime(): both axes select which values go into which operand tensor and nothing else,
    # so all nine share the one ELF the (op, formats, dest_acc) triple decides.
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

    # Marked before the stimulus is built, so an empty class still skips rather than xfailing.
    key = (mathop, operand, edge_class)
    reason = _TERNARY_EDGE_REASON.get(key)
    # The arch gate defaults to every arch, so an entry without one behaves as before.
    arch_ok = TestConfig.CHIP_ARCH in _TERNARY_EDGE_ARCH_GATE.get(
        key, (TestConfig.CHIP_ARCH,)
    )
    if (
        reason is not None
        and arch_ok
        and (formats.input_format, formats.output_format, dest_acc)
        in _TERNARY_EDGE_KNOWN_DIVERGENCES[key]
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
                if edge_class not in _TERNARY_SPECIALS_CLASSES or specials
                else " (cat B is off for this op or this pipeline)"
            )
        )

    # Keep the numerator off zero when the probed operand is the divisor, so the variant
    # asserts the pole rather than the 0/0 indeterminate form.
    guard = operand == Operand.C and mathop in _TERNARY_DIVIDES_BY_C
    specs = {
        Operand.A: _TERNARY_NONZERO_A if guard else None,
        Operand.B: _TERNARY_NONZERO_B if guard else None,
        Operand.C: None,
    }
    # cycle=True: the probed operand fills its face, so the verdict is not dominated by the
    # (0, random, random) triples a zero tail would create.
    specs[operand] = StimuliSpec.custom(values=vals, seed=0, cycle=True)

    # A narrowing pipeline turns an emitted NaN's sign into the observable result, and
    # Wormhole leaves that sign unspecified; which lanes is the golden's own mask.
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


# The addc multiplier: `value` is a compile-time template argument, and was not an axis at
# all before: 2.0 everywhere. At 0.0 both ops collapse to the identity in `a` -- a strong
# check that neither kernel reads the wrong Dst tile, since one returning `b` or `c` would
# pass every other variant here. 1.0 removes the multiply; -2.0 flips a sign. One format
# column and one dest_acc value, the scalar being orthogonal to both, and scoped to the
# ops' ordinary domains so the identity at value = 0 is a clean assertion. Its interaction
# with addcdiv's pole was measured (NaN on Blackhole, matching the golden) but not driven.

_SCALAR_PROBES = (0.0, 1.0, -2.0)

# The two ops that read the scalar at all. lerp's weight is operand C and snake_beta has no
# multiplier, so SFPU_TERNARY_SCALAR is a dead template argument for both.
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

    # Exactly, not within a tolerance. passed_test() already compared against the golden; this
    # asserts it against the *stimulus*, so a golden with the same mistake could not hide it.
    expected = src_A.flatten().to(format_dict[formats.output_format])[: len(res_tensor)]
    mismatched = int((res_tensor != expected).sum())
    assert mismatched == 0, (
        f"{mathop.name} with value = 0 must return `a` bit for bit, but {mismatched} of "
        f"{len(res_tensor)} lanes differ — the kernel is reading an operand it should be "
        "multiplying away"
    )


# addcmul's cancellation edge. addcmul is smooth in all three operands, so the sweep above
# gives it only cat B. What it does have is exact cancellation: at a = -value * b * c the
# result must be zero. An explicit triple rather than a StimuliSpec, the relation being
# *between* the operands; every b and c is a power of two, so the product is exact in
# every format here.

# (b, c) pairs; a is derived as -_SCALAR_VALUE * b * c. Powers of two, both signs, three
# decades of magnitude, so the product is exact across the exponent range.
_ADDCMUL_CANCELLATION_BC = (
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
    (0.5, 0.5),
    (-0.25, 8.0),
    (64.0, 0.125),
    (-16.0, -4.0),
    (0.0, 1.0),  # a = -0.0 exactly: the signed-zero case of the same relation
    (1.0, 0.0),
)


def _addcmul_cancellation_specs():
    """(spec_A, spec_B, spec_C) whose lanes satisfy a + value*b*c == 0 exactly."""
    b = [b for b, _ in _ADDCMUL_CANCELLATION_BC]
    c = [c for _, c in _ADDCMUL_CANCELLATION_BC]
    a = [-_SCALAR_VALUE * bv * cv for bv, cv in _ADDCMUL_CANCELLATION_BC]
    # cycle=True on all three so the relation holds in every lane: a zero tail would leave
    # 0 + value*0*0 == 0 across ~96% of the tensor.
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
    # under PCC_SIGNAL_FLOOR and the verdict falls back to atol=0.05, where a lane returning
    # 0.01 would pass. `!= 0` and not a bitwise test, because the cancelled zero's sign is an
    # arch split (SFPMAD flushes on Wormhole) that needs the comparator tt-metal#52938 tracks.
    nonzero = int((res_tensor != 0).sum())
    assert nonzero == 0, (
        f"a + value*b*c with a = -value*b*c must cancel to zero, but {nonzero} of "
        f"{len(res_tensor)} lanes are non-zero (largest magnitude "
        f"{float(res_tensor.abs().max())}) — the product or the add is losing bits the "
        "operands were chosen to keep exact"
    )


# Mixed-magnitude block-float blocks, ternary side. The stimulus is
# sfpu_domains.block_spread_spec(), shared with the unary half; the reasoning for its
# shape is there. addcmul is the only ternary op the suite drives on Bfp8_b and a good
# subject: no pole and no knee, so a mixed block is the only thing the variant asks about,
# and each of its three operands carries its own shared exponent -- hence the spread on
# all three.


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


# TTNNWhere: where(cond, t, f) is a select: the result is one of the two data verbatim.
# The driver below is shared rather than copied, the variants differing only in the three
# tensors they hand it.


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

    The formats go to the golden too, turning on the pack-path modelling: a NaN selected into
    a narrowing pipeline arrives as a signed infinity. Nothing changes for finite data.
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

    # Int32 is compared as bfloat16: the kernel moves raw bits, so reinterpreting both sides
    # the same way keeps the comparison exact.
    dtype = (
        format_dict[formats.output_format]
        if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
        else torch.bfloat16
    )
    golden_tensor = torch.tensor(golden, dtype=dtype).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=dtype).flatten()

    assert torch_equal_nan(golden_tensor, res_tensor), "Assert against golden failed"


# The condition for the `mixed` variant, mixed *by construction* on every format. It used to be
# `uniform(0.0, 1.0)`, which gave 0 exact zeros in 4096 on Float32, so `mixed` was bit-for-bit
# `all_ones` there and only Int32's narrowing hid it.
# `uniform(intervals=[(0.0, 0.0), (0.5, 1.0)])` looks like the fix and is not -- an interval is
# chosen by *length*, so a zero-length one is never chosen. The non-zero half is a signed
# spread, since `where` selects on `cond != 0`, and small integers because the same tensor runs
# on Int32, where anything in (0, 1) would quantize to zero and restore the bug.
_WHERE_MIXED_NONZERO = (1.0, 2.0, -1.0, -2.0)


def _where_mixed_condition(size, dtype, generator):
    """Half zeros exactly, the rest a signed spread of small non-zero magnitudes."""
    # Built half and half, then shuffled: the callable runs once per face, so the split is
    # exactly half on *every* face. Bucketing one uniform draw per element puts it at half
    # only in expectation, leaving a face free to come out any ratio at all.
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
        # The failure is silent -- an all-true condition passes against an all-true golden --
        # so assert the stimulus. The split is exact, so the bound can be.
        frac_true = float((src_A.flatten().to(torch.float32) != 0.0).float().mean())
        assert frac_true == 0.5, (
            f"the 'mixed' condition is {frac_true:.1%} true, not the half it is built to be "
            "-- this variant is drifting towards a duplicate of all_ones/all_zeros"
        )

    _run_ttnn_where(formats, dest_acc, mathop, src_A, src_B, src_C)


# MCW test: the main test's format sweep, input format == output format.
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
# Two questions. On the *condition*: is `cond != 0` still right at +/-inf, NaN or -0.0? The
# predicate is SFPSETCC, whose contract holds only "provided that VC is neither negative zero
# nor any kind of NaN". On *true/false*: does a special survive being selected and packed? The
# non-probed operands are constants and the condition is pinned to the branch under test. Two
# variants because a -0.0 condition is the only where probe that diverges, and the driver
# asserts a whole tile at once.
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
    """*values* tiled across the whole tensor, for the reason edge_spec() gives -- and because
    a zero filler in the *condition* operand would silently add the false-branch case.
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
    # Unpartitioned: where has no pole and no knee, so edge_values() is cat B alone and every
    # special it returns is wanted. Applying _ternary_edge_class_values()' split here would
    # drop the NaN and the infinities from the variant whose whole subject is them.
    vals = edge_values(
        mathop,
        formats.input_format,
        formats.output_format,
        operand=operand,
        specials=specials,
        dest_acc=dest_acc,
    )

    if operand == Operand.A:
        # The -0.0 condition goes to test_ttnn_where_negative_zero_condition instead, since
        # the xfail it needs would absorb a regression on the NaN or either infinity in the
        # same tile. Dropped unconditionally, and before the producer guard below -- a filter
        # downstream of it could empty the list and skip the build the others share.
        vals = [v for v in vals if not _is_negative_zero(v)]

    if not vals and TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        # Inert today -- cat B is where's only probe source and empties on the same cells for
        # every operand -- but a per-operand registry entry later would hit the starvation.
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

    Runs on every cell, including the ones that flatten the probe and pass vacuously, which
    keeps the xfail derived from negative_zero_delivered() rather than listed."""
    _skip_unsupported_where(formats, dest_acc)

    # A -0.0 condition selects the true branch where a real -0.0 reaches the LREG; `-0.0 == 0`
    # makes it the false branch. Outside the documented contract rather than a hardware fault:
    # SFPSETCC is specified only for inputs that are not negative zero, the caveat that scopes
    # Sign and Heaviside. Measured on a Blackhole p150.
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
