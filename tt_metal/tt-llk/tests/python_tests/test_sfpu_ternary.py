# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from helpers.bf16_ties import (
    BF16_FRAC_BITS,
    BF16_SIG_ONE,
    assert_is_bf16_tie,
    fp32_bits,
)
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
from helpers.param_config import input_output_formats, parametrize
from helpers.sfpu_domains import (
    _OP_DOMAIN_REGISTRY,
    Operand,
    edge_spec,
    exclude_undefined_pair,
    for_op,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    DISABLE_SRC_ZERO_FLAG,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    SFPU_TERNARY_OP,
    SFPU_TERNARY_SCALAR,
)
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    MAX_TILE_ELEMENTS,
)
from helpers.utils import passed_test

_SCALAR_VALUE = 2.0
_SCALAR_VALUE_BITS = fp32_bits(_SCALAR_VALUE)


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
    operands=None,
    exact=False,
):
    """Drive one ternary variant.

    *operands* = (a, b, c) replaces the generated stimuli with explicit tensors of
    *input_dimensions* elements each; *exact* then compares bit for bit instead of within
    tolerance, which only a kernel that narrows with round-to-nearest-even can pass.
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

    if operands is not None:
        explicit = []
        for generated, given in zip((src_A, src_B, src_C), operands):
            assert (
                given.numel() == generated.numel()
            ), "each explicit operand must fill the whole buffer"
            explicit.append(given.to(generated.dtype).reshape(generated.shape))
        src_A, src_B, src_C = explicit

    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden = generate_golden(
        mathop,
        src_A,
        src_B,
        src_C,
        _SCALAR_VALUE_BITS,
        formats.output_format,
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

    if exact:
        # max_ulp=0 is the bit-exact gate: zero representable steps between result and golden.
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format, max_ulp=0
        ), "Result is not bit-identical to the golden"
        return

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


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
# Deliberate edge values on the third operand
#
# The random sweep holds c away from zero for the ops that divide by it, so the pole is
# unreachable by construction; these variants drive it, via each op's registered edge metadata.
# Only C gets edge values: A and B keep their random domains, since the divisor is the
# interesting operand and pinning all three would test one point rather than a spread.
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


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_TERNARY_EDGE_OPS,
)
def test_sfpu_ternary_edges(formats, dest_acc, mathop):
    """Drive each ternary op's operand-C pole or knee, where it has one."""
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    spec_C = edge_spec(
        mathop,
        formats.input_format,
        formats.output_format,
        operand=Operand.C,
        dest_acc=dest_acc,
    )
    if spec_C is None:
        # addcmul: c is a multiplicand, so there is no pole or knee for a probe to reach.
        pytest.skip(
            reason=f"{mathop.name} has no operand-C edge (no pole, no knee) for this "
            "pipeline"
        )

    # Keep the numerator off zero so the variant asserts the pole, not 0/0.
    nonzero = mathop in _TERNARY_DIVIDES_BY_C
    _run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        spec_A=_TERNARY_NONZERO_A if nonzero else None,
        spec_B=_TERNARY_NONZERO_B if nonzero else None,
        spec_C=spec_C,
    )


# ─────────────────────────────────────────────────────────────────────────────
# bf16 round-to-nearest-even narrowing. lerp and addcdiv round their fp32 result into a bf16
# Dest with RNE; the default tolerance (0.05) cannot tell RNE from truncation or from a miswired
# rounding bias, so both are pinned bit for bit on exact ties. addcdiv's divisor goes through
# the reciprocal iteration, which only a power-of-two divisor survives exactly; with c = 1.0
# and the module's value of 2.0, value * b / c stays an exact power of two and a tie can be
# formed. A non-power-of-two divisor rules one out.
# ─────────────────────────────────────────────────────────────────────────────

_LERP_TIE_WEIGHT = 0.5
# addcdiv: c = 1.0 so sfpu_reciprocal_iter<2>(c) is exactly 1.0 (Newton on 1.0 is a fixed
# point under SFPMAD's round-to-nearest-even), and b = 2^-9 so value * b / c = 2^-8, half a
# bf16 ULP of [1, 2).
_ADDCDIV_TIE_DIVISOR = 1.0
_ADDCDIV_TIE_ADDEND = math.ldexp(1.0, -(BF16_FRAC_BITS + 2))


def _lerp_tie_operands(count, seed=0):
    """(a, b, c) bf16 tensors of *count* lanes with a + c * (b - a) an exact fp32 tie.

    b is a's upper bf16 neighbour and c = 0.5, so the exact result is a + ULP/2: b - a and the
    product are powers of two and the sum needs nine significand bits, all exact in fp32 for
    the kernel's SFPMAD and for torch alike. Random signs and exponents put about half the
    lanes on an even LSB, the only ones where round-to-even and round-half-away disagree.
    """
    # Seeded torch draws: the same lanes every run, so a failure is reproducible.
    gen = torch.Generator().manual_seed(seed)
    exps = torch.randint(-20, 21, (count,), generator=gen).tolist()
    sigs = torch.randint(0, BF16_SIG_ONE, (count,), generator=gen).tolist()
    signs = (torch.randint(0, 2, (count,), generator=gen) * 2 - 1).tolist()
    a, b = [], []
    for exp, sig, sign in zip(exps, sigs, signs):
        sig += BF16_SIG_ONE
        a.append(sign * math.ldexp(sig, exp - BF16_FRAC_BITS))
        b.append(sign * math.ldexp(sig + 1, exp - BF16_FRAC_BITS))
    # Self-check on the host: every lane must be an exact tie, else the test proves nothing.
    for x, y in zip(a, b):
        assert_is_bf16_tie(x + _LERP_TIE_WEIGHT * (y - x), f"lerp({x}, {y}, 0.5)")
    return (
        torch.tensor(a, dtype=torch.bfloat16),
        torch.tensor(b, dtype=torch.bfloat16),
        torch.full((count,), _LERP_TIE_WEIGHT, dtype=torch.bfloat16),
    )


def _addcdiv_tie_operands(count):
    """(a, b, c) bf16 tensors of *count* lanes with a + value * b / c an exact fp32 tie.

    a = +/-(1 + m / 128) for every m, b = +/-2^-9 with a's sign and c = 1.0, so with the
    module's value of 2.0 the exact result is a +/- half a bf16 ULP of [1, 2): the reciprocal
    of 1.0 is exact, the product is a power of two and the sum needs nine significand bits,
    all exact in fp32 for the kernel's SFPMAD chain and for torch alike. Both parities of the
    bf16 LSB occur; m = 127 is the carry case and rounds to +/-2.0.
    """
    assert _SCALAR_VALUE == 2.0, "the addend is sized for value = 2.0"
    lanes = [
        (sign * (1.0 + m / BF16_SIG_ONE), sign * _ADDCDIV_TIE_ADDEND)
        for sign in (1.0, -1.0)
        for m in range(BF16_SIG_ONE)
    ]
    # Self-check on the host: every lane must be an exact tie, else the test proves nothing.
    for x, y in lanes:
        assert_is_bf16_tie(
            x + _SCALAR_VALUE * y / _ADDCDIV_TIE_DIVISOR,
            f"addcdiv({x}, {y}, {_ADDCDIV_TIE_DIVISOR})",
        )
    return (
        torch.tensor(
            [lanes[i % len(lanes)][0] for i in range(count)], dtype=torch.bfloat16
        ),
        torch.tensor(
            [lanes[i % len(lanes)][1] for i in range(count)], dtype=torch.bfloat16
        ),
        torch.full((count,), _ADDCDIV_TIE_DIVISOR, dtype=torch.bfloat16),
    )


_TIE_OPERANDS = {
    MathOperation.SfpuLerp: _lerp_tie_operands,
    MathOperation.SfpuAddcdiv: _addcdiv_tie_operands,
}


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    mathop=list(_TIE_OPERANDS),
)
def test_sfpu_ternary_bf16_rne_ties(formats, mathop):
    """Exact bf16 ties through lerp's and addcdiv's narrowing must round to even, bit for bit.

    Guards the RNE helper the kernels share with the binary and scalar SFPU ops: a truncating
    store or SFPSTOCHRND (ties away from zero on Blackhole) fails half the lanes. For addcdiv
    it also pins sfpu_reciprocal_iter<2>(1.0) == 1.0: anything else breaks every tie.
    """
    _run_sfpu_ternary(
        formats,
        DestAccumulation.No,
        mathop,
        input_dimensions=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
        operands=_TIE_OPERANDS[mathop](MAX_TILE_ELEMENTS),
        exact=True,
    )


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

    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")

    # 64x64 = 2x2 tiles: exercises the multi-tile block loop in sfpu_ternary_test.cpp.
    input_dimensions = [64, 64]
    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    src_C, tile_cnt_C, _, _ = generate_stimuli(
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

    golden_generator = get_golden_generator(WhereGolden)
    golden = golden_generator(src_A, src_B, src_C)

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

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    assert torch_equal_nan(golden_tensor, res_tensor), "Assert against golden failed"


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

    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")

    # Create alternating pattern for condition (0, 1, 0, 1, ...)
    pattern = torch.arange(height * width) % 2
    C = pattern.view(height, width).to(format_dict[formats.input_format])

    # Set specific values for true and false tensors
    T = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 2
    F = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 11

    golden_generator = get_golden_generator(WhereGolden)
    golden = golden_generator(C, T, F)
    tile_count = height * width // (32 * 32)

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
            C.flatten(),
            formats.input_format,
            T.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
            buffer_C=F.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_count,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[: len(golden)]

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    golden_tensor = golden_tensor.flatten()

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"
    assert torch_equal_nan(golden_tensor, res_tensor), "Assert against golden failed"
