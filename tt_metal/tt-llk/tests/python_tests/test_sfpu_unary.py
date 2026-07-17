# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from itertools import chain, product

import pytest
import torch
from conftest import skip_for_coverage
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.sfpu_domains import exclude_undefined, for_op
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)
from helpers.utils import passed_test

SUPPORTED_FAST_MODE_OPS = [
    MathOperation.Log1p,
    MathOperation.Exp,
    MathOperation.Rsqrt,
    MathOperation.Sqrt,
]

ALL_MATHOPS = [
    MathOperation.Abs,
    MathOperation.Atanh,
    MathOperation.Asinh,
    MathOperation.Acosh,
    MathOperation.Cos,
    MathOperation.Log,
    MathOperation.Log1p,
    MathOperation.Reciprocal,
    MathOperation.Sin,
    MathOperation.Sqrt,
    MathOperation.Rsqrt,
    MathOperation.Square,
    MathOperation.Tanh,
    MathOperation.Celu,
    MathOperation.Silu,
    MathOperation.Tanhshrink,
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Frac,
    MathOperation.Gelu,
    MathOperation.GeluTanh,
    MathOperation.Neg,
    MathOperation.Fill,
    MathOperation.Elu,
    MathOperation.Exp,
    MathOperation.Exp2,
    MathOperation.Hardsigmoid,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    MathOperation.ReluMin,
]

FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
    ]
)

FORMATS_INCLUDE_BFP4_B = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
        DataFormat.Float16,
        DataFormat.Bfp4_b,
    ]
)

MATHOPS_INCLUDE_BFP4_B = [
    MathOperation.Abs,
    MathOperation.Atanh,
    MathOperation.Asinh,
    MathOperation.Acosh,
    MathOperation.Cos,
    MathOperation.Log,
    # MathOperation.Log1p,
    # MathOperation.Reciprocal,
    # MathOperation.Sin,
    # MathOperation.Sqrt,
    MathOperation.Rsqrt,
    MathOperation.Square,
    MathOperation.Tanh,
    MathOperation.Celu,
    MathOperation.Silu,
    MathOperation.Gelu,
    MathOperation.GeluTanh,
    MathOperation.Neg,
    MathOperation.Fill,
    MathOperation.Elu,
    # MathOperation.Exp,
    # MathOperation.Exp2,
    # MathOperation.Hardsigmoid,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    MathOperation.ReluMin,
]

FLOAT_TEST_PARAMS = list(
    chain(
        (
            (fmt, approx, mathop, fast, dest)
            for fmt, approx, mathop, fast, dest in product(
                FORMATS,
                [ApproximationMode.No, ApproximationMode.Yes],
                SUPPORTED_FAST_MODE_OPS,
                [FastMode.No, FastMode.Yes],
                [DestAccumulation.No, DestAccumulation.Yes],
            )
        ),
        (
            (fmt, approx, mathop, FastMode.No, dest)
            for fmt, approx, mathop, dest in product(
                FORMATS,
                [ApproximationMode.No, ApproximationMode.Yes],
                [op for op in ALL_MATHOPS if op not in SUPPORTED_FAST_MODE_OPS],
                [DestAccumulation.No, DestAccumulation.Yes],
            )
        ),
    )
)


# Skipped because of: https://github.com/tenstorrent/tt-llk/issues/1435
@skip_for_coverage
@pytest.mark.nightly
@pytest.mark.parametrize(
    "formats,approx_mode,mathop,fast_mode,dest_acc",
    FLOAT_TEST_PARAMS,
)
@pytest.mark.parametrize(
    "input_dimensions",
    [[64, 64], [128, 256]],
)
def test_eltwise_unary_sfpu_float(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    fast_mode: FastMode,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    if TestConfig.WITH_COVERAGE and mathop in [
        MathOperation.Acosh,
        MathOperation.Log,
        MathOperation.Log1p,
        MathOperation.Reciprocal,
        MathOperation.Sin,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        MathOperation.Square,
        MathOperation.Celu,
        MathOperation.Silu,
        MathOperation.Neg,
        MathOperation.Exp2,
        MathOperation.Hardsigmoid,
        MathOperation.Threshold,
        MathOperation.ReluMax,
        MathOperation.ReluMin,
        MathOperation.Tanh,
    ]:
        # SFPI Issue link: https://github.com/tenstorrent/tt-metal/issues/33268
        pytest.skip(
            reason="When these SFPU ops get compiled with coverage, `#pragma GCC unroll X` marked loops get compiled to invalid assembly"
        )

    if mathop == MathOperation.ReluMin:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1120")

    if mathop == MathOperation.Tanh and approx_mode == ApproximationMode.Yes:
        pytest.skip(reason="Metal tanh does not support approximation mode")

    if TestConfig.WITH_COVERAGE and mathop == MathOperation.Gelu:
        # Issue link: https://github.com/tenstorrent/tt-llk/issues/883
        pytest.skip(
            reason="Compilation error when this mathop gets compiled with coverage"
        )

    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        if formats.input_format == DataFormat.Float16 or formats == InputOutputFormat(
            DataFormat.Float32, DataFormat.Float16
        ):
            pytest.skip(reason="This combination is not supported on BH architecture")

    if (
        approx_mode == ApproximationMode.Yes
        and mathop in [MathOperation.Exp, MathOperation.Exp2, MathOperation.Elu]
        and (
            formats.input_format == DataFormat.Bfp8_b
            or formats.output_format == DataFormat.Bfp8_b
        )
    ):
        pytest.skip(
            reason="Exp-related operations are not supported for bf8_b format in approximation mode."
        )

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        fast_mode,
        input_dimensions,
    )


FLOAT_TEST_PARAMS_BFP4_B = list(
    chain(
        (
            (fmt, approx, mathop, fast, dest)
            for fmt, approx, mathop, fast, dest in product(
                FORMATS_INCLUDE_BFP4_B,
                [ApproximationMode.No, ApproximationMode.Yes],
                [op for op in SUPPORTED_FAST_MODE_OPS if op in MATHOPS_INCLUDE_BFP4_B],
                [FastMode.No, FastMode.Yes],
                [DestAccumulation.No, DestAccumulation.Yes],
            )
        ),
        (
            (fmt, approx, mathop, FastMode.No, dest)
            for fmt, approx, mathop, dest in product(
                FORMATS_INCLUDE_BFP4_B,
                [ApproximationMode.No, ApproximationMode.Yes],
                [
                    op
                    for op in MATHOPS_INCLUDE_BFP4_B
                    if op not in SUPPORTED_FAST_MODE_OPS
                ],
                [DestAccumulation.No, DestAccumulation.Yes],
            )
        ),
    )
)


# Skipped because of: https://github.com/tenstorrent/tt-llk/issues/1435
@skip_for_coverage
@pytest.mark.nightly
@pytest.mark.parametrize(
    "formats,approx_mode,mathop,fast_mode,dest_acc",
    FLOAT_TEST_PARAMS_BFP4_B,
)
@pytest.mark.parametrize(
    "input_dimensions",
    [[64, 64], [128, 256]],
)
def test_eltwise_unary_sfpu_float_bfp4_b(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    fast_mode: FastMode,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    if TestConfig.WITH_COVERAGE and mathop in [
        MathOperation.Acosh,
        MathOperation.Log,
        MathOperation.Log1p,
        MathOperation.Reciprocal,
        MathOperation.Sin,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        MathOperation.Square,
        MathOperation.Celu,
        MathOperation.Silu,
        MathOperation.Neg,
        MathOperation.Exp2,
        MathOperation.Hardsigmoid,
        MathOperation.Threshold,
        MathOperation.ReluMax,
        MathOperation.ReluMin,
        MathOperation.Tanh,
    ]:
        # SFPI Issue link: https://github.com/tenstorrent/tt-metal/issues/33268
        pytest.skip(
            reason="When these SFPU ops get compiled with coverage, `#pragma GCC unroll X` marked loops get compiled to invalid assembly"
        )

    if (
        formats.input_format != DataFormat.Bfp4_b
        and formats.input_format_B != DataFormat.Bfp4_b
    ):
        pytest.skip(reason="Not a Bfp4_b test")

    if mathop == MathOperation.ReluMin:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1120")

    if mathop == MathOperation.Tanh and approx_mode == ApproximationMode.Yes:
        pytest.skip(reason="Metal tanh does not support approximation mode")

    if TestConfig.WITH_COVERAGE and mathop == MathOperation.Gelu:
        # Issue link: https://github.com/tenstorrent/tt-llk/issues/883
        pytest.skip(
            reason="Compilation error when this mathop gets compiled with coverage"
        )

    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        if formats.input_format == DataFormat.Float16 or formats == InputOutputFormat(
            DataFormat.Float32, DataFormat.Float16
        ):
            pytest.skip(reason="This combination is not supported on BH architecture")

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        fast_mode,
        input_dimensions,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    approx_mode=[ApproximationMode.No, ApproximationMode.Yes],
    mathop=[
        MathOperation.Neg,
        MathOperation.Fill,
    ],
    fast_mode=[FastMode.No, FastMode.Yes],
    dest_acc=[DestAccumulation.Yes],
    input_dimensions=[[128, 256]],
)
def test_eltwise_unary_sfpu_int(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    fast_mode: FastMode,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    if formats.input_format == DataFormat.Int32:
        pytest.skip(reason=f"Int32 tests break fast tilize, tracked in #495")

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_int.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        fast_mode,
        input_dimensions,
    )


# Unary SFPU ops that require per-op input domains
DOMAIN_MATHOPS = [
    MathOperation.Add1,
    MathOperation.CastFp32ToFp16a,
    MathOperation.Cbrt,
    MathOperation.Clamp,
    MathOperation.Digamma,
    MathOperation.EqualZero,
    MathOperation.Erf,
    MathOperation.Erfc,
    MathOperation.Erfinv,
    MathOperation.Expm1,
    MathOperation.Expm1Cw,
    MathOperation.Fmod,
    MathOperation.GreaterThanEqualZero,
    MathOperation.GreaterThanZero,
    MathOperation.Hardmish,
    MathOperation.Hardshrink,
    MathOperation.Hardtanh,
    MathOperation.Heaviside,
    MathOperation.I0,
    MathOperation.I1,
    MathOperation.Identity,
    MathOperation.LessThanEqualZero,
    MathOperation.LessThanZero,
    MathOperation.Lgamma,
    MathOperation.Lrelu,
    MathOperation.Mish,
    MathOperation.NotEqualZero,
    MathOperation.Polygamma,
    MathOperation.Prelu,
    MathOperation.Rdiv,
    MathOperation.Remainder,
    MathOperation.Rpow,
    MathOperation.RsqrtCompat,
    MathOperation.Selu,
    MathOperation.Sigmoid,
    MathOperation.SigmoidAppx,
    MathOperation.Sign,
    MathOperation.Signbit,
    MathOperation.Softplus,
    MathOperation.Softshrink,
    MathOperation.Softsign,
    MathOperation.SqrtCustom,
    MathOperation.TanhDerivative,
    MathOperation.TanhDerivativeLut,
    MathOperation.UnaryGe,
    MathOperation.UnaryGt,
    MathOperation.UnaryLe,
    MathOperation.UnaryLt,
    MathOperation.UnaryMax,
    MathOperation.UnaryMin,
    MathOperation.UnaryPower,
    MathOperation.Xielu,
    # Trigonometric / inverse / hyperbolic and round (per-op safe domains).
    MathOperation.Tan,
    MathOperation.Atan,
    MathOperation.Asin,
    MathOperation.Acos,
    MathOperation.Sinh,
    MathOperation.Cosh,
    MathOperation.Round,
    # gelu derivative and log-with-base (log2).
    MathOperation.GeluDerivative,
    MathOperation.LogWithBase,
]

# Per-op (atol, rtol) overrides for coarse LUT/polynomial ops; others use the
# per-format default in passed_test.
DOMAIN_CUSTOM_TOLERANCES = {
    # Coarse 3-segment LUT: good PCC but abs error peaks ~0.12 near the knees.
    MathOperation.SigmoidAppx: (0.13, 0.05),
}


# Large matrix (2 formats x ~52 ops x 2 dest_acc); nightly-gated to keep presubmit fast.
@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    mathop=DOMAIN_MATHOPS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_domain(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        # Only Float32->Float32 is supported on BH with dest_acc=No; skip the rest.
        if formats != InputOutputFormat(DataFormat.Float32, DataFormat.Float32):
            pytest.skip(reason="This combination is not supported on BH architecture")

    if TestConfig.WITH_COVERAGE and mathop in [
        MathOperation.GeluDerivative,
        MathOperation.LogWithBase,
    ]:
        # gelu/log-family `#pragma GCC unroll` loops compile to invalid assembly
        # under coverage instrumentation (tt-metal#33268 / tt-llk#883), same as the
        # float-sweep skips for Gelu/Log above.
        pytest.skip(
            reason="gelu/log-family ops fail to compile under coverage instrumentation"
        )

    # Per-op input domain, clipped to where the op is defined (e.g. erfinv: |x| < 1).
    specs = for_op(mathop, formats.input_format)
    spec_A = exclude_undefined(mathop, specs.spec_A)

    custom_atol, custom_rtol = DOMAIN_CUSTOM_TOLERANCES.get(mathop, (None, None))

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=spec_A,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_signbit(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        # Only Float32->Float32 is supported on BH with dest_acc=No; skip the rest.
        if formats != InputOutputFormat(DataFormat.Float32, DataFormat.Float32):
            pytest.skip(reason="This combination is not supported on BH architecture")

    # Sample both signs, avoiding 0 to sidestep -0.0 / rounding ambiguity.
    spec_A = StimuliSpec.uniform(intervals=[(-100.0, -0.5), (0.5, 100.0)])

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        MathOperation.Signbit,
        FastMode.No,
        input_dimensions,
        spec_A=spec_A,
    )


# Predicate ops (write 1.0/0.0). Finite-only stimuli give constant output (PCC
# undefined), so drive them with a spec interleaving +inf / -inf / nan and finite values.
ISINF_ISNAN_MATHOPS = [
    MathOperation.Isinf,
    MathOperation.Isposinf,
    MathOperation.Isneginf,
    MathOperation.Isnan,
    MathOperation.Isfinite,
]


def _isinf_isnan_stimuli_spec():
    def dist(size, dtype, generator):
        # Finite ramp in [-5, 5] with regular +inf / -inf / nan injected so every
        # face carries all special classes plus finite values.
        idx = torch.arange(size, dtype=torch.float32)
        x = (idx % 11) - 5.0
        x[0::7] = float("inf")
        x[1::7] = float("-inf")
        x[2::7] = float("nan")
        return x.to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    mathop=ISINF_ISNAN_MATHOPS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_isinf_isnan(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        # Only Float32->Float32 is supported on BH with dest_acc=No; skip the rest.
        if formats != InputOutputFormat(DataFormat.Float32, DataFormat.Float32):
            pytest.skip(reason="This combination is not supported on BH architecture")

    # bf16->fp32 dest unpack (non-32-bit input + dest_acc=Yes) doesn't preserve
    # -inf/nan, mangling is_neg/is_nan; skip — covered by the other input cases.
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip(
            reason="bf16->fp32 dest unpack does not preserve -inf/nan special values"
        )

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=_isinf_isnan_stimuli_spec(),
    )


def _logical_not_stimuli_spec():
    # logical_not(x) = (x == 0) ? 1 : 0. Random floats never hit 0, so force a
    # regular subset to exactly 0.0 so both branches fire and output is non-constant.
    def dist(size, dtype, generator):
        idx = torch.arange(size, dtype=torch.float32)
        x = (idx % 7) - 3.0  # spans [-3, 3], hits 0 once per 7 elements
        x[0::3] = 0.0  # additional guaranteed zeros
        return x.to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    mathop=[MathOperation.LogicalNotUnary],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_logical_not(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        # Only Float32->Float32 is supported on BH with dest_acc=No; skip the rest.
        if formats != InputOutputFormat(DataFormat.Float32, DataFormat.Float32):
            pytest.skip(reason="This combination is not supported on BH architecture")

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=_logical_not_stimuli_spec(),
    )


def _unary_eq_ne_stimuli_spec():
    # unary_eq/ne compare against threshold 0.5. Random floats effectively never
    # equal 0.5, so the output would be constant (PCC undefined). Force a regular
    # subset to exactly 0.5 so both the equal and not-equal branches fire.
    def dist(size, dtype, generator):
        idx = torch.arange(size, dtype=torch.float32)
        x = (idx % 5) - 2.0  # spans {-2, -1, 0, 1, 2}; none equal 0.5
        x[0::3] = 0.5  # guaranteed threshold hits
        return x.to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    mathop=[MathOperation.UnaryEq, MathOperation.UnaryNe],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_unary_eq_ne(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        # Only Float32->Float32 is supported on BH with dest_acc=No; skip the rest.
        if formats != InputOutputFormat(DataFormat.Float32, DataFormat.Float32):
            pytest.skip(reason="This combination is not supported on BH architecture")

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=_unary_eq_ne_stimuli_spec(),
    )


def eltwise_unary_sfpu(
    test_name,
    formats: list[InputOutputFormat],
    dest_acc,
    approx_mode,
    mathop,
    fast_mode: FastMode,
    input_dimensions: list[int],
    spec_A=None,
    custom_atol=None,
    custom_rtol=None,
):
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        test_name,
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(approx_mode),
            FAST_MODE(fast_mode),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        # dest_acc off: Float32 unpacks to 16-bit in src regs (later copied to dest for SFPU op)
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
    )

    res_from_L1 = configuration.run().result

    # res_from_L1 = res_from_L1[:1024]
    # golden_tensor = golden_tensor[:1024]
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    ), "Assert against golden failed"


# Test exponential with APPROX_MODE=true, FAST_MODE=true, and CLAMP_NEGATIVE=true/false
@pytest.mark.parametrize("clamp_negative", [True, False])
def test_exponential_clamp_negative(clamp_negative: bool):
    torch.manual_seed(0)
    input_dimensions = [32, 32]
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    dest_acc = DestAccumulation.No

    # Generate custom stimuli with range [-5, 0.7]
    num_elements = input_dimensions[0] * input_dimensions[1]
    src_A = torch.rand(num_elements, dtype=torch.bfloat16) * 5.7 - 5.0
    # Set some values to be large and negative:
    src_A[0] = -10000
    src_A[1] = -1000
    src_A[2] = -200
    src_A[3] = -100
    src_A[4] = -88.5

    src_B = torch.zeros(num_elements, dtype=torch.bfloat16)
    tile_cnt_A = (input_dimensions[0] // 32) * (input_dimensions[1] // 32)
    tile_cnt_B = tile_cnt_A

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Exp,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.Yes),
            FAST_MODE(FastMode.Yes),
            CLAMP_NEGATIVE(clamp_negative),
            MATH_OP(mathop=MathOperation.Exp),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # clamp_negative=False: require inputs < -88 to be negative (not necessarily
    # correct) and exclude them from the isclose check.
    if not clamp_negative:
        assert torch.all(
            res_tensor[:5] <= 0
        ), "Some of the first 5 elements are positive"
        res_tensor[:5] = golden_tensor[:5]

    # Use relaxed tolerance for this test
    atol, rtol = 0.02, 0.02
    is_close = torch.isclose(golden_tensor, res_tensor, rtol=rtol, atol=atol)
    is_nan = torch.isnan(golden_tensor) & torch.isnan(res_tensor)
    is_valid = is_close | is_nan

    assert torch.all(
        is_valid
    ), f"Test failed: {(~is_valid).sum()} elements outside tolerance (atol={atol}, rtol={rtol})"
