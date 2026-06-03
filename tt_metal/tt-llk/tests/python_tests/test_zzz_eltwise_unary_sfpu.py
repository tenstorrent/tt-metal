# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import pytest
import torch
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
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
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
    MathOperation.Gelu,
    MathOperation.Neg,
    MathOperation.Fill,
    MathOperation.Elu,
    MathOperation.Exp,
    MathOperation.Exp2,
    MathOperation.Hardsigmoid,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    # MathOperation.ReluMin permanently broken: https://github.com/tenstorrent/tt-llk/issues/1120
]

FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
    ]
)

FORMATS_INCLUDE_BFP4_B = [
    fmt
    for fmt in input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
            DataFormat.Float16,
            DataFormat.Bfp4_b,
        ]
    )
    if fmt.input_format == DataFormat.Bfp4_b
]

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
    MathOperation.Neg,
    MathOperation.Fill,
    MathOperation.Elu,
    # MathOperation.Exp,
    # MathOperation.Exp2,
    # MathOperation.Hardsigmoid,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    # MathOperation.ReluMin permanently broken: https://github.com/tenstorrent/tt-llk/issues/1120
]

# SFPI Issue link: https://github.com/tenstorrent/tt-metal/issues/33268
# When these SFPU ops get compiled with coverage, `#pragma GCC unroll X` marked
# loops get compiled to invalid assembly.
_COVERAGE_SKIPPED_OPS = frozenset(
    [
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
        MathOperation.Tanh,
        # Gelu: https://github.com/tenstorrent/tt-llk/issues/883
        MathOperation.Gelu,
    ]
)


def _is_valid_sfpu_float_combo(
    fmt: InputOutputFormat,
    approx: ApproximationMode,
    mathop: MathOperation,
    fast: FastMode,
    dest: DestAccumulation,
) -> bool:
    """Return False for parameter combos that should never run."""
    # Coverage: certain ops produce invalid assembly under coverage builds.
    if TestConfig.WITH_COVERAGE and mathop in _COVERAGE_SKIPPED_OPS:
        return False

    # Metal tanh does not support approximation mode.
    if mathop == MathOperation.Tanh and approx == ApproximationMode.Yes:
        return False

    # BH arch limitation: Float16 input without dest accumulation is unsupported.
    if (
        dest == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and (
            fmt.input_format == DataFormat.Float16
            or fmt == InputOutputFormat(DataFormat.Float32, DataFormat.Float16)
        )
    ):
        return False

    # Exp-related operations are not supported for Bfp8_b format in approximation mode.
    if (
        approx == ApproximationMode.Yes
        and mathop in [MathOperation.Exp, MathOperation.Exp2, MathOperation.Elu]
        and (
            fmt.input_format == DataFormat.Bfp8_b
            or fmt.output_format == DataFormat.Bfp8_b
        )
    ):
        return False

    return True


@dataclass(frozen=True, repr=False)
class SfpuConfig:
    formats: InputOutputFormat
    approx_mode: ApproximationMode
    mathop: MathOperation
    fast_mode: FastMode
    dest_acc: DestAccumulation

    def __repr__(self):
        f = self.formats
        return (
            f"{self.mathop.name}-approx_{self.approx_mode.name}"
            f"-fast_{self.fast_mode.name}-{self.dest_acc.name}"
            f"-{f.input_format.name}->{f.output_format.name}"
        )


def _build_sfpu_configs(formats_list, mathops_list):
    """Build SfpuConfig list with template params (mathop, approx, fast, dest_acc)
    in outer loops and formats (runtime, doesn't affect compile hash) innermost."""
    configs = []
    for mathop in mathops_list:
        fast_modes = (
            [FastMode.No, FastMode.Yes]
            if mathop in SUPPORTED_FAST_MODE_OPS
            else [FastMode.No]
        )
        for approx in [ApproximationMode.No, ApproximationMode.Yes]:
            for fast in fast_modes:
                for dest in [DestAccumulation.No, DestAccumulation.Yes]:
                    for fmt in formats_list:
                        if _is_valid_sfpu_float_combo(fmt, approx, mathop, fast, dest):
                            configs.append(SfpuConfig(fmt, approx, mathop, fast, dest))
    return configs


FLOAT_CONFIGS = _build_sfpu_configs(FORMATS, ALL_MATHOPS)


# Skipped because of: https://github.com/tenstorrent/tt-llk/issues/1435
@pytest.mark.nightly
@pytest.mark.parametrize(
    "config",
    FLOAT_CONFIGS if not TestConfig.WITH_COVERAGE else [],
)
@pytest.mark.parametrize(
    "input_dimensions",
    [[64, 64], [128, 256]],
)
def test_eltwise_unary_sfpu_float(
    config: SfpuConfig,
    input_dimensions: list[int],
):
    formats = config.formats
    approx_mode = config.approx_mode
    mathop = config.mathop
    fast_mode = config.fast_mode
    dest_acc = config.dest_acc

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        fast_mode,
        input_dimensions,
    )


FLOAT_CONFIGS_BFP4_B = _build_sfpu_configs(
    FORMATS_INCLUDE_BFP4_B, MATHOPS_INCLUDE_BFP4_B
)


# Skipped because of: https://github.com/tenstorrent/tt-llk/issues/1435
@pytest.mark.nightly
@pytest.mark.parametrize(
    "config",
    FLOAT_CONFIGS_BFP4_B if not TestConfig.WITH_COVERAGE else [],
)
@pytest.mark.parametrize(
    "input_dimensions",
    [[64, 64], [128, 256]],
)
def test_eltwise_unary_sfpu_float_bfp4_b(
    config: SfpuConfig,
    input_dimensions: list[int],
):
    formats = config.formats
    approx_mode = config.approx_mode
    mathop = config.mathop
    fast_mode = config.fast_mode
    dest_acc = config.dest_acc

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        fast_mode,
        input_dimensions,
    )


@pytest.mark.skip(reason="Int32 tests break fast tilize, tracked in #495")
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
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_int.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        fast_mode,
        input_dimensions,
    )


def eltwise_unary_sfpu(
    test_name,
    formats: list[InputOutputFormat],
    dest_acc,
    approx_mode,
    mathop,
    fast_mode: FastMode,
    input_dimensions: list[int],
):
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    tile_cnt_A = (input_dimensions[0] // TILE_DIMENSIONS[0]) * (
        input_dimensions[1] // TILE_DIMENSIONS[1]
    )
    tile_cnt_B = tile_cnt_A

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=tile_cnt_A,
    )

    configuration = TestConfig(
        test_name,
        formats,
        templates=[
            APPROX_MODE(approx_mode),
            FAST_MODE(fast_mode),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=stimuli,
        dest_acc=dest_acc,
        # If dest_acc is off, we unpack Float32 into 16-bit format in src registers (later copied over in dest reg for SFPU op)
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
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

    stimuli.set_buffers(src_A, src_B)

    res_from_L1 = configuration.run().result

    # res_from_L1 = res_from_L1[:1024]
    # golden_tensor = golden_tensor[:1024]
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# Test exponential with APPROX_MODE=true, FAST_MODE=true, and CLAMP_NEGATIVE=true/false
@pytest.mark.parametrize("clamp_negative", [True, False])
def test_exponential_clamp_negative(clamp_negative: bool):
    torch.manual_seed(0)
    input_dimensions = [32, 32]
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    dest_acc = DestAccumulation.No

    tile_cnt_A = (input_dimensions[0] // 32) * (input_dimensions[1] // 32)
    tile_cnt_B = tile_cnt_A

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=tile_cnt_A,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.Yes),
            FAST_MODE(FastMode.Yes),
            CLAMP_NEGATIVE(clamp_negative),
            MATH_OP(mathop=MathOperation.Exp),
        ],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=stimuli,
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

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

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Exp,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    stimuli.set_buffers(src_A, src_B)

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # When clamp_negative = False require inputs < -88 to be negative (but not necessarily correct),
    # and don't include them in the resulting isclose check.
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
