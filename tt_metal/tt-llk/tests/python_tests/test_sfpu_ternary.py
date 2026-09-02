# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest
import torch
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
from helpers.utils import passed_test

_SCALAR_VALUE = 2.0
_SCALAR_VALUE_BITS = struct.unpack("<I", struct.pack("<f", _SCALAR_VALUE))[0]


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
):
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
# The random sweep holds c in uniform(1, 2) for addcdiv and snake_beta because both divide by
# it, so the pole is unreachable by construction; this drives it.
# `edge_spec(op, ..., operand=Operand.C)` resolves it through the usual metadata:
#
#   addcdiv    a + value * b / c    -> _OP_SINGULARITIES C = (0.0, BOTH)
#   snake_beta a + sin(b*a)^2 / c   -> _OP_SINGULARITIES C = (0.0, BOTH)
#   lerp       a + c * (b - a)      -> _OP_OPERAND_EDGE_POINTS C = (-1, 0, 1, 2)
#   addcmul    a + value * b * c    -> nothing; a multiply has no pole, so edge_spec is None
#
# Only C gets edge values; A and B keep their random domains, since the divisor is the
# interesting operand and pinning all three would test one point rather than a spread.
# ─────────────────────────────────────────────────────────────────────────────

_TERNARY_EDGE_OPS = [
    MathOperation.SfpuAddcdiv,
    MathOperation.SfpuAddcmul,
    MathOperation.SfpuLerp,
    MathOperation.SfpuSnakeBeta,
]

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
        # addcmul: c is a multiplicand, so it has no pole and no knee, and cat B is gated on
        # SPECIALS_READY_OPS. The random sweep already covers everything a probe could add.
        pytest.skip(
            reason=f"{mathop.name} has no operand-C edge (no pole, no knee) for this "
            "pipeline"
        )

    # Keep the numerator off zero for the dividing ops, so the variant asserts the pole
    # rather than the 0/0 indeterminate form. See _TERNARY_DIVIDES_BY_C.
    nonzero = mathop in _TERNARY_DIVIDES_BY_C
    _run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        spec_A=_TERNARY_NONZERO_A if nonzero else None,
        spec_B=_TERNARY_NONZERO_B if nonzero else None,
        spec_C=spec_C,
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
    # For "mixed" case, use the generated stimuli as-is

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

    # Generate dtype dynamically based on current input format

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
