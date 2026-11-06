# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest
import torch
from conftest import blackhole_only
from helpers.constraints import skip_if_32bit_input_without_dest_acc
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
    PerfRunType,
    Transpose,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import (
    ALL_PERF_RUN_TYPES,
    create_test_or_perf_config,
)
from helpers.sfpu_accuracy_budget import assert_against_contract
from helpers.sfpu_domains import (
    _OP_DOMAIN_REGISTRY,
    Operand,
    edge_spec,
    exclude_undefined_pair,
    for_op,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    StimuliSpec,
    calculate_tile_and_face_counts,
    generate_stimuli,
)
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    ITERATIONS,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    SFPU_CLOBBER_LREGS,
    SFPU_TERNARY_OP,
    SFPU_TERNARY_SCALAR,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES

_SCALAR_VALUE = 2.0

# Default approx mode. The clobber test overrides it.
_APPROX_MODE = ApproximationMode.No
_SCALAR_VALUE_BITS = struct.unpack("<I", struct.pack("<f", _SCALAR_VALUE))[0]
# Fixed. The kernel reads this constexpr.
_SFPU_ITERATIONS = 8
# [64, 64] only. [128, 64] shares this ULP key and raised the Wormhole Float32 ceiling.
_INPUT_DIMENSIONS = [[64, 64]]


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


def _sfpu_ternary_compile_params(
    mathop, approx_mode, tile_count, loop_factor, faces_to_generate
):
    return [
        SFPU_TERNARY_OP(mathop),
        SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
        APPROX_MODE(approx_mode),
        ITERATIONS(_SFPU_ITERATIONS),
        TILE_COUNT(tile_count),
        LOOP_FACTOR(loop_factor),
        NUM_FACES(num_faces=faces_to_generate),
        UNPACK_TRANS_FACES(Transpose.No),
        UNPACK_TRANS_WITHIN_FACE(Transpose.No),
    ]


def _run_sfpu_ternary(
    formats,
    dest_acc,
    mathop,
    input_dimensions=[64, 64],
    spec_A=None,
    spec_B=None,
    spec_C=None,
    approx_mode=_APPROX_MODE,
    clobber_lregs=False,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    """Drive sources/sfpu_ternary_test.cpp for one functional check or one perf row."""
    if run_types is None:
        run_types = ALL_PERF_RUN_TYPES if is_perf else []
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    tile_count, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions,
        input_dimensions,
        face_r_dim=MAX_FACE_R_DIM,
        num_faces=MAX_NUM_FACES,
    )

    if is_perf:
        # Literal list: the perf schema gate reads templates= and runtimes= lists.
        templates = [
            SFPU_TERNARY_OP(mathop),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(approx_mode),
            ITERATIONS(_SFPU_ITERATIONS),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            DEST_SYNC(),
        ]
        if clobber_lregs:
            templates.append(SFPU_CLOBBER_LREGS())
        runtimes = []
        # C must exist so L1_TO_L1 builds the third Operand; values are unused.
        perf_c = torch.zeros(input_dimensions)
        stimuli = StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
            buffer_C=perf_c.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_count,
        )
    else:
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

        generate_golden = get_golden_generator(TernarySFPUGolden)
        golden = generate_golden(
            mathop,
            src_A,
            src_B,
            src_C,
            _SCALAR_VALUE_BITS,
            formats.output_format,
        )

        templates = _sfpu_ternary_compile_params(
            mathop, approx_mode, tile_count, 1, faces_to_generate
        ) + [
            DEST_SYNC(),
        ]
        if clobber_lregs:
            templates.append(SFPU_CLOBBER_LREGS())
        # Not named runtimes: that assignment would add block columns to the perf schema.
        functional_runtime_params = [
            NUM_BLOCKS(tile_cnt_A),
            NUM_TILES_IN_BLOCK(1),
        ]
        runtimes = functional_runtime_params
        stimuli = StimuliConfig(
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
        )

    # A 32-bit input unpacks straight to Dest.
    unpack_to_dest = formats.input_format.is_32_bit()

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs={
            "test_name": "sources/sfpu_ternary_test.cpp",
            "formats": formats,
            "templates": templates,
            "runtimes": runtimes,
            "variant_stimuli": stimuli,
            "unpack_to_dest": unpack_to_dest,
            "dest_acc": dest_acc,
            "compile_time_formats": True,
        },
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[: len(golden)]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    golden_tensor = torch.tensor(golden, dtype=torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    # Pass approx_mode so a contract row keyed approx "No" still matches.
    assert_against_contract(
        mathop, formats, dest_acc, golden_tensor, res_tensor, approx_mode=approx_mode
    )


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
    input_dimensions=_INPUT_DIMENSIONS,
)
def test_sfpu_ternary(formats, dest_acc, mathop, input_dimensions):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)
    if (
        formats.input_format == DataFormat.Bfp8_b
        and mathop != MathOperation.SfpuAddcmul
    ):
        pytest.skip("Bfp8_b is only supported for addcmul")

    _run_sfpu_ternary(formats, dest_acc, mathop, input_dimensions=input_dimensions)


# Newton init must program vConstFloatPrgm0 itself (tt-llk#1701 items 2 and 5).
_TERNARY_RECIP_NEWTON_OPS = [
    MathOperation.SfpuAddcdiv,
    MathOperation.SfpuSnakeBeta,
]


# Wormhole programs the reciprocal constants in both modes, so this is Blackhole-only.
@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_TERNARY_RECIP_NEWTON_OPS,
    input_dimensions=_INPUT_DIMENSIONS,
)
def test_sfpu_ternary_approx_after_lreg_clobber(
    formats, dest_acc, mathop, input_dimensions
):
    """Clobber SFPU registers before APPROX_MODE=Yes, or 1/c is about 47.6x too large."""
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        input_dimensions=input_dimensions,
        approx_mode=ApproximationMode.Yes,
        clobber_lregs=True,
    )


# Operand C only. A and B stay in their random domains.
_TERNARY_EDGE_OPS = [
    MathOperation.SfpuAddcdiv,
    MathOperation.SfpuAddcmul,
    MathOperation.SfpuLerp,
    MathOperation.SfpuSnakeBeta,
]

# Divides by c, so the numerator stays off zero and the edge is the pole rather than 0/0.
_TERNARY_DIVIDES_BY_C = frozenset(
    {MathOperation.SfpuAddcdiv, MathOperation.SfpuSnakeBeta}
)

# Different seeds so a == b cannot hide a kernel that reads the wrong operand.
_TERNARY_NONZERO_A = StimuliSpec.uniform(intervals=[(-1.0, -0.5), (0.5, 1.0)], seed=0)
_TERNARY_NONZERO_B = StimuliSpec.uniform(intervals=[(-1.0, -0.5), (0.5, 1.0)], seed=1)


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=_TERNARY_EDGE_OPS,
    input_dimensions=_INPUT_DIMENSIONS,
)
def test_sfpu_ternary_edges(formats, dest_acc, mathop, input_dimensions):
    """Drive each ternary op's operand-C pole or knee, where it has one."""
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    spec_C = edge_spec(
        mathop,
        formats.input_format,
        formats.output_format,
        operand=Operand.C,
        dest_acc=dest_acc,
    )
    if spec_C is None:
        pytest.skip(
            reason=f"{mathop.name} has no operand-C edge (no pole, no knee) for this "
            "pipeline"
        )

    nonzero = mathop in _TERNARY_DIVIDES_BY_C
    _run_sfpu_ternary(
        formats,
        dest_acc,
        mathop,
        input_dimensions=input_dimensions,
        spec_A=_TERNARY_NONZERO_A if nonzero else None,
        spec_B=_TERNARY_NONZERO_B if nonzero else None,
        spec_C=spec_C,
    )


def _skip_where_config(formats, dest_acc):
    """where's supported (format, dest_acc) pairs. Shared by the functional tests and perf."""
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")


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
    input_dimensions=_INPUT_DIMENSIONS,
)
def test_ttnn_where(
    formats,
    dest_acc,
    mathop,
    test_case,
    input_dimensions,
):
    _skip_where_config(formats, dest_acc)

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

    if test_case == "all_ones":
        src_A = torch.ones_like(src_A)
    elif test_case == "all_zeros":
        src_A = torch.zeros_like(src_A)

    golden_generator = get_golden_generator(WhereGolden)
    golden = golden_generator(src_A, src_B, src_C)

    configuration = TestConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        templates=_sfpu_ternary_compile_params(
            mathop, _APPROX_MODE, tile_cnt_A, 1, MAX_NUM_FACES
        )
        + [
            DEST_SYNC(),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
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
    input_dimensions=_INPUT_DIMENSIONS,
)
def test_ttnn_where_mcw(
    formats,
    dest_acc,
    mathop,
    input_dimensions,
):
    height, width = input_dimensions
    _skip_where_config(formats, dest_acc)

    pattern = torch.arange(height * width) % 2
    C = pattern.view(height, width).to(format_dict[formats.input_format])

    T = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 2
    F = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 11

    golden_generator = get_golden_generator(WhereGolden)
    golden = golden_generator(C, T, F)
    tile_count = height * width // (32 * 32)

    configuration = TestConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        templates=_sfpu_ternary_compile_params(
            mathop, _APPROX_MODE, tile_count, 1, MAX_NUM_FACES
        )
        + [
            DEST_SYNC(),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
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
