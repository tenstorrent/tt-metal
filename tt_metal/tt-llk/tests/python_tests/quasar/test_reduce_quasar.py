# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
from itertools import product

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ReduceGapoolGolden,
    ReduceGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    PerfRunType,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import SUPPORTED_TILE_SIZES, is_mx_unsupported_tile_dims
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

# Helper dictionary to map reduce dimensions to math operations
mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}

MATH_FIDELITY_MODES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]
POOL_TYPES = [ReducePool.Max, ReducePool.Sum, ReducePool.Average]


REDUCE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
)


def reduce_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def reduce_dest_acc_modes(*, is_perf=False):
    return (
        [DestAccumulation.No]
        if is_perf
        else [DestAccumulation.No, DestAccumulation.Yes]
    )


def reduce_implied_math_formats(formats, *, is_perf=False):
    if is_perf:
        return [ImpliedMathFormat.Yes]
    if formats.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def reduce_input_dimensions(*, is_perf=False):
    return [64, 64]


def generate_pool_type_and_math_fidelity_combinations(*, is_perf=False):
    def is_valid_combination(pool_type, math_fidelity):
        # Max pool only supports LoFi
        if pool_type == ReducePool.Max:
            return math_fidelity == MathFidelity.LoFi
        # Sum and Average support all fidelities
        return True

    if is_perf:
        return [
            combo
            for combo in product(POOL_TYPES, [MathFidelity.LoFi])
            if is_valid_combination(*combo)
        ]

    return [
        combo
        for combo in product(POOL_TYPES, MATH_FIDELITY_MODES)
        if is_valid_combination(*combo)
    ]


def reduce_pool_type_and_math_fidelity_combinations(*, is_perf=False):
    return generate_pool_type_and_math_fidelity_combinations(is_perf=is_perf)


def generate_int8_pool_type_and_math_fidelity_combinations():
    # Int8 reduce is exact-integer accumulation: only LoFi
    return [
        (ReducePool.Max, MathFidelity.LoFi),
        (ReducePool.Sum, MathFidelity.LoFi),
    ]


@pytest.mark.quasar
@parametrize(
    formats=REDUCE_FORMATS + [InputOutputFormat(DataFormat.Int8, DataFormat.Int32)],
    # Int8 reduce uses int32 dest accumulation
    dest_acc=lambda formats: (
        [DestAccumulation.Yes]
        if formats.input_format == DataFormat.Int8
        else reduce_dest_acc_modes(is_perf=False)
    ),
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type_and_math_fidelity=lambda formats: (
        generate_int8_pool_type_and_math_fidelity_combinations()
        if formats.input_format == DataFormat.Int8
        else reduce_pool_type_and_math_fidelity_combinations(is_perf=False)
    ),
    # Int8→Int32 FPU reduce is 32x32-only for now (no tiny-tile int path yet).
    tile_dimensions=lambda formats: (
        [(32, 32)]
        if formats.input_format == DataFormat.Int8
        else [
            td
            for td in SUPPORTED_TILE_SIZES
            if not is_mx_unsupported_tile_dims(
                formats.input_format, formats.output_format, td
            )
        ]
    ),
    dest_sync_mode=lambda: reduce_dest_sync_modes(is_perf=False),
    # MX formats REQUIRE implied_math_format=Yes on Quasar (bypass format inference pipeline)
    implied_math_format=lambda formats: (
        [ImpliedMathFormat.No]
        if formats.input_format == DataFormat.Int8
        else reduce_implied_math_formats(formats, is_perf=False)
    ),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_reduce_quasar(
    formats,
    tile_dimensions,
    dest_acc,
    reduce_dim,
    pool_type_and_math_fidelity,
    dest_sync_mode,
    implied_math_format,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
):

    pool_type, math_fidelity = pool_type_and_math_fidelity
    tile_shape = construct_tile_shape(tile_dimensions)

    if (
        formats.input_format == DataFormat.Int8
        and reduce_dim == ReduceDimension.Scalar
        and pool_type in (ReducePool.Sum, ReducePool.Average)
    ):
        pytest.skip("Int8->Int32 scalar SUM/AVG reduce is not supported yet on Quasar ")

    if (
        formats.input_format == DataFormat.MxInt8
        and formats.output_format == DataFormat.MxInt2
        and dest_acc == DestAccumulation.No
        and reduce_dim == ReduceDimension.Column
        and pool_type == ReducePool.Sum
        and math_fidelity == MathFidelity.HiFi2
        and dest_sync_mode == DestSync.Full
        and implied_math_format == ImpliedMathFormat.Yes
    ):
        pytest.skip(
            "MxInt8->MxInt2 Column Sum HiFi2 lands on an MxInt2 quantization "
            "bin boundary. torch.matmul's fp32-internal accumulation rounds "
            "in the opposite direction from HW for this specific value, "
            "flipping one element into an adjacent bin. Modeling HW's exact "
            "per-mul-add rounding schedule (FMA experiment) regressed other "
            "Row reduce variants, so the residual is accepted as expected."
        )

    input_dimensions = (
        reduce_input_dimensions(is_perf=True)
        if is_perf
        else [tile_dimensions[0] * 2, tile_dimensions[1] * 2]
    )

    if formats.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127, high=127)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
    )

    if pool_type in [
        ReducePool.Max,
        ReducePool.Sum,
    ]:
        src_B = torch.full((tile_shape.total_tile_size(),), 1)
    else:
        # reduce average divides by length of elements in array we reduce
        if reduce_dim == ReduceDimension.Row:
            src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[1])
        elif reduce_dim == ReduceDimension.Column:
            src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[0])
        else:  # Scalar
            src_B = torch.full(
                (tile_shape.total_tile_size(),),
                1 / math.sqrt(tile_dimensions[0] * tile_dimensions[1]),
            )

    if not is_perf:
        if pool_type == ReducePool.Max:
            generate_golden = get_golden_generator(ReduceGolden)
            golden_tensor = generate_golden(
                src_A,
                reduce_dim,
                pool_type,
                formats.output_format,
                tile_cnt,
                tile_shape=tile_shape,
                input_format=formats.input_format,
            )
        else:
            generate_golden = get_golden_generator(ReduceGapoolGolden)
            golden_tensor = generate_golden(
                src_A,
                src_B,
                formats.output_format,
                reduce_dim,
                math_fidelity,
                tile_cnt,
                tile_shape=tile_shape,
                input_format=formats.input_format,
                dest_acc=dest_acc,
            )

    mathop = mathop_mapping[reduce_dim]

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/reduce_quasar_test.cpp",
        "formats": formats,
        "templates": [
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop, pool_type=pool_type),
            UNPACKER_ENGINE_SEL(),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            NUM_FACES(tile_shape.total_num_faces()),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": (
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
        "dest_acc": dest_acc,
        "disable_format_inference": (
            implied_math_format == ImpliedMathFormat.Yes
            and formats.input_format.is_mx_format()
        ),
    }

    if is_perf:
        configuration = PerfConfig(run_types=run_types, **test_config_kwargs)
        configuration.run(perf_report)
        return

    configuration = TestConfig(
        **{
            **test_config_kwargs,
            "templates": test_config_kwargs["templates"]
            + [PERF_RUN_TYPE(PerfRunType.L1_TO_L1)],
        },
    )
    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
        print_errors=True,
    ), "Assert against golden failed"


from helpers.chip_architecture import ChipArchitecture, get_chip_architecture

_ARCH = get_chip_architecture()


# 2x-packed FP4 register-format variants for the reduce-GAPOOL pipeline. L1 stays MxFp4;
# the unpacker produces MxFp4_2x_A/B in src registers. GAPOOL is one of the op_mmul-gated
# instructions (alongside MVMUL/MVMULDI per tt_instruction_issue.sv).
# Sub-datum expansion in the SrcA format-mux fires correctly for Sum/Average pool types.
# Max pool uses GMPOOL which is NOT in the op_mmul list and is therefore excluded.
#
# Only reduce_dim=Column is exercised here. MXFP4_2x is op_mmul-family-only on Quasar
# (MVMUL/MVMULDI/GAPOOL); the column-reduce LLK path issues only GAPOOLs and works as
# designed. The row/scalar paths in llk_math_reduce.h commit per-face results via
# MOVD2B -> ZEROSRC -> ELWADDDI, and ELWADDDI is not op_mmul, so it reads SrcB through
# the FP4 zf mux while srca_fmt_spec is still MXFP4_2x -- producing all-zero Dest.
@pytest.mark.quasar
@pytest.mark.skipif(
    _ARCH != ChipArchitecture.QUASAR,
    reason="MxFp4_2x GAPOOL reduce is op_mmul-family-only and exists on Quasar. Architecture derivations don't support it.",
)
@parametrize(
    register_format_hint=[DataFormat.MxFp4_2x_A, DataFormat.MxFp4_2x_B],
    formats=lambda register_format_hint: [
        InputOutputFormat(
            DataFormat.MxFp4,
            DataFormat.Float16,
            register_format_hint=register_format_hint,
        ),
        InputOutputFormat(
            DataFormat.MxFp4,
            DataFormat.Float16_b,
            register_format_hint=register_format_hint,
        ),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    reduce_dim=[ReduceDimension.Column],
    pool_type=[ReducePool.Sum, ReducePool.Average],
    math_fidelity=MATH_FIDELITY_MODES,
    dest_sync_mode=lambda: reduce_dest_sync_modes(is_perf=False),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_reduce_quasar_mxfp4_2x_gapool(
    register_format_hint,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    dest_sync_mode,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
):
    input_dimensions = reduce_input_dimensions(is_perf=is_perf)
    tile_shape = construct_tile_shape((32, 32))

    src_A, tile_cnt, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # SrcB scale: Sum uses 1.0 per element; Average uses 1/32 so the row/col reduce gives
    # the mean across the 32-element pool dimension.
    if pool_type == ReducePool.Sum:
        src_B = torch.full((tile_shape.total_tile_size(),), 1)
    else:  # Average
        src_B = torch.full((tile_shape.total_tile_size(),), 1 / 32)

    if not is_perf:
        generate_golden = get_golden_generator(ReduceGapoolGolden)
        golden_tensor = generate_golden(
            src_A,
            src_B,
            formats.output_format,
            reduce_dim,
            math_fidelity,
            tile_cnt,
            input_format=formats.input_format,
        )

    mathop = mathop_mapping[reduce_dim]

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/reduce_quasar_test.cpp",
        "formats": formats,
        "templates": [
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop, pool_type=pool_type),
            UNPACKER_ENGINE_SEL(),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            NUM_FACES(tile_shape.total_num_faces()),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=(32, 32),
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
        "disable_format_inference": False,
    }

    if is_perf:
        configuration = PerfConfig(run_types=run_types, **test_config_kwargs)
        configuration.run(perf_report)
        return

    configuration = TestConfig(
        **{
            **test_config_kwargs,
            "templates": test_config_kwargs["templates"]
            + [PERF_RUN_TYPE(PerfRunType.L1_TO_L1)],
        },
    )
    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    test_passed = passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        print_errors=False,
    )

    assert test_passed, "Assert against golden failed"
