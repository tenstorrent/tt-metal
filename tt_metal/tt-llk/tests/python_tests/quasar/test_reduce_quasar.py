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
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
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


def generate_pool_type_and_math_fidelity_combinations():
    def is_valid_combination(pool_type, math_fidelity):
        # Max pool only supports LoFi
        if pool_type == ReducePool.Max:
            return math_fidelity == MathFidelity.LoFi
        # Sum and Average support all fidelities
        return True

    return [
        combo
        for combo in product(POOL_TYPES, MATH_FIDELITY_MODES)
        if is_valid_combination(*combo)
    ]


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
    ),
    tile_dimensions=lambda formats: [
        td
        for td in SUPPORTED_TILE_SIZES
        if not is_mx_unsupported_tile_dims(
            formats.input_format, formats.output_format, td
        )
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type_and_math_fidelity=generate_pool_type_and_math_fidelity_combinations(),
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    implied_math_format=[ImpliedMathFormat.No, ImpliedMathFormat.Yes],
)
def test_reduce_quasar(
    formats,
    tile_dimensions,
    dest_acc,
    reduce_dim,
    pool_type_and_math_fidelity,
    dest_sync_mode,
    implied_math_format,
):

    pool_type, math_fidelity = pool_type_and_math_fidelity
    tile_shape = construct_tile_shape(tile_dimensions)

    input_dimensions = [tile_dimensions[0] * 2, tile_dimensions[1] * 2]

    src_A, tile_cnt, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
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

    configuration = TestConfig(
        "sources/quasar/reduce_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop, pool_type=pool_type),
            UNPACKER_ENGINE_SEL(),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            NUM_FACES(tile_shape.total_num_faces()),
        ],
        variant_stimuli=StimuliConfig(
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
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
        dest_acc=dest_acc,
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
    dest_sync_mode=[DestSync.Half, DestSync.Full],
)
def test_reduce_quasar_mxfp4_2x_gapool(
    register_format_hint,
    formats,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    dest_sync_mode,
):
    input_dimensions = [64, 64]
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

    configuration = TestConfig(
        "sources/quasar/reduce_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop, pool_type=pool_type),
            UNPACKER_ENGINE_SEL(),
            # MX input -> implied math format on the kernel side (matches matmul 2x).
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            NUM_FACES(tile_shape.total_num_faces()),
        ],
        variant_stimuli=StimuliConfig(
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
        unpack_to_dest=False,
        dest_acc=dest_acc,
        disable_format_inference=False,
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
