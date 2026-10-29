# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from helpers.chip_architecture import (
    ChipArchitecture,
    get_chip_architecture,
    is_4row_arch,
)
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ReduceGapoolGolden,
    ReduceGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    PerfRunType,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
    quasar_mx_smoke,
    runtime,
    select_perf_tile_sizes,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.reduce_perf_matrix import reduce_to_one_modes
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    REDUCE_TO_ONE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
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

POOL_TYPES = [ReducePool.Max, ReducePool.Sum, ReducePool.Average]


# Float32 joins only through the two pairs the shared perf matrix uses.
REDUCE_FORMATS = (
    input_output_formats([DataFormat.Float16_b, DataFormat.Float16])
    + [
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
    ]
    + quasar_mx_smoke(DataFormat.MxFp4, DataFormat.Float16_b)
)


def reduce_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def reduce_dest_acc_modes(formats):
    return get_valid_dest_accumulation_modes(formats)


def reduce_tile_dimensions(formats, *, is_perf=False):
    functional_tile_sizes = [
        td
        for td in SUPPORTED_TILE_SIZES
        if not is_mx_unsupported_tile_dims(
            formats.input_format, formats.output_format, td
        )
    ]
    return (
        select_perf_tile_sizes(functional_tile_sizes)
        if is_perf
        else functional_tile_sizes
    )


def reduce_is_reduce_to_one_modes(formats, reduce_dim, pool_type):
    if formats.input_format.is_mx_format():
        return [False]
    return reduce_to_one_modes(reduce_dim, pool_type)


def reduce_input_dimensions():
    return [64, 64]


def reduce_math_fidelities(formats, pool_type):
    # GMPOOL (Max) ignores fidelity, so only LoFi is meaningful.
    if pool_type == ReducePool.Max:
        return [MathFidelity.LoFi]
    # MX inputs decode to Float16_b in the src registers, so LoFi is already full
    # precision for them; get_valid_math_fidelities applies that cap.
    return get_valid_math_fidelities(formats)


@pytest.mark.quasar
@parametrize(
    formats=REDUCE_FORMATS,
    tile_dimensions=runtime(
        lambda formats: reduce_tile_dimensions(formats, is_perf=False)
    ),
    dest_acc=reduce_dest_acc_modes,
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=POOL_TYPES,
    math_fidelity=reduce_math_fidelities,
    is_reduce_to_one=reduce_is_reduce_to_one_modes,
    dest_sync_mode=lambda: reduce_dest_sync_modes(is_perf=False),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_reduce_quasar(
    formats,
    tile_dimensions,
    dest_acc,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    dest_sync_mode,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
    input_dimensions=None,
):
    tile_shape = construct_tile_shape(tile_dimensions)

    if input_dimensions is None:
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

    output_tile_cnt = 1 if is_reduce_to_one else tile_cnt

    if not is_perf:
        if pool_type == ReducePool.Max:
            generate_golden = get_golden_generator(ReduceGolden)
            golden_tensor = generate_golden(
                src_A,
                reduce_dim,
                pool_type,
                formats.output_format,
                tile_cnt,
                reduce_to_one=is_reduce_to_one,
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
            if is_reduce_to_one:
                # Sum the fidelity-masked per-tile results so the golden still tracks fidelity.
                golden_tensor = (
                    golden_tensor.to(torch.float32)
                    .view(tile_cnt, -1)
                    .sum(dim=0)
                    .to(format_dict[formats.output_format])
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
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                tile_dimensions=tile_dimensions,
            ),
            TILE_COUNT(tile_cnt),
            REDUCE_TO_ONE(is_reduce_to_one),
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
            tile_count_res=output_tile_cnt,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        # Reduce needs its operands in SrcA/SrcB; it never unpacks to Dest.
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

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
    _ARCH != ChipArchitecture.QUASAR or is_4row_arch(),
    reason="MxFp4_2x GAPOOL reduce is not implemented on this architecture",
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
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
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
    input_dimensions = reduce_input_dimensions()
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
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt),
            REDUCE_TO_ONE(False),
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
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

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
