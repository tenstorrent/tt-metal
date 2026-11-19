# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.data_format_inference import is_format_combination_outlier
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    BroadcastType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathOperation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    select_perf_tile_sizes,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DEST_SYNC,
    ENABLE_DIRECT_INDEXING,
    IN_FACE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    REUSE_DEST_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    generate_input_dim,
)
from helpers.tile_constants import FACE_C_DIM, SUPPORTED_TILE_SIZES, get_tile_params
from helpers.tile_shape import construct_tile_shape
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test, result_tensor

ALL_TILE_DIMENSIONS = [list(td) for td in SUPPORTED_TILE_SIZES]

BASE_MATH_OPS = [
    MathOperation.Elwmul,
    MathOperation.Elwadd,
    MathOperation.Elwsub,
]
BASE_PERF_MATH_OPS = [MathOperation.Elwadd, MathOperation.Elwmul]
BFP4_MATH_OPS = [MathOperation.Elwadd, MathOperation.Elwsub]
# Elwadd is already in the base BFP4 perf sweep, so this list stays Elwsub-only.
BFP4_PERF_MATH_OPS = [MathOperation.Elwsub]
INT8_MATH_OPS = [MathOperation.Elwadd, MathOperation.Elwsub]
# BH/WH covers Int8 -> Int8; Quasar separately exercises its Int8 -> Int32 path.
INT8_FORMAT = InputOutputFormat(DataFormat.Int8, DataFormat.Int8)


def get_eltwise_binary_math_ops(formats, *, is_perf=False):
    """Int8 is an exact integer add/sub. The base sweeps keep multiply."""
    if formats.input_format == DataFormat.Int8:
        return INT8_MATH_OPS
    return BASE_PERF_MATH_OPS if is_perf else BASE_MATH_OPS


def get_eltwise_binary_transpose(formats):
    """Int8 coverage stays on the non-transposed path used by the integer test."""
    if formats.input_format == DataFormat.Int8:
        return [Transpose.No]
    return [Transpose.Yes, Transpose.No]


def _unique_dimensions(dimensions):
    return [list(dim) for dim in dict.fromkeys(tuple(dim) for dim in dimensions)]


# Fills SrcB past its own tile; one datum reaching math breaks the result.
SRCB_POISON = 1024.0


def _faces(tile, tile_dimensions):
    """Split one tilized tile into its faces, keyed by (face row, face column)."""
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    face_size = face_r_dim * FACE_C_DIM
    return {
        (fr, fc): tile[
            (fr * num_faces_c_dim + fc)
            * face_size : (fr * num_faces_c_dim + fc + 1)
            * face_size
        ].reshape(face_r_dim, FACE_C_DIM)
        for fr in range(num_faces_r_dim)
        for fc in range(num_faces_c_dim)
    }


def _srcb_own_shape_operands(
    src_B_tilized_flat, tile_cnt, tile_dimensions, tile_dimensions_B
):
    """SrcB tiles of their own shape in SrcA-tile-sized L1 slots, poisoned past each tile.

    Returns the L1 stimuli and, for the broadcast golden, an SrcA-shaped copy of each SrcB tile:
    each SrcB face in the SrcA face at the same position, cut to SrcA's face rows, and poison
    elsewhere.
    """
    size_A = tile_dimensions[0] * tile_dimensions[1]
    size_B = tile_dimensions_B[0] * tile_dimensions_B[1]
    assert size_B <= size_A, "SrcB tile must fit in an SrcA tile slot"
    face_r_dim_A = get_tile_params(tile_dimensions)[0]

    stimuli, golden_view = [], []
    for tile in range(tile_cnt):
        tile_B = src_B_tilized_flat[tile * size_A : tile * size_A + size_B]
        poison = torch.full((size_A - size_B,), SRCB_POISON, dtype=tile_B.dtype)
        stimuli.append(torch.cat([tile_B, poison]))

        view = torch.full((size_A,), SRCB_POISON, dtype=tile_B.dtype)
        faces_A = _faces(view, tile_dimensions)
        for position, face_B in _faces(tile_B, tile_dimensions_B).items():
            if position in faces_A:
                rows = min(face_r_dim_A, face_B.shape[0])
                faces_A[position][:rows] = face_B[:rows]
        golden_view.append(view)
    return torch.cat(stimuli), torch.cat(golden_view)


def get_eltwise_binary_tile_dimensions(transpose_srca, broadcast_type):
    """Functional tile shapes, filtered by transpose and broadcast constraints."""
    if transpose_srca == Transpose.Yes:
        return [[32, 32]]
    if broadcast_type in (BroadcastType.Column, BroadcastType.Row):
        return [td for td in ALL_TILE_DIMENSIONS if td != [32, 16]]
    return ALL_TILE_DIMENSIONS


def get_eltwise_binary_perf_tile_dimensions(transpose_srca, broadcast_type):
    return select_perf_tile_sizes(
        get_eltwise_binary_tile_dimensions(transpose_srca, broadcast_type)
    )


def _dest_full_dimensions(dest_acc, dest_sync, formats, tile_dimensions):
    return generate_perf_input_dimensions(
        dest_acc,
        dest_sync,
        construct_tile_shape(tuple(tile_dimensions)),
    )


def get_eltwise_binary_input_dimensions(
    dest_acc,
    dest_sync,
    formats,
    tile_dimensions,
    *,
    existing_dimensions,
):
    """Keep established matrices and add dest-full tall/wide coverage."""
    tile_rows, tile_cols = tile_dimensions
    divisible_existing = [
        dimensions
        for dimensions in existing_dimensions
        if dimensions[0] % tile_rows == 0 and dimensions[1] % tile_cols == 0
    ]
    return _unique_dimensions(
        divisible_existing
        + _dest_full_dimensions(dest_acc, dest_sync, formats, tile_dimensions)
    )


def get_eltwise_binary_perf_input_dimensions(
    dest_acc, dest_sync, formats, tile_dimensions
):
    return _dest_full_dimensions(dest_acc, dest_sync, formats, tile_dimensions)


def get_eltwise_binary_acc_to_dest(input_dimensions, tile_dimensions):
    """Enable pairwise destination accumulation when at least two tiles exist."""
    tile_rows, tile_cols = tile_dimensions
    tile_count = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    return [False, True] if tile_count >= 2 and tile_count % 2 == 0 else [False]


def get_eltwise_binary_perf_acc_to_dest(input_dimensions, tile_dimensions):
    return get_eltwise_binary_acc_to_dest(input_dimensions, tile_dimensions)


def _accumulated_output_dimensions(
    input_dimensions, tile_dimensions, num_tiles_per_accumulation
):
    if num_tiles_per_accumulation == 1:
        return input_dimensions

    tile_rows, tile_cols = tile_dimensions
    input_rt = input_dimensions[0] // tile_rows
    input_ct = input_dimensions[1] // tile_cols
    if input_rt % num_tiles_per_accumulation == 0:
        return [
            input_dimensions[0] // num_tiles_per_accumulation,
            input_dimensions[1],
        ]
    if input_ct % num_tiles_per_accumulation == 0:
        return [
            input_dimensions[0],
            input_dimensions[1] // num_tiles_per_accumulation,
        ]
    raise ValueError(
        f"Input dimensions {input_dimensions} contain a tile count that cannot "
        f"be grouped by {num_tiles_per_accumulation}"
    )


def _fp32_dest_pair(formats):
    """32-bit DEST when either operand format or the output is Float32."""
    return (
        formats.input_format == DataFormat.Float32
        or formats.output_format == DataFormat.Float32
    )


def _get_valid_formats(dest_acc):
    """Valid pairs for this dest_acc, minus the ones the packer cannot do."""
    all_formats = input_output_formats(
        [
            DataFormat.Bfp4_b,
            DataFormat.Bfp8_b,
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=False,
    )
    selected = []
    for formats in all_formats:
        if dest_acc not in get_valid_dest_accumulation_modes(formats):
            continue
        if dest_acc == DestAccumulation.Yes:
            if _fp32_dest_pair(formats) or is_format_combination_outlier(
                formats.input_format, formats.output_format, DestAccumulation.No
            ):
                selected.append(formats)
            continue
        if formats.output_format != DataFormat.Float32:
            selected.append(formats)
    if dest_acc == DestAccumulation.Yes:
        selected.append(INT8_FORMAT)
    return selected


def get_base_perf_formats(dest_acc):
    """Keep the base perf format axis identical to the functional base test."""
    return _get_valid_formats(dest_acc)


def get_bfp4_formats():
    return [
        formats
        for formats in input_output_formats(
            [
                DataFormat.Bfp4_b,
                DataFormat.Float16_b,
                DataFormat.Bfp8_b,
                DataFormat.Float32,
            ]
        )
        if formats.input_format == DataFormat.Bfp4_b
    ]


def _run_eltwise_binary_test(
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    broadcast_type,
    mathop,
    math_fidelity,
    transpose_srca,
    input_dimensions,
    tile_dimensions,
    acc_to_dest,
    *,
    tile_dimensions_B=None,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    if transpose_srca == Transpose.Yes and broadcast_type == BroadcastType.Scalar:
        pytest.skip("SrcA transpose is not supported with scalar broadcast")

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    # SrcB takes SrcA's shape unless tile_dimensions_B gives it its own.
    face_r_dim_B, num_faces_r_dim_B, num_faces_c_dim_B = get_tile_params(
        tile_dimensions_B or tile_dimensions
    )

    tile_rows, tile_cols = tile_dimensions
    tile_cnt_A = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    tile_cnt_B = tile_cnt_A
    num_tiles_per_accumulation = 2 if acc_to_dest else 1
    tile_cnt_output = tile_cnt_A // num_tiles_per_accumulation
    output_dimensions = _accumulated_output_dimensions(
        input_dimensions, tile_dimensions, num_tiles_per_accumulation
    )

    # Generate stimuli with correct face dimensions for smaller tiles
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format_B,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )
    int8_inputs = formats.input_format == DataFormat.Int8
    if int8_inputs:
        # Keep values in range so Int8 add/sub can be compared exactly.
        src_A = (src_A % 101) - 50
        src_B = (src_B % 101) - 50

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        output_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )
    input_num_tiles_in_block = num_tiles_in_block * num_tiles_per_accumulation

    binary_golden = get_golden_generator(EltwiseBinaryGolden)

    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )

    src_A_tilized_flat = src_A_tilized.flatten()
    src_B_tilized_flat = src_B_tilized.flatten()

    stimuli_A = src_A_tilized_flat
    stimuli_B = src_B_tilized_flat
    golden_src_B = src_B_tilized_flat
    if tile_dimensions_B is not None:
        stimuli_B, golden_src_B = _srcb_own_shape_operands(
            src_B_tilized_flat, tile_cnt_B, tile_dimensions, tile_dimensions_B
        )

    # Prepare golden src_A: apply tile-level transpose if enabled
    # Hardware does transpose_faces then transpose_within_faces during unpack
    golden_src_A = src_A_tilized_flat
    if transpose_srca == Transpose.Yes:
        transpose_golden = get_golden_generator(TransposeGolden)
        # Apply face transpose — also quantizes BFP formats to float16_b
        golden_src_A = transpose_golden.transpose_faces_multi_tile(
            src_A,
            formats.input_format,
            num_tiles=tile_cnt_A,
            tilize=True,
            untilize=False,
            input_dimensions=tuple(input_dimensions),
        )
        # Apply within-face transpose on already-quantized data.
        # Use Float32 to match hardware source register precision (TF32)
        # and avoid double-quantization — hardware only quantizes once
        # during unpack, then transposes at full precision.
        golden_src_A = transpose_golden.transpose_within_faces_multi_tile(
            golden_src_A,
            (
                DataFormat.Float16_b
                if formats.input_format.is_block_float()
                else formats.input_format
            ),
            num_tiles=tile_cnt_A,
            tilize=False,
            untilize=False,
            input_dimensions=tuple(input_dimensions),
        )

    # Prepare golden src_B: apply broadcast if enabled
    if broadcast_type != BroadcastType.None_:
        broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_src_B = broadcast_golden(
            broadcast_type,
            golden_src_B,
            formats.input_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
        )

    # When transpose/broadcast already quantized an operand (BFP -> float16_b),
    # pass None to skip re-quantization in EltwiseBinaryGolden.
    golden_input_format_A = (
        None if transpose_srca == Transpose.Yes else formats.input_format
    )
    golden_input_format_B = (
        None if broadcast_type != BroadcastType.None_ else formats.input_format
    )
    golden_tensor = binary_golden(
        mathop,
        golden_src_A,
        golden_src_B,
        formats.output_format,
        math_fidelity,
        input_format=golden_input_format_A,
        input_format_B=golden_input_format_B,
        acc_to_dest=acc_to_dest,
        tile_shape=construct_tile_shape(tuple(tile_dimensions)),
        num_tiles_per_accumulation=num_tiles_per_accumulation,
        dest_acc=dest_acc,
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]

    test_config_kwargs = {
        "test_name": "sources/eltwise_binary_test.cpp",
        "formats": formats,
        "templates": [
            MATH_FIDELITY(math_fidelity),
            BROADCAST_TYPE(broadcast_type),
            MATH_OP(mathop=mathop),
            DEST_SYNC(dest_sync),
            REUSE_DEST_TYPE(reuse_dest_type=EltwiseBinaryReuseDestType.NONE),
            ACC_TO_DEST(acc_to_dest),
            # Direct indexing is a Quasar-only feature; recorded as off so the perf CSV carries
            # the same column as the Quasar side.
            ENABLE_DIRECT_INDEXING(False),
        ],
        "runtimes": [
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                tile_dimensions=tile_dimensions,
            ),
            UNPACK_TRANS_FACES(transpose_srca),
            UNPACK_TRANS_WITHIN_FACE(transpose_srca),
            TILE_COUNT(tile_cnt_A),
            NUM_TILES_IN_BLOCK(
                num_tiles_in_block,
                input_num_tiles_in_block=input_num_tiles_in_block,
                output_num_tiles_in_block=num_tiles_in_block,
            ),
            NUM_BLOCKS(
                num_blocks,
                input_num_blocks=num_blocks,
                output_num_blocks=num_blocks,
            ),
            IN_FACE_DIMS(face_r_dim, FACE_C_DIM, face_r_dim_B, FACE_C_DIM),
            NUM_FACES_R_DIM(num_faces_r_dim, num_faces_r_dim_B),
            NUM_FACES_C_DIM(num_faces_c_dim, num_faces_c_dim_B),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            stimuli_A,
            formats.input_format,
            stimuli_B,
            formats.input_format_B,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_output,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": unpack_to_dest,
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

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        **({"print_errors": False} if int8_inputs else {}),
    ), "Assert against golden failed"


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=lambda dest_acc: _get_valid_formats(dest_acc),
    mathop=lambda formats: get_eltwise_binary_math_ops(formats),
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(formats, mathop),
    transpose_srca=get_eltwise_binary_transpose,
    tile_dimensions=lambda transpose_srca: get_eltwise_binary_tile_dimensions(
        transpose_srca, BroadcastType.None_
    ),
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: get_eltwise_binary_input_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        existing_dimensions=[[256, 32]],
    ),
    acc_to_dest=get_eltwise_binary_acc_to_dest,
)
def test_eltwise_binary(
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    mathop,
    math_fidelity,
    transpose_srca,
    input_dimensions,
    tile_dimensions,
    acc_to_dest,
):
    _run_eltwise_binary_test(
        dest_acc,
        dest_sync,
        unpack_to_dest,
        formats,
        BroadcastType.None_,
        mathop,
        math_fidelity,
        transpose_srca,
        input_dimensions,
        tile_dimensions,
        acc_to_dest,
    )


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=get_bfp4_formats(),
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    transpose_srca=Transpose.No,
    mathop=BFP4_MATH_OPS,
    tile_dimensions=lambda transpose_srca: get_eltwise_binary_tile_dimensions(
        transpose_srca, BroadcastType.None_
    ),
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: get_eltwise_binary_input_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        existing_dimensions=[[32, 32], [64, 32], [32, 64], [256, 32]],
    ),
    acc_to_dest=get_eltwise_binary_acc_to_dest,
)
def test_eltwise_binary_bfp4_b(
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    math_fidelity,
    transpose_srca,
    mathop,
    input_dimensions,
    tile_dimensions,
    acc_to_dest,
):
    return _run_eltwise_binary_test(
        dest_acc,
        dest_sync,
        unpack_to_dest,
        formats,
        BroadcastType.None_,
        mathop,
        math_fidelity,
        transpose_srca,
        input_dimensions,
        tile_dimensions,
        acc_to_dest,
    )
