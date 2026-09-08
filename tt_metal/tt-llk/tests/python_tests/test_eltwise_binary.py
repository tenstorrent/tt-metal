# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
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
    MathFidelity,
    MathOperation,
    PerfRunType,
    Transpose,
    format_dict,
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
    BROADCAST_TYPE,
    DEST_SYNC,
    EN_DEST_REUSE,
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
)
from helpers.tile_constants import FACE_C_DIM, SUPPORTED_TILE_SIZES, get_tile_params
from helpers.tile_shape import construct_tile_shape
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

ALL_TILE_DIMENSIONS = [list(td) for td in SUPPORTED_TILE_SIZES]

BASE_MATH_OPS = [
    MathOperation.Elwmul,
    MathOperation.Elwadd,
    MathOperation.Elwsub,
]
BASE_PERF_MATH_OPS = [MathOperation.Elwadd, MathOperation.Elwmul]
BFP4_MATH_OPS = [MathOperation.Elwadd, MathOperation.Elwsub]
DEST_REUSE_MATH_OPS = [
    MathOperation.Elwadd,
    MathOperation.Elwsub,
    MathOperation.Elwmul,
]
INT8_MATH_OPS = [MathOperation.Elwadd, MathOperation.Elwsub]


def _unique_dimensions(dimensions):
    return [list(dim) for dim in dict.fromkeys(tuple(dim) for dim in dimensions)]


def _effective_dest_acc(dest_acc, formats):
    return (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else dest_acc
    )


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
        _effective_dest_acc(dest_acc, formats),
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


def _get_valid_formats(dest_acc):
    """
    Filter formats based on dest accumulation:
    - If dest accumulation is enabled, input must be Float32
    """
    all_formats = input_output_formats(
        [
            DataFormat.Bfp4_b,
            DataFormat.Bfp8_b,
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=False,
    )
    if dest_acc == DestAccumulation.Yes:
        return [f for f in all_formats if f.input_format == DataFormat.Float32]
    return all_formats


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


def get_dest_reuse_formats(math_op):
    return input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32]
        + ([] if math_op == MathOperation.Elwmul else [DataFormat.Bfp8_b]),
        same=True,
    )


INT8_FORMAT = InputOutputFormat(DataFormat.Int8, DataFormat.Int8)


def _get_valid_math_fidelity(formats, math_op=None):
    """
    Filter math fidelity based on input data format:
    - Bfp8_b: LoFi only
    - Float16_b: LoFi or HiFi2
    - Float32: HiFi3 and HiFi4

    Math fidelity > LoFi is only supported for Elwmul (hardware constraint),
    so non-multiply ops are restricted to LoFi regardless of format.
    """
    if math_op is not None and math_op != MathOperation.Elwmul:
        return [MathFidelity.LoFi]
    input_format = formats.input_format
    if input_format in [DataFormat.Bfp8_b, DataFormat.Bfp4_b]:
        return [MathFidelity.LoFi]
    elif input_format == DataFormat.Float16_b:
        return [MathFidelity.LoFi, MathFidelity.HiFi2]
    elif input_format == DataFormat.Float32:
        return [MathFidelity.HiFi3, MathFidelity.HiFi4]
    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


def _run_eltwise_binary_test(
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    broadcast_type,
    math_op,
    math_fidelity,
    transpose_srca,
    input_dimensions,
    tile_dimensions,
    *,
    int8_inputs=False,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    if transpose_srca == Transpose.Yes and broadcast_type == BroadcastType.Scalar:
        pytest.skip("SrcA transpose is not supported with scalar broadcast")

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_rows, tile_cols = tile_dimensions
    tile_cnt_A = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    tile_cnt_B = tile_cnt_A

    # Generate stimuli with correct face dimensions for smaller tiles
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format_B,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )
    if int8_inputs:
        # Keep values in range so Int8 add/sub can be compared exactly.
        src_A = (src_A % 101) - 50
        src_B = (src_B % 101) - 50

    effective_dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else dest_acc
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        effective_dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

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
                if formats.input_format in [DataFormat.Bfp4_b, DataFormat.Bfp8_b]
                else formats.input_format
            ),
            num_tiles=tile_cnt_A,
            tilize=False,
            untilize=False,
            input_dimensions=tuple(input_dimensions),
        )

    # Prepare golden src_B: apply broadcast if enabled
    golden_src_B = src_B_tilized_flat
    if broadcast_type != BroadcastType.None_:
        broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_src_B = broadcast_golden(
            broadcast_type,
            src_B_tilized_flat,
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
        math_op,
        golden_src_A,
        golden_src_B,
        formats.output_format,
        math_fidelity,
        input_format=golden_input_format_A,
        input_format_B=golden_input_format_B,
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
            MATH_OP(mathop=math_op),
            DEST_SYNC(dest_sync),
            REUSE_DEST_TYPE(reuse_dest_type=EltwiseBinaryReuseDestType.NONE),
        ],
        "runtimes": [
            UNPACK_TRANS_FACES(transpose_srca),
            UNPACK_TRANS_WITHIN_FACE(transpose_srca),
            TILE_COUNT(tile_cnt_A),
            NUM_TILES_IN_BLOCK(
                num_tiles_in_block,
                input_num_tiles_in_block=num_tiles_in_block,
                output_num_tiles_in_block=num_tiles_in_block,
            ),
            NUM_BLOCKS(
                num_blocks,
                input_num_blocks=num_blocks,
                output_num_blocks=num_blocks,
            ),
            NUM_FACES_R_DIM(num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim),
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
            tile_count_res=tile_cnt_A,
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

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result tensor ({len(res_from_L1)}) and golden tensor ({len(golden_tensor)}) are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

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
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    math_op=BASE_MATH_OPS,
    math_fidelity=lambda formats, math_op: _get_valid_math_fidelity(formats, math_op),
    transpose_srca=[Transpose.Yes, Transpose.No],
    tile_dimensions=lambda transpose_srca, broadcast_type: get_eltwise_binary_tile_dimensions(
        transpose_srca, broadcast_type
    ),
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: get_eltwise_binary_input_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        existing_dimensions=[[256, 32]],
    ),
)
def test_eltwise_binary(
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    broadcast_type,
    math_op,
    math_fidelity,
    transpose_srca,
    input_dimensions,
    tile_dimensions,
):
    _run_eltwise_binary_test(
        dest_acc,
        dest_sync,
        unpack_to_dest,
        formats,
        broadcast_type,
        math_op,
        math_fidelity,
        transpose_srca,
        input_dimensions,
        tile_dimensions,
    )


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=get_bfp4_formats(),
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    math_fidelity=lambda formats: _get_valid_math_fidelity(formats),
    transpose_srca=Transpose.No,
    math_op=BFP4_MATH_OPS,
    tile_dimensions=lambda transpose_srca, broadcast_type: get_eltwise_binary_tile_dimensions(
        transpose_srca, broadcast_type
    ),
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: get_eltwise_binary_input_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        existing_dimensions=[[32, 32], [64, 32], [32, 64], [256, 32]],
    ),
)
def test_eltwise_binary_bfp4_b(
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    broadcast_type,
    math_fidelity,
    transpose_srca,
    math_op,
    input_dimensions,
    tile_dimensions,
):
    return _run_eltwise_binary_test(
        dest_acc,
        dest_sync,
        unpack_to_dest,
        formats,
        broadcast_type,
        math_op,
        math_fidelity,
        transpose_srca,
        input_dimensions,
        tile_dimensions,
    )


def _prepare_dest_reuse_inputs(
    formats,
    input_dimensions,
    output_dimensions,
    tile_dimensions,
    dest_sync,
    dest_acc,
):
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_rows, tile_cols = tile_dimensions
    tile_cnt_input = (input_dimensions[0] // tile_rows) * (
        input_dimensions[1] // tile_cols
    )
    tile_cnt_output = (output_dimensions[0] // tile_rows) * (
        output_dimensions[1] // tile_cols
    )

    assert tile_cnt_input % tile_cnt_output == 0, (
        f"Input tile count ({tile_cnt_input}) must be divisible by "
        f"output tile count ({tile_cnt_output})"
    )
    inner_dim = tile_cnt_input // tile_cnt_output
    assert inner_dim > 1, "Dest reuse requires at least one reuse accumulation"

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    effective_dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else dest_acc
    )
    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        effective_dest_acc,
        formats,
        output_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )
    input_tiles_in_block = inner_dim * output_tiles_in_block

    src_A_tilized_flat = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    ).flatten()
    src_B_tilized_flat = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    ).flatten()

    return {
        "face_r_dim": face_r_dim,
        "num_faces_r_dim": num_faces_r_dim,
        "num_faces_c_dim": num_faces_c_dim,
        "num_faces": num_faces,
        "tile_cnt_input": tile_cnt_input,
        "tile_cnt_output": tile_cnt_output,
        "inner_dim": inner_dim,
        "output_num_blocks": output_num_blocks,
        "output_tiles_in_block": output_tiles_in_block,
        "input_tiles_in_block": input_tiles_in_block,
        "input_num_blocks": output_num_blocks,
        "src_A_tilized_flat": src_A_tilized_flat,
        "src_B_tilized_flat": src_B_tilized_flat,
        "tile_elements": num_faces * face_r_dim * FACE_C_DIM,
        "torch_format": format_dict[formats.output_format],
    }


def _apply_dest_reuse_op(
    math_op, math_fidelity, formats, torch_format, binary_golden, srcA, srcB
):
    if math_op != MathOperation.Elwmul:
        return (srcA + srcB) if math_op == MathOperation.Elwadd else (srcA - srcB)

    # Mask from the original operands each fidelity phase. Do not use
    # _compute_eltwise: it mutates t1/t2 across iterations, so later phases
    # mask already-truncated values and collapse to ~LoFi.
    fidelity_iters = {
        MathFidelity.LoFi: 1,
        MathFidelity.HiFi2: 2,
        MathFidelity.HiFi3: 3,
        MathFidelity.HiFi4: 4,
    }[math_fidelity]
    result = None
    for fidelity_iter in range(fidelity_iters):
        a_m, b_m = binary_golden._apply_fidelity_masking(
            formats.output_format, srcA, srcB, fidelity_iter
        )
        phase = a_m.to(torch.float32) * b_m.to(torch.float32)
        result = phase if result is None else result + phase
    return result.to(torch_format)


def _compute_dest_reuse_golden(
    math_op, reuse_dest_type, math_fidelity, formats, prepared
):
    """Simulate seeded dest reuse (seed first tile, then fold via DEST_TO_SRCA/B)."""
    tile_elements = prepared["tile_elements"]
    torch_format = prepared["torch_format"]
    src_A = prepared["src_A_tilized_flat"]
    src_B = prepared["src_B_tilized_flat"]
    golden_tensor = torch.zeros(
        prepared["tile_cnt_output"] * tile_elements, dtype=torch_format
    )
    # Instantiate directly: get_golden_generator returns a DummyGoldenGenerator
    # during compile-producer, which has no _apply_fidelity_masking helper.
    binary_golden = EltwiseBinaryGolden()

    for out_t in range(prepared["tile_cnt_output"]):
        block_idx = out_t // prepared["output_tiles_in_block"]
        tile_in_block = out_t % prepared["output_tiles_in_block"]
        dest = torch.zeros(tile_elements, dtype=torch_format)

        for i in range(prepared["inner_dim"]):
            input_tile_idx = (
                block_idx * prepared["input_tiles_in_block"]
                + i * prepared["output_tiles_in_block"]
                + tile_in_block
            )
            start = input_tile_idx * tile_elements
            end = start + tile_elements
            a_tile = src_A[start:end].to(torch_format)
            b_tile = src_B[start:end].to(torch_format)

            if i == 0:
                srcA, srcB = a_tile, b_tile
            elif reuse_dest_type == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
                srcA, srcB = dest.clone(), b_tile
            else:
                srcA, srcB = a_tile, dest.clone()

            dest = _apply_dest_reuse_op(
                math_op, math_fidelity, formats, torch_format, binary_golden, srcA, srcB
            )

        out_start = out_t * tile_elements
        golden_tensor[out_start : out_start + tile_elements] = dest

    return golden_tensor


DEST_REUSE_TILE_DIMENSIONS = [
    [32, 32],
    [16, 16],
    [32, 16],
    [1, 32],
    [16, 32],
    [8, 32],
]


def get_dest_reuse_perf_tile_dimensions():
    return select_perf_tile_sizes(DEST_REUSE_TILE_DIMENSIONS)


def get_dest_reuse_output_candidates(dest_acc, dest_sync, formats, tile_dimensions):
    existing = [[128, 32], [256, 32]] if tile_dimensions == [32, 32] else [[128, 32]]
    return get_eltwise_binary_input_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        existing_dimensions=existing,
    )


def get_dest_reuse_input_dimensions(dest_acc, dest_sync, formats, tile_dimensions):
    """Inputs with at least two tiles per destination-reuse accumulation."""
    tile_rows, tile_cols = tile_dimensions
    dimensions = [[512, 32]] if 512 % tile_rows == 0 and 32 % tile_cols == 0 else []
    for output_dimensions in generate_perf_input_dimensions(
        _effective_dest_acc(dest_acc, formats),
        dest_sync,
        construct_tile_shape(tuple(tile_dimensions)),
    ):
        dimensions.extend(
            [
                [output_dimensions[0] * 2, output_dimensions[1]],
                [output_dimensions[0], output_dimensions[1] * 2],
            ]
        )
    return _unique_dimensions(dimensions)


def get_dest_reuse_output_dimensions(
    dest_acc,
    dest_sync,
    formats,
    tile_dimensions,
    input_dimensions,
):
    """Only emit divisible, capacity-safe output matrices."""
    tile_rows, tile_cols = tile_dimensions
    input_tiles = (input_dimensions[0] // tile_rows) * (
        input_dimensions[1] // tile_cols
    )
    valid = []
    for output_dimensions in get_dest_reuse_output_candidates(
        dest_acc, dest_sync, formats, tile_dimensions
    ):
        if (
            output_dimensions[0] % tile_rows != 0
            or output_dimensions[1] % tile_cols != 0
        ):
            continue
        output_tiles = (output_dimensions[0] // tile_rows) * (
            output_dimensions[1] // tile_cols
        )
        if output_tiles < input_tiles and input_tiles % output_tiles == 0:
            # This also validates the per-block destination capacity.
            get_num_blocks_and_num_tiles_in_block(
                dest_sync,
                _effective_dest_acc(dest_acc, formats),
                formats,
                output_dimensions,
                tile_dimensions,
                BlocksCalculationAlgorithm.Standard,
            )
            valid.append(output_dimensions)
    return _unique_dimensions(valid)


def get_dest_reuse_perf_input_dimensions(dest_acc, dest_sync, formats, tile_dimensions):
    """Perf inputs derived from doubled dest-full tall/wide outputs."""
    perf_outputs = generate_perf_input_dimensions(
        _effective_dest_acc(dest_acc, formats),
        dest_sync,
        construct_tile_shape(tuple(tile_dimensions)),
    )
    return _unique_dimensions(
        [
            dimensions
            for output in perf_outputs
            for dimensions in (
                [output[0] * 2, output[1]],
                [output[0], output[1] * 2],
            )
        ]
    )


def get_dest_reuse_perf_output_dimensions(
    dest_acc, dest_sync, formats, tile_dimensions, input_dimensions
):
    perf_outputs = generate_perf_input_dimensions(
        _effective_dest_acc(dest_acc, formats),
        dest_sync,
        construct_tile_shape(tuple(tile_dimensions)),
    )
    functional_outputs = {
        tuple(dimensions)
        for dimensions in get_dest_reuse_output_dimensions(
            dest_acc,
            dest_sync,
            formats,
            tile_dimensions,
            input_dimensions,
        )
    }
    return [output for output in perf_outputs if tuple(output) in functional_outputs]


def _run_eltwise_binary_dest_reuse_test(
    reuse_dest_type,
    math_op,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    math_fidelity,
    tile_dimensions,
    input_dimensions,
    output_dimensions,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    prepared = _prepare_dest_reuse_inputs(
        formats,
        input_dimensions,
        output_dimensions,
        tile_dimensions,
        dest_sync=dest_sync,
        dest_acc=dest_acc,
    )
    golden_tensor = _compute_dest_reuse_golden(
        math_op, reuse_dest_type, math_fidelity, formats, prepared
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
            BROADCAST_TYPE(BroadcastType.None_),
            MATH_OP(mathop=math_op),
            DEST_SYNC(dest_sync),
            EN_DEST_REUSE(),
            REUSE_DEST_TYPE(reuse_dest_type=reuse_dest_type),
        ],
        "runtimes": [
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            NUM_TILES_IN_BLOCK(
                prepared["output_tiles_in_block"],
                input_num_tiles_in_block=prepared["input_tiles_in_block"],
                output_num_tiles_in_block=prepared["output_tiles_in_block"],
            ),
            NUM_BLOCKS(
                prepared["output_num_blocks"],
                input_num_blocks=prepared["input_num_blocks"],
                output_num_blocks=prepared["output_num_blocks"],
            ),
            NUM_FACES_R_DIM(prepared["num_faces_r_dim"]),
            NUM_FACES_C_DIM(prepared["num_faces_c_dim"]),
            TEST_FACE_DIMS(face_r_dim=prepared["face_r_dim"]),
            TILE_COUNT(prepared["tile_cnt_input"]),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            prepared["src_A_tilized_flat"],
            formats.input_format,
            prepared["src_B_tilized_flat"],
            formats.input_format,
            formats.output_format,
            tile_count_A=prepared["tile_cnt_input"],
            tile_count_B=prepared["tile_cnt_input"],
            tile_count_res=prepared["tile_cnt_output"],
            num_faces=prepared["num_faces"],
            face_r_dim=prepared["face_r_dim"],
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
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=prepared["torch_format"])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@parametrize(
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    math_op=DEST_REUSE_MATH_OPS,
    formats=get_dest_reuse_formats,
    dest_acc=[DestAccumulation.No],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    math_fidelity=lambda formats, math_op: _get_valid_math_fidelity(formats, math_op),
    tile_dimensions=DEST_REUSE_TILE_DIMENSIONS,
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: get_dest_reuse_input_dimensions(
        dest_acc, dest_sync, formats, tile_dimensions
    ),
    output_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions, input_dimensions: get_dest_reuse_output_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        input_dimensions,
    ),
)
def test_eltwise_binary_dest_reuse(
    reuse_dest_type,
    math_op,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    math_fidelity,
    tile_dimensions,
    input_dimensions,
    output_dimensions,
):
    _run_eltwise_binary_dest_reuse_test(
        reuse_dest_type,
        math_op,
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        math_fidelity,
        tile_dimensions,
        input_dimensions,
        output_dimensions,
    )


@parametrize(
    dest_acc=[DestAccumulation.Yes],  # Dest accumulation is required for int8.
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    formats=INT8_FORMAT,
    broadcast_type=[
        BroadcastType.None_,
    ],
    math_fidelity=MathFidelity.LoFi,
    transpose_srca=Transpose.No,
    math_op=INT8_MATH_OPS,
    tile_dimensions=lambda transpose_srca, broadcast_type: get_eltwise_binary_tile_dimensions(
        transpose_srca, broadcast_type
    ),
    input_dimensions=lambda dest_acc, dest_sync, formats, tile_dimensions: get_eltwise_binary_input_dimensions(
        dest_acc,
        dest_sync,
        formats,
        tile_dimensions,
        existing_dimensions=[[32, 32], [512, 32]],
    ),
)
def test_eltwise_binary_int8_format(
    dest_acc,
    dest_sync,
    unpack_to_dest,
    formats,
    broadcast_type,
    math_fidelity,
    transpose_srca,
    math_op,
    input_dimensions,
    tile_dimensions,
):
    return _run_eltwise_binary_test(
        dest_acc,
        dest_sync,
        unpack_to_dest,
        formats,
        broadcast_type,
        math_op,
        math_fidelity,
        transpose_srca,
        input_dimensions,
        tile_dimensions,
        int8_inputs=True,
    )
