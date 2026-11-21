# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.constraints import (
    REUSE_DEST_MATH_OPS,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
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
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DEST_SYNC,
    EN_DEST_REUSE,
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
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.tile_shape import construct_tile_shape
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from test_eltwise_binary import _unique_dimensions, get_eltwise_binary_input_dimensions

DEST_REUSE_MATH_OPS = REUSE_DEST_MATH_OPS


def get_dest_reuse_formats(mathop):
    return input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32]
        + ([] if mathop == MathOperation.Elwmul else [DataFormat.Bfp8_b]),
        same=True,
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

    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
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
    mathop, math_fidelity, formats, torch_format, binary_golden, srcA, srcB
):
    if mathop != MathOperation.Elwmul:
        return (srcA + srcB) if mathop == MathOperation.Elwadd else (srcA - srcB)

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
    mathop, reuse_dest_type, math_fidelity, formats, prepared
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
                mathop, math_fidelity, formats, torch_format, binary_golden, srcA, srcB
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


# Keep the established, numerically stable reuse depth. Deep folds accumulate
# 16-bit Dest rounding error even when the packed output format is Float32.
MAX_DEST_REUSE_ACCUMULATIONS = 4


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


def _doubled_tall_wide(dimensions):
    """Tall and wide matrices with twice the tile count of one output."""
    return [
        [dimensions[0] * 2, dimensions[1]],
        [dimensions[0], dimensions[1] * 2],
    ]


def _quadrupled_tall(dimensions):
    """Tall matrix with four times the tile count of one output."""
    return [dimensions[0] * 4, dimensions[1]]


def get_dest_reuse_input_dimensions(dest_acc, dest_sync, formats, tile_dimensions):
    """Inputs with at least two tiles per destination-reuse accumulation."""
    tile_rows, tile_cols = tile_dimensions
    dimensions = [[512, 32]] if 512 % tile_rows == 0 and 32 % tile_cols == 0 else []
    for output_dimensions in generate_perf_input_dimensions(
        dest_acc,
        dest_sync,
        construct_tile_shape(tuple(tile_dimensions)),
    ):
        dimensions.extend(_doubled_tall_wide(output_dimensions))
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
        accumulations = input_tiles // output_tiles
        if (
            output_tiles < input_tiles
            and input_tiles % output_tiles == 0
            and accumulations <= MAX_DEST_REUSE_ACCUMULATIONS
        ):
            # This also validates the per-block destination capacity.
            get_num_blocks_and_num_tiles_in_block(
                dest_sync,
                dest_acc,
                formats,
                output_dimensions,
                tile_dimensions,
                BlocksCalculationAlgorithm.Standard,
            )
            valid.append(output_dimensions)
    return _unique_dimensions(valid)


def get_dest_reuse_perf_input_dimensions(dest_acc, dest_sync, formats, tile_dimensions):
    """Perf inputs at 2:1, plus one 4:1 tall point on 32x32."""
    perf_outputs = generate_perf_input_dimensions(
        dest_acc,
        dest_sync,
        construct_tile_shape(tuple(tile_dimensions)),
    )
    dimensions = [
        doubled for output in perf_outputs for doubled in _doubled_tall_wide(output)
    ]
    if list(tile_dimensions) == [32, 32]:
        tall_outputs = [output for output in perf_outputs if output[0] >= output[1]]
        if tall_outputs:
            dimensions.append(_quadrupled_tall(tall_outputs[0]))
    return _unique_dimensions(dimensions)


def get_dest_reuse_perf_output_dimensions(
    dest_acc, dest_sync, formats, tile_dimensions, input_dimensions
):
    perf_outputs = generate_perf_input_dimensions(
        dest_acc,
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
    valid_outputs = [
        output for output in perf_outputs if tuple(output) in functional_outputs
    ]
    if len(valid_outputs) <= 1:
        return valid_outputs

    tile_rows, tile_cols = tile_dimensions
    input_rt = input_dimensions[0] // tile_rows
    input_ct = input_dimensions[1] // tile_cols
    prefer_tall = input_rt >= input_ct
    preferred = [
        output
        for output in valid_outputs
        if ((output[0] // tile_rows) >= (output[1] // tile_cols)) == prefer_tall
    ]
    return preferred[:1] if preferred else valid_outputs[:1]


def _run_eltwise_binary_dest_reuse_test(
    reuse_dest_type,
    mathop,
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
        mathop, reuse_dest_type, math_fidelity, formats, prepared
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
            MATH_OP(mathop=mathop),
            DEST_SYNC(dest_sync),
            EN_DEST_REUSE(),
            REUSE_DEST_TYPE(reuse_dest_type=reuse_dest_type),
            ACC_TO_DEST(False),
            ENABLE_DIRECT_INDEXING(False),
        ],
        "runtimes": [
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                tile_dimensions=tile_dimensions,
            ),
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
            IN_FACE_DIMS(
                prepared["face_r_dim"],
                FACE_C_DIM,
                prepared["face_r_dim"],
                FACE_C_DIM,
            ),
            NUM_FACES_R_DIM(prepared["num_faces_r_dim"], prepared["num_faces_r_dim"]),
            NUM_FACES_C_DIM(prepared["num_faces_c_dim"], prepared["num_faces_c_dim"]),
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
    mathop=DEST_REUSE_MATH_OPS,
    formats=get_dest_reuse_formats,
    dest_acc=[DestAccumulation.No],
    dest_sync=[DestSync.Half],
    unpack_to_dest=[False],
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(formats, mathop),
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
def test_eltwise_binary_reuse_dest(
    reuse_dest_type,
    mathop,
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
        mathop,
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        math_fidelity,
        tile_dimensions,
        input_dimensions,
        output_dimensions,
    )
