# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Test for eltwise binary operations with reuse_dest on Quasar.
import pytest
import torch
from helpers.constraints import get_perf_math_operations
from helpers.format_config import DataFormat
from helpers.golden_generator.heavyweight.mismatch import describe_mismatch
from helpers.golden_generator.heavyweight.operations.quasar_operations import (
    QuasarEltwiseBinaryReuseDestGolden,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    INPUT_TILE_CNT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    REUSE_DEST_TYPE,
    TEST_FACE_DIMS,
    generate_input_dim,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import MXFP_MANTISSA_BITS, passed_test

INPUT_DIMENSIONS = [
    [512, 32],
]
OUTPUT_DIMENSIONS = [
    [128, 32],
]

REUSE_DEST_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
)

TILE_DIMENSIONS = [32, 32]


def reuse_dest_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def reuse_dest_mathops(formats, *, is_perf=False):
    if (
        formats.input_format == DataFormat.MxFp8R
        or formats.input_format == DataFormat.MxFp8P
    ):
        supported_mathops = [MathOperation.Elwadd, MathOperation.Elwsub]
    else:
        supported_mathops = [
            MathOperation.Elwadd,
            MathOperation.Elwsub,
            MathOperation.Elwmul,
        ]
    if is_perf:
        return [
            mathop
            for mathop in get_perf_math_operations()
            if mathop in supported_mathops
        ]
    return supported_mathops


def reuse_dest_math_fidelities(mathop):
    if mathop in [MathOperation.Elwadd, MathOperation.Elwsub]:
        return [MathFidelity.LoFi]
    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


def reuse_dest_implied_math_format(formats, *, is_perf=False):
    use_mx = formats.input_format.is_mx_format() or formats.output_format.is_mx_format()
    if is_perf or use_mx:
        return ImpliedMathFormat.Yes
    return ImpliedMathFormat.No


def _reuse_dest_tile_count(dimensions) -> int:
    tile_rows, tile_cols = TILE_DIMENSIONS
    return (dimensions[0] // tile_rows) * (dimensions[1] // tile_cols)


def valid_output_dimensions(formats, dest_sync_mode, input_dimensions) -> list:
    """Output dims compatible with reuse_dest for a given input size, format and dest_sync.

    Three constraints, all decidable at collection time so incompatible combinations are
    never generated (instead of generated then skipped):
      - input tile count must be an exact multiple of the output tile count, and
      - that multiple (`inner_dim`) must be > 1 (reuse_dest needs accumulation), and
      - the output must fit in a single block (the Quasar reuse_dest kernel uses
        block-relative indexing; multi-block accumulates wrongly).
    """
    tile_cnt_input = _reuse_dest_tile_count(input_dimensions)
    valid = []
    for out_dims in OUTPUT_DIMENSIONS:
        tile_cnt_output = _reuse_dest_tile_count(out_dims)
        if tile_cnt_output == 0 or tile_cnt_input % tile_cnt_output != 0:
            continue
        if tile_cnt_input // tile_cnt_output <= 1:
            continue
        try:
            num_blocks, _ = get_num_blocks_and_num_tiles_in_block(
                dest_sync_mode,
                DestAccumulation.No,
                formats,
                out_dims,
                (TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]),
                BlocksCalculationAlgorithm.Standard,
            )
        except ValueError:
            continue  # tiles don't divide evenly into blocks for this combination
        if num_blocks > 1:
            continue
        valid.append(out_dims)
    return valid


@pytest.mark.quasar
@parametrize(
    formats=REUSE_DEST_FORMATS,
    mathop=lambda formats: reuse_dest_mathops(formats, is_perf=False),
    math_fidelity=reuse_dest_math_fidelities,
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    dest_sync_mode=lambda: reuse_dest_dest_sync_modes(is_perf=False),
    input_dimensions=runtime(INPUT_DIMENSIONS),
    output_dimensions=runtime(valid_output_dimensions),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_eltwise_binary_reuse_dest_quasar(
    formats,
    mathop,
    reuse_dest_type,
    math_fidelity,
    dest_sync_mode,
    input_dimensions,
    output_dimensions,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):

    implied_math_format = reuse_dest_implied_math_format(formats, is_perf=is_perf)
    use_mx = formats.input_format.is_mx_format() or formats.output_format.is_mx_format()
    disable_format_inference = use_mx

    tile_rows, tile_cols = TILE_DIMENSIONS
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(
        [tile_rows, tile_cols]
    )
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_cnt_input = (input_dimensions[0] // tile_rows) * (
        input_dimensions[1] // tile_cols
    )
    tile_cnt_output = (output_dimensions[0] // tile_rows) * (
        output_dimensions[1] // tile_cols
    )

    inner_dim = tile_cnt_input // tile_cnt_output

    tile_dimensions_tuple = (tile_rows, tile_cols)
    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync_mode,
        DestAccumulation.No,
        formats,
        output_dimensions,
        tile_dimensions_tuple,
        BlocksCalculationAlgorithm.Standard,
    )
    input_tiles_in_block = inner_dim * output_tiles_in_block
    input_num_blocks = output_num_blocks

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=[tile_rows, tile_cols],
    )
    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=[tile_rows, tile_cols],
        face_r_dim=face_r_dim,
    )
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=[tile_rows, tile_cols],
        face_r_dim=face_r_dim,
    )
    src_A_t = src_A_tilized.flatten()
    src_B_t = src_B_tilized.flatten()

    torch_format = format_dict[formats.output_format]
    # No MX pre-quantization: the stimuli and the golden go through the same
    # packer, so they land on the same lattice by construction. Pre-quantizing
    # with a different rounding rule and then packing would round twice.

    golden_dest = []
    if not is_perf:
        generate_golden = QuasarEltwiseBinaryReuseDestGolden(
            mathop, math_fidelity, reuse_dest_type
        )
        golden_tensor = generate_golden.run(
            [src_A_t, src_B_t],
            formats.input_format,
            formats.output_format,
            inner_dim=inner_dim,
            output_tiles_in_block=output_tiles_in_block,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            # Keep Dest as it stood before the pack, so a failure can say
            # whether the math or the packer produced the disagreement.
            dest_out=golden_dest,
        )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_binary_reuse_dest_quasar_test.cpp",
        "formats": formats,
        "templates": [
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            REUSE_DEST_TYPE(reuse_dest_type),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            generate_input_dim(input_dimensions, input_dimensions),
            INPUT_TILE_CNT(tile_cnt_input),
            OUTPUT_TILE_CNT(tile_cnt_output),
            NUM_TILES_IN_BLOCK(
                output_tiles_in_block,
                input_num_tiles_in_block=input_tiles_in_block,
                output_num_tiles_in_block=output_tiles_in_block,
            ),
            NUM_BLOCKS(
                output_num_blocks,
                input_num_blocks=input_num_blocks,
                output_num_blocks=output_num_blocks,
            ),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=FACE_C_DIM),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A_t,
            formats.input_format,
            src_B_t,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_input,
            tile_count_B=tile_cnt_input,
            tile_count_res=tile_cnt_output,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=[tile_rows, tile_cols],
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": False,
        "dest_acc": DestAccumulation.No,
        "disable_format_inference": disable_format_inference,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
        boot_mode=boot_mode,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    if not passed_test(golden_tensor, res_tensor, formats.output_format):
        pytest.fail(
            describe_mismatch(
                golden_tensor,
                res_tensor,
                context=(
                    f"{formats.input_format}->{formats.output_format} "
                    f"{mathop.name} {math_fidelity.name} {reuse_dest_type.name} "
                    f"{dest_sync_mode.name} in={input_dimensions} "
                    f"out={output_dimensions} inner_dim={inner_dim} "
                    f"tiles_in_block={output_tiles_in_block}"
                ),
                datums_per_tile=num_faces * face_r_dim * FACE_C_DIM,
                # Rank failures the way passed_test judges them: in lattice
                # steps at each element's own magnitude, not absolute error.
                mantissa_bits=MXFP_MANTISSA_BITS.get(formats.output_format, 0),
                chain=generate_golden.last_chain,
                dest=torch.cat(golden_dest) if golden_dest else None,
            )
        )
