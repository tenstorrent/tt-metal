# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
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
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    SRCA_REUSE_COUNT,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test, result_tensor

MATHOPS = [MathOperation.Elwsub, MathOperation.Elwadd, MathOperation.Elwmul]
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
SRCA_REUSE_COUNTS = [2, 4, 8]
# One fidelity for the golden and, through the MATH_FIDELITY template, the kernel's math init:
# Elwadd and Elwsub ignore it (the init static_asserts anything above LoFi for them), and
# Elwmul at LoFi is the single-phase multiply this test has always checked.
FIDELITY = MathFidelity.LoFi
INPUT_DIMENSIONS = [
    [128, 32],
    [32, 128],
    [64, 128],
    [128, 256],
]

# Float32 input with a 32-bit Dest unpacks SrcA as Tf32, so the SrcA rows are
# 64 bytes apart instead of 32: the row-broadcast unpacker must take its
# SrcA Y-stride from the dst format. The merge-gate test pairs each dest_acc with
# the format that exercises it (16-bit SrcA rows without fp32 dest, Tf32 SrcA rows
# with it), 72 cases. The other two (format, dest_acc) pairs of the grid, where
# SrcA rows stay 16-bit but one side is 32-bit (Float16_b into an fp32 Dest,
# Float32 in L1 narrowed to Float16_b in SrcA), run nightly as the other 72, so
# the two tests together cover the grid once. Both call the same body.
REGULAR_FORMATS = {
    DestAccumulation.No: input_output_formats([DataFormat.Float16_b], same=True),
    DestAccumulation.Yes: input_output_formats([DataFormat.Float32], same=True),
}
NIGHTLY_FORMATS = {
    DestAccumulation.No: input_output_formats([DataFormat.Float32], same=True),
    DestAccumulation.Yes: input_output_formats([DataFormat.Float16_b], same=True),
}


@skip_for_blackhole
@parametrize(
    dest_acc=DEST_ACC_MODES,
    formats=lambda dest_acc: REGULAR_FORMATS[dest_acc],
    mathop=MATHOPS,
    srca_reuse_count=SRCA_REUSE_COUNTS,
    input_dimensions=INPUT_DIMENSIONS,
)
def test_unp_bcast_sub_sdpa(
    formats,
    mathop,
    dest_acc,
    input_dimensions,
    srca_reuse_count,
):
    _run_unp_bcast_sub_sdpa(
        formats, mathop, dest_acc, input_dimensions, srca_reuse_count
    )


@skip_for_blackhole
@pytest.mark.nightly
@parametrize(
    dest_acc=DEST_ACC_MODES,
    formats=lambda dest_acc: NIGHTLY_FORMATS[dest_acc],
    mathop=MATHOPS,
    srca_reuse_count=SRCA_REUSE_COUNTS,
    input_dimensions=INPUT_DIMENSIONS,
)
def test_unp_bcast_sub_sdpa_mixed_width(
    formats,
    mathop,
    dest_acc,
    input_dimensions,
    srca_reuse_count,
):
    _run_unp_bcast_sub_sdpa(
        formats, mathop, dest_acc, input_dimensions, srca_reuse_count
    )


def _run_unp_bcast_sub_sdpa(
    formats,
    mathop,
    dest_acc,
    input_dimensions,
    srca_reuse_count,
):
    # Precompute constants
    input_tiles = input_dimensions[0] * input_dimensions[1] // 1024
    reuse_factor = input_tiles // srca_reuse_count

    if input_tiles % srca_reuse_count != 0:
        pytest.skip("Input tiles must be divisible by reuse factor")

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    src_A = src_A[: 1024 * reuse_factor]

    reshaped_a = src_A.reshape(64 * reuse_factor, 16)

    take = []
    for i in range(0, reshaped_a.shape[0], 64):
        take.append(reshaped_a[i])
        take.append(reshaped_a[i + 16])

    # Reconstruct tiles with broadcasted data

    reconstructed_tiles = []
    for i in range(0, len(take), 2):
        if i + 1 < len(take):
            # Combine pair into 1x32 element
            combined = torch.cat([take[i], take[i + 1]], dim=0)
            # Replicate to create 32x32 tile
            tile_32x32 = combined.repeat(32, 1)
            reconstructed_tiles.append(tile_32x32.flatten())

    tilized_reconstructed_tiles = [
        tilize(tile, stimuli_format=formats.input_format)
        for tile in reconstructed_tiles
    ]

    # The kernel reuses each broadcast A tile against the next srca_reuse_count B tiles.
    # EltwiseBinaryGolden is elementwise, so one call on the whole operand pair is the
    # same golden as one call per (A tile, B tile) and is the form the stimuli cache
    # needs: --stimuli-only / --use-stimuli keep one golden per test node, the result of
    # the last generator call, so a per-tile loop would replay the last tile's golden
    # for every tile.
    bcast_a = torch.cat(
        [tile.repeat(srca_reuse_count) for tile in tilized_reconstructed_tiles]
    )
    binary_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = binary_golden(
        mathop,
        bcast_a,
        src_B,
        formats.output_format,
        FIDELITY,
        formats.input_format,
        dest_acc=dest_acc,
    ).to(dtype=format_dict[formats.output_format])

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    if srca_reuse_count > num_tiles_in_block:
        pytest.skip("srca_reuse_count tiles must fit in one dest block")

    configuration = TestConfig(
        "sources/unpack_a_bcast_eltwise_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_FIDELITY(FIDELITY),
            MATH_OP(mathop=mathop),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            SRCA_REUSE_COUNT(srca_reuse_count),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=reuse_factor,
            tile_count_B=input_tiles,
            tile_count_res=input_tiles,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
