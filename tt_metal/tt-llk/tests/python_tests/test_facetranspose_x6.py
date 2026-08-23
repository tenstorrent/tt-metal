# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""X6 face-transpose VEHICLE test (lane FV, 2026-08-22).

Runs sources/facetranspose_x6_test.cpp -- the transpose_dest kernel with
the math-thread face transpose spelled on the TYPED X6 surface
(sfpi::face_transpose_dst_32b_batch) -- against the SAME golden as the
hand LLK's <transpose_of_faces=No, is_32bit=True> combination
(transpose_dest_test.cpp + UNPACK_TRANSPOSE_FACES=Yes): full-tile
transpose = unpacker face rearrangement + math within-face 16x16.
Int32/Float32 lanes assert BIT-EXACT equality (the unpack-to-dest 32-bit
path), the hand test's own strictness rule.
"""

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, Transpose, format_dict
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)


@parametrize(
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Int32], same=True
    ),
)
def test_facetranspose_x6(formats):
    if isinstance(formats, tuple):
        formats = formats[0]
    dest_acc = DestAccumulation.Yes
    unpack_to_dest = True
    input_dimensions = [128, 128]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # The hand test's zero-flag sentinel stimuli for Int32 (values whose
    # low bytes vanish) -- the X6 cfg block's zero-flag arm is load-bearing
    # for exactly this class.
    if formats.input_format == DataFormat.Int32:
        src_A = (torch.arange(0, src_A.numel()) * 10000).reshape_as(src_A)
        src_B = (torch.arange(0, src_B.numel()) * 10000).reshape_as(src_B)

    generate_datacopy_golden = get_golden_generator(DataCopyGolden)
    datacopy_tensor = generate_datacopy_golden(
        src_A, formats.output_format, num_faces=4, input_dimensions=input_dimensions
    )

    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        t_matrix = get_golden_generator(TransposeGolden)
        golden_tensor = t_matrix.transpose_faces_multi_tile(
            datacopy_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )
        golden_tensor = t_matrix.transpose_within_faces_multi_tile(
            golden_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            untilize=False,
            input_dimensions=input_dimensions,
        )
    else:
        golden_tensor = []

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
    )

    configuration = TestConfig(
        "sources/facetranspose_x6_test.cpp",
        formats,
        templates=[],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.Yes),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # 32-bit unpack-to-dest path: bit-exact assertion (the hand test's rule).
    assert torch.equal(res_tensor, golden_tensor), "Assert against golden failed"
