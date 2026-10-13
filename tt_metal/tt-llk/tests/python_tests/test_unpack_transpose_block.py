# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, format_dict
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
    TemplateParameter,
)


@dataclass
class UNPACK_FORM(TemplateParameter):
    unpack_form: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t UNPACK_FORM = {self.unpack_form};"


@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float16], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    unpack_form=[1, 2, 3, 4],
)
def test_unpack_transpose_block(formats, dest_acc, unpack_form):
    unpack_transpose_block(formats, dest_acc, unpack_form, [256, 128])


@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Int8], same=True),
    dest_acc=[DestAccumulation.Yes],
    unpack_form=[1, 2, 3, 4],
)
def test_unpack_transpose_block_int8(formats, dest_acc, unpack_form):
    unpack_transpose_block(formats, dest_acc, unpack_form, [128, 128])


def unpack_transpose_block(formats, dest_acc, unpack_form, input_dimensions):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        t_matrix = get_golden_generator(TransposeGolden)
        golden_tensor = t_matrix.transpose_faces_multi_tile(
            src_A,
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
        DestSync.Half, dest_acc, formats, input_dimensions, TILE_DIMENSIONS
    )

    configuration = TestConfig(
        "sources/unpack_transpose_block_test.cpp",
        formats,
        templates=[UNPACK_FORM(unpack_form)],
        runtimes=[
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
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert torch.equal(res_tensor, golden_tensor), "Assert against golden failed"
