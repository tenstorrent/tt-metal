# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    TilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    Tilize,
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
    DEST_INDEX,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from helpers.utils import passed_test


def get_valid_tilize_datacopy(formats):
    """
    Get valid tilize options for _llk_math_eltwise_unary_datacopy_

    - Blackhole and Wormhole have differing APIs:
        - Blackhole: Has tilize argument (SW workaround for HW bug)
        - Wormhole: No tilize argument (No SW workaround needed)
    - Tilize cannot be enabled if input format is Bfp8_b (HW limitation)

    Therefore we only test tilization on Blackhole
    """

    chip_arch = get_chip_architecture()

    if chip_arch == ChipArchitecture.WORMHOLE:
        return [Tilize.No]

    if formats.input_format == DataFormat.Bfp8_b:
        return [Tilize.No]

    return [Tilize.No, Tilize.Yes]


def get_valid_num_faces_datacopy(tilize):
    """
    Get valid num_faces options for _llk_math_eltwise_unary_datacopy_

    - Number of faces must be 4 when tilization is enabled (SW limitation)
    - Otherwise num_faces can be 1, 2, or 4
    """

    if tilize == Tilize.Yes:
        return [4]

    return [1, 2, 4]


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    num_faces=lambda tilize: get_valid_num_faces_datacopy(tilize),
    tilize=lambda formats: get_valid_tilize_datacopy(formats),
    input_dimensions=[[64, 64], [32, 256], [128, 256]],
)
def test_unary_datacopy(
    formats, dest_acc, num_faces, tilize, input_dimensions, workers_tensix_coordinates
):

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    if tilize == Tilize.No:
        generate_golden = get_golden_generator(DataCopyGolden)
        golden_tensor = generate_golden(
            src_A, formats.output_format, num_faces, input_dimensions
        )
    else:
        generate_golden = get_golden_generator(TilizeGolden)
        golden_tensor = generate_golden(src_A, input_dimensions, formats.output_format)

    unpack_to_dest = (
        False
        if tilize == Tilize.Yes and formats.input_format == DataFormat.Float32
        else formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    blocks_calculation_algorithm = (
        BlocksCalculationAlgorithm.Standard
        if tilize == Tilize.No
        else BlocksCalculationAlgorithm.Tilize
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        blocks_calculation_algorithm,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_datacopy_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILIZE(tilize),
        ],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
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
            num_faces=num_faces,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
