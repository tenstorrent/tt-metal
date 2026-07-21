# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    NarrowTile,
    StochasticRounding,
    Transpose,
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
    INPUT_DIMENSIONS,
    NARROW_TILE,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    STOCHASTIC_ROUNDING,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    generate_input_dim,
)
from helpers.tile_constants import DEFAULT_TILE_R_DIM, FACE_C_DIM
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test


def _narrow_path(
    src_A, input_dimensions, formats, num_faces, tile_dimensions, torch_format
):
    # TilizeGolden hardcodes 32x32; narrow tiles need tilize_block directly.
    golden_tensor = (
        tilize_block(
            src_A,
            input_dimensions,
            formats.output_format,
            num_faces=num_faces,
            tile_dimensions=tile_dimensions,
        )
        .flatten()
        .to(torch_format)
    )
    num_narrow_tiles = input_dimensions[0] // tile_dimensions[0]
    input_dim_runtime = INPUT_DIMENSIONS(
        full_rt_dim=num_narrow_tiles,
        full_ct_dim=1,
        block_ct_dim=1,
        block_rt_dim=num_narrow_tiles,
    )
    stimuli_extra = {
        "tile_dimensions": tile_dimensions,
        "use_dense_tile_dimensions": True,
    }
    return golden_tensor, input_dim_runtime, stimuli_extra


def _regular_path(src_A, input_dimensions, formats, num_faces, torch_format):
    tilize_function = get_golden_generator(TilizeGolden)
    golden_tensor = tilize_function(
        src_A,
        input_dimensions,
        formats.output_format,
        num_faces,
    ).to(torch_format)
    return (
        golden_tensor,
        generate_input_dim(input_dimensions, input_dimensions),
        {},
    )


# narrow_tile=Yes covers [32, 16] tiles (2 vertical 16x16 faces, num_faces=2).
# BH narrow_tile unimplemented for non-8-bit formats (tt-llk#1281).
@parametrize(
    # Int32 is Int32→Int32 only (unpacker constraint); concatenated via same=True.
    # Intentionally added to both narrow and non-narrow sweeps to exercise the
    # Int32 unpack_to_dest tilize path in each.
    formats=lambda narrow_tile: (
        input_output_formats(
            [DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b]
            + ([DataFormat.Bfp8_b] if narrow_tile == NarrowTile.No else [])
        )
        + input_output_formats([DataFormat.Int32], same=True)
    ),
    stoch_rnd_type=[StochasticRounding.No],
    transpose=[Transpose.No],
    narrow_tile=[NarrowTile.No, NarrowTile.Yes],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_faces=lambda narrow_tile: ([4, 2, 1] if narrow_tile == NarrowTile.No else [2]),
    input_dimensions=lambda narrow_tile: (
        [[32, 32], [64, 64], [32, 64], [32, 128], [128, 32], [128, 256]]
        if narrow_tile == NarrowTile.No
        else [[32, 16], [64, 16], [128, 16], [1024, 16]]
    ),
)
def test_unpack_tilize_comprehensive(
    formats,
    stoch_rnd_type,
    transpose,
    narrow_tile,
    dest_acc,
    num_faces,
    input_dimensions,
):
    """Comprehensive parameter sweep test for unpack_tilize operation."""

    # Get architecture for architecture-specific skips
    arch = get_chip_architecture()

    # BH narrow_tile unimplemented for non-8-bit formats (tt-llk#1281).
    if narrow_tile == NarrowTile.Yes and arch == ChipArchitecture.BLACKHOLE:
        pytest.skip("BH narrow_tile unimplemented for non-8-bit formats (tt-llk#1281)")

    # BFP8_b input format not supported by tilize unpacker
    # Tilize unpacker cannot correctly read row-major BFP8_b data with shared exponents
    # Note: BFP8_b input works in regular unpack mode (test_eltwise_unary_datacopy with tilize_en=false)
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip(
            "BFP8_b input format not supported by tilize unpacker: "
            "cannot read row-major BFP8_b shared exponent data"
        )

    # Blackhole BFP8_b output fails for num_faces=1,2 due to hardcoded tilize packer values
    # Root cause: llk_pack.h has hardcoded MOP_OUTER_LOOP=2, PACK_INTF_SEL for 4-face tiles
    # The packer tries to process 2+ faces even when only 1-2 faces exist, corrupting BFP8_b output
    if (
        arch == ChipArchitecture.BLACKHOLE
        and formats.output_format == DataFormat.Bfp8_b
        and num_faces in [1, 2]
    ):
        pytest.skip(
            "Blackhole BFP8_b output fails for num_faces=1,2: tilize packer has hardcoded "
            "MOP_OUTER_LOOP=2 and PACK_INTF_SEL values that don't adapt to tiny tiles"
        )

    # BH unpack_tilize does not support num_faces=1 (LLK asserts num_faces in {2, 4}).
    # WH supports num_faces=1.
    if arch == ChipArchitecture.BLACKHOLE and num_faces == 1:
        pytest.skip("BH unpack_tilize does not support num_faces=1")

    is_narrow = narrow_tile == NarrowTile.Yes
    # Narrow tile: 2 vertical 16x16 faces (num_faces=2).
    tile_dimensions = [DEFAULT_TILE_R_DIM, FACE_C_DIM] if is_narrow else None

    stimuli_kwargs = {"tile_dimensions": tile_dimensions} if is_narrow else {}
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        **stimuli_kwargs,
    )

    torch_format = format_dict[formats.output_format]

    golden_tensor, input_dim_runtime, stimuli_extra = (
        _narrow_path(
            src_A, input_dimensions, formats, num_faces, tile_dimensions, torch_format
        )
        if is_narrow
        else _regular_path(src_A, input_dimensions, formats, num_faces, torch_format)
    )

    block_tile_dimensions = tile_dimensions if tile_dimensions is not None else [32, 32]
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        block_tile_dimensions,
    )

    configuration = TestConfig(
        "sources/unpack_tilize_sweep_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(stoch_rnd_type),
        ],
        runtimes=[
            input_dim_runtime,
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            NARROW_TILE(narrow_tile),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
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
            write_full_tiles=True,  # Tilize tests need full tiles in L1
            **stimuli_extra,
        ),
        unpack_to_dest=(formats.input_format in [DataFormat.Int32, DataFormat.UInt32]),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
