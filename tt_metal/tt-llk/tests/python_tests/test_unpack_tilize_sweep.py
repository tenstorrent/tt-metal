# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    NarrowTile,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NARROW_TILE,
    NUM_FACES,
    STOCHASTIC_ROUNDING,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    stoch_rnd_type=[
        StochasticRounding.No,
        StochasticRounding.Fpu,
        StochasticRounding.Pack,
        StochasticRounding.All,
    ],
    transpose=[Transpose.No],
    narrow_tile=[NarrowTile.No],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_faces=[4, 2, 1],
    input_dimensions=[[32, 32], [64, 64], [32, 64], [32, 128], [128, 32]],
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

    # Bfp8_b output + Stochastic Rounding Pack/All causes output corruption (value -508 becomes 0)
    if formats.output_format == DataFormat.Bfp8_b and stoch_rnd_type in [
        StochasticRounding.Pack,
        StochasticRounding.All,
    ]:
        pytest.skip(
            "Bfp8_b output with StochasticRounding.Pack/All causes the resulting value to be 0 when input is -508"
        )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    torch_format = format_dict[formats.output_format]

    # Generate golden reference using TilizeGolden model
    tilize_function = get_golden_generator(TilizeGolden)
    golden_tensor = tilize_function(
        src_A,
        input_dimensions,
        formats.output_format,
        num_faces,
    )
    golden_tensor = golden_tensor.to(torch_format)

    configuration = TestConfig(
        "sources/unpack_tilize_sweep_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(stoch_rnd_type),
        ],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            NARROW_TILE(narrow_tile),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
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


# A narrow tile has num_faces_c_dim < num_faces_r_dim: a single column of 16-wide
# faces stacked vertically. The canonical narrow shape is [32, 16] — 2 faces of
# 16x16 stacked vertically (num_faces=2).
#
# With narrow_tile=true the unpack-tilize kernel uses block_c_dim = ct_dim *
# FACE_C_DIM (vs TILE_C_DIM) and num_loops = 2, so the inner stride changes;
# the input must therefore be 16-wide. The pack side on WH also has narrow_tile
# wiring (_llk_pack_init_/hw_configure_/dest_init_), so the source threads
# params.NARROW_TILE through both unpack and pack for end-to-end validation.
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16,
            DataFormat.Float16_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_faces=[2],
    num_narrow_tiles=[1, 2, 4],
)
def test_unpack_tilize_narrow_tile(formats, dest_acc, num_faces, num_narrow_tiles):
    """Exercise the narrow_tile=true path of _llk_unpack_tilize_ end-to-end.

    Validates that the unpack-tilize kernel correctly tilizes [32, 16] narrow
    tiles (2 faces of 16x16 stacked vertically) when narrow_tile=true, and the
    pack writes them back in the matching narrow layout. Multiple narrow tiles
    can be processed sequentially by stacking them vertically in L1.
    """

    arch = get_chip_architecture()

    # BH unpack-tilize narrow_tile=true is unimplemented for non-8-bit formats:
    # tt_llk_blackhole/llk_lib/llk_unpack_tilize.h:184 has a FIXME ("This should
    # be revisited for narrow tiles") and the corresponding num_loops=2 /
    # bot_face_offset logic is commented out. Tracked by tt-llk#1281. Skip BH
    # end-to-end pending the kernel work; the test still compiles on BH.
    if arch == ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "BH unpack-tilize narrow_tile=true unimplemented for non-8-bit "
            "formats (FIXME at tt_llk_blackhole/llk_lib/llk_unpack_tilize.h:184, "
            "tt-llk#1281). Test compiles; promote once kernel is fixed."
        )

    tile_dimensions = [32, 16]
    tile_r, tile_c = tile_dimensions
    input_dimensions = [tile_r * num_narrow_tiles, tile_c]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli_v2(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    torch_format = format_dict[formats.output_format]

    # TilizeGolden hardcodes 32x32 tile shape; for narrow tiles tilize_block
    # must be called directly with tile_dimensions=[32, 16].
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

    # The unpack-tilize kernel walks BLOCK_CT_DIM x BLOCK_RT_DIM narrow tiles;
    # with input [num_narrow_tiles*32, 16] that's 1 column x num_narrow_tiles rows.
    from helpers.test_variant_parameters import INPUT_DIMENSIONS

    configuration = TestConfig(
        "sources/unpack_tilize_sweep_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(StochasticRounding.No),
        ],
        runtimes=[
            INPUT_DIMENSIONS(
                full_rt_dim=num_narrow_tiles,
                full_ct_dim=1,
                block_ct_dim=1,
                block_rt_dim=num_narrow_tiles,
            ),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            NARROW_TILE(NarrowTile.Yes),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
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
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
            write_full_tiles=True,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert len(res_tensor) == len(
        golden_tensor
    ), f"Result length {len(res_tensor)} != golden length {len(golden_tensor)}"

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Narrow-tile unpack-tilize result did not match golden"
