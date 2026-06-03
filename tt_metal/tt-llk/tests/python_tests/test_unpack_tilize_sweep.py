# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
from helpers.utils import passed_test


def _tilize_sweep_formats():
    fmts = input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    )
    return [f for f in fmts if f.input_format != DataFormat.Bfp8_b]


def _tilize_sweep_num_faces(formats):
    if (
        get_chip_architecture() == ChipArchitecture.BLACKHOLE
        and formats.output_format == DataFormat.Bfp8_b
    ):
        return [4]
    return [4, 2, 1]


def _tilize_sweep_stoch_rnd(formats):
    if formats.output_format == DataFormat.Bfp8_b:
        return [StochasticRounding.No, StochasticRounding.Fpu]
    return [
        StochasticRounding.No,
        StochasticRounding.Fpu,
        StochasticRounding.Pack,
        StochasticRounding.All,
    ]


@parametrize(
    formats=_tilize_sweep_formats(),
    stoch_rnd_type=lambda formats: _tilize_sweep_stoch_rnd(formats),
    transpose=[Transpose.No],
    narrow_tile=[NarrowTile.No],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_faces=lambda formats: _tilize_sweep_num_faces(formats),
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
