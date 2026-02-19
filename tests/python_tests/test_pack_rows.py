# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import PackRowsGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_ROWS_TO_PACK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

max_tiles = 4
tile_dim = 32

dimension_combinations = [
    [m, n]
    for m in range(tile_dim, max_tiles * tile_dim + 1, tile_dim)
    for n in range(tile_dim, max_tiles * tile_dim + 1, tile_dim)
    if m * n <= max_tiles * tile_dim * tile_dim
]


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Bfp8_b,
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_rows_to_pack=[1, 16, 50, 64],
    dimensions=dimension_combinations,
)
def test_pack_rows(
    formats, dest_acc, num_rows_to_pack, dimensions, workers_tensix_coordinates
):
    row_num_datums = 16

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dimensions,
    )

    generate_golden = get_golden_generator(PackRowsGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        dimensions,
        num_rows_to_pack=num_rows_to_pack,
        tile_count=tile_cnt_A,
    )

    # Calculate expected output size per tile
    output_elements_per_tile = num_rows_to_pack * row_num_datums

    configuration = TestConfig(
        "sources/pack_rows_test.cpp",
        formats,
        templates=[
            generate_input_dim(dimensions, dimensions),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_ROWS_TO_PACK(num_rows_to_pack),
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
        unpack_to_dest=formats.input_format.is_32_bit(),
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    extracted_data = [
        res_tensor[
            i * 1024 : i * 1024 + output_elements_per_tile
        ]  # Each tile has 1024 elements in result tensor
        for i in range(tile_cnt_A)
    ]

    res_tensor_sliced = (
        torch.cat(extracted_data)
        if extracted_data
        else torch.tensor([], dtype=res_tensor.dtype)
    )

    assert len(res_tensor_sliced) == len(golden_tensor)

    assert passed_test(golden_tensor, res_tensor_sliced, formats.output_format)
