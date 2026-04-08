# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test


@skip_for_wormhole
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
)
def test_unary_datacopy_custom(formats, dest_acc, workers_tensix_coordinates):
    input_dimensions = [32, 32]
    num_faces = 4
    tile_cnt = 1

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A, formats.output_format, num_faces, input_dimensions
    )

    configuration = TestConfig(
        "sources/eltwise_unary_datacopy_custom_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A), NUM_FACES(num_faces)],
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
        unpack_to_dest=False,
    )

    outcome = configuration.run(workers_tensix_coordinates)
    res_from_L1 = outcome.result

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
