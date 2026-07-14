# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Replicates ttnn tilize compute kernel flow (compute_kernel_hw_startup +
fast_tilize_init/block/uninit) through the LLK test infra.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    NUM_GUARD_TILES,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

TILE_R = 32
TILE_C = 32


@parametrize(
    formats=[*input_output_formats([DataFormat.Float16_b], same=True)],
    dest_acc=[DestAccumulation.No],
    dimensions=[(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (10, 12)],
)
def test_fast_tilize_metal_api(formats, dest_acc, dimensions):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    rt, ct = dimensions
    input_dimensions = [rt * TILE_R, ct * TILE_C]
    tile_count = rt * ct

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    golden = get_golden_generator(TilizeGolden)(
        src_A, input_dimensions, formats.output_format
    )

    cfg = TestConfig(
        "sources/fast_tilize_metal_api_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(1),
            NUM_FACES(4),
            NUM_GUARD_TILES(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res = cfg.run().result
    assert len(res) == len(golden)
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert passed_test(golden, res_tensor, formats.output_format)
