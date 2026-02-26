# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict, format_tile_sizes
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli_w_tile_dimensions
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.tile_constants import calculate_tile_size_bytes, get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from ttexalens.tt_exalens_lib import read_from_device

TILE_DIMENSIONS = [16, 32]

# Width in tiles (16x32); height fixed at 1 row of tiles
WIDTHS = [
    2,
    10,
    18,
    56,
    224,
]  # base case, two banks, three banks, and Deepseek model sizes


@skip_for_blackhole
@parametrize(
    formats=input_output_formats([DataFormat.Float32, DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    dimensions=[(1, w) for w in WIDTHS],
)
def test_fast_tilize_tiny_tiles(
    formats, dest_acc, dimensions, workers_tensix_coordinates
):

    if (
        formats.input == DataFormat.Float32 or formats.output == DataFormat.Float32
    ) and dimensions[1] == 224:
        pytest.skip("Can't do 226 tiles for Float32")

    input_height, input_width = dimensions

    input_dimensions = [input_height * 16, input_width * 32]

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(TILE_DIMENSIONS)
    num_faces = num_faces_r_dim * num_faces_c_dim

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli_w_tile_dimensions(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=TILE_DIMENSIONS,
    )

    golden_tensor = (
        tilize_block(
            src_A,
            input_dimensions,
            formats.output,
            num_faces=num_faces,
            tile_dimensions=TILE_DIMENSIONS,
        )
        .flatten()
        .to(format_dict[formats.output])
    )

    configuration = TestConfig(
        "sources/fast_tilize_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(
                full_rt_dim=input_height,
                full_ct_dim=input_width,
                block_ct_dim=input_width,
                block_rt_dim=input_height,
            )
        ],
        runtimes=[TILE_COUNT(tile_cnt_A), LOOP_FACTOR(1), NUM_FACES(num_faces)],
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
            face_r_dim=face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
            operand_res_tile_size=format_tile_sizes[formats.output_format],
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    # Verify the kernel wrote a contiguous block of tiles to L1 (no gaps
    # from incorrect bank-switch addressing). This confirms the packer's
    # multi-DEST-bank L1 pointer transition is correct.
    tile_size_actual = calculate_tile_size_bytes(
        formats.output_format, TILE_DIMENSIONS, format_tile_sizes
    )
    expected_bytes = tile_size_actual * tile_cnt_A
    raw_res = read_from_device(
        workers_tensix_coordinates,
        configuration.variant_stimuli.buf_res_addr,
        num_bytes=expected_bytes,
    )
    assert (
        len(raw_res) == expected_bytes
    ), f"L1 result region size mismatch: expected {expected_bytes}, got {len(raw_res)}"

    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output])

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
