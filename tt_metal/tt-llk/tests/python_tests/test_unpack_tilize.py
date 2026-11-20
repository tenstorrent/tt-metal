# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import effective_dest_acc
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    ELEMENTS_PER_FACE,
    FACES_PER_TILE,
    TILE_DIMENSIONS,
    TilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.param_config import (
    input_output_formats,
    max_dest_tiles,
    parametrize,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import get_tile_params
from helpers.utils import passed_test, result_tensor

# Tile shapes the unpack_tilize tests sweep: full 32x32 tiles and two-face 16x32 tiles. Shared with
# the perf module so the two sweeps stay aligned.
UNPACK_TILIZE_TILE_DIMENSIONS = [[32, 32], [16, 32]]


def _get_block_config(formats, dest_acc, unpack_to_dest, tile_count):
    configured_dest_acc = dest_acc if dest_acc is not None else DestAccumulation.No
    configured_dest_acc = effective_dest_acc(
        formats.input_format,
        formats.output_format,
        configured_dest_acc,
    )
    tiles_per_block = min(
        tile_count,
        max_dest_tiles(
            configured_dest_acc == DestAccumulation.Yes or unpack_to_dest,
            DestSync.Half,
        ),
    )
    return math.ceil(tile_count / tiles_per_block), tiles_per_block


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,  # Unpack Tilize doesn't work for block float formats (Bfp8_b) due to shared exponent at start of input tensor
            DataFormat.Fp8_e4m3,
        ]
    ),
    tile_dimensions=UNPACK_TILIZE_TILE_DIMENSIONS,
)
def test_unpack_tilize_float(
    formats,
    tile_dimensions,
):
    if (
        formats.input_format == DataFormat.Fp8_e4m3
        or formats.output_format == DataFormat.Fp8_e4m3
    ) and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "Unpack Tilize does not support Fp8_e4m3 format on non-BLACKHOLE architectures"
        )

    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Unpack Tilize does not support Bfp8_b input format")

    if formats.output_format == DataFormat.Bfp8_b and tile_dimensions != [32, 32]:
        pytest.skip("Bfp8_b output format only works with 32x32 tiles (num_faces=4)")

    unpack_tilize(formats, tile_dimensions=tile_dimensions)


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.Yes],
    tile_dimensions=UNPACK_TILIZE_TILE_DIMENSIONS,
)
def test_unpack_tilize_float32_lossless(
    formats,
    dest_acc,
    tile_dimensions,
):
    unpack_tilize(
        formats,
        unpack_to_dest=True,
        validate_lossless=True,
        dest_acc=dest_acc,
        tile_dimensions=tile_dimensions,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    tile_dimensions=UNPACK_TILIZE_TILE_DIMENSIONS,
)
def test_unpack_tilize_int(
    formats,
    tile_dimensions,
):
    unpack_tilize(formats, unpack_to_dest=True, tile_dimensions=tile_dimensions)


@parametrize(
    formats=input_output_formats([DataFormat.Int8]),
    tile_dimensions=UNPACK_TILIZE_TILE_DIMENSIONS,
)
def test_unpack_tilize_int8(
    formats,
    tile_dimensions,
):
    unpack_tilize(
        formats,
        unpack_to_dest=False,
        dest_acc=DestAccumulation.Yes,
        tile_dimensions=tile_dimensions,
    )


@pytest.mark.parametrize(
    "formats",
    input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
)
def test_unpack_tilize_partial_block(formats):
    unpack_tilize(formats, input_dimensions=[96, 96])


def unpack_tilize(
    formats,
    unpack_to_dest=False,
    validate_lossless=False,
    dest_acc=None,
    tile_dimensions=None,
    input_dimensions=None,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    input_dimensions = input_dimensions or [64, 64]
    # 32x32 is four faces; 16x32 is two (face_r_dim stays 16). num_faces follows the tile shape.
    tile_dimensions = list(tile_dimensions or [32, 32])
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    assert face_r_dim == 16, "unpack_tilize runs full-face tiles only"
    num_faces = num_faces_r_dim * num_faces_c_dim
    if is_perf:
        src_A = src_B = None
        tile_cnt_A = tile_cnt_B = (input_dimensions[0] * input_dimensions[1]) // (
            TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]
        )
        golden_tensor = None
    else:
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=input_dimensions,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=input_dimensions,
        )
        generate_golden = get_golden_generator(TilizeGolden)
        golden_tensor = generate_golden(
            src_A, input_dimensions, formats.output_format, num_faces=num_faces
        )

    num_blocks, num_tiles_in_block = _get_block_config(
        formats, dest_acc, unpack_to_dest, tile_cnt_A
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]

    test_config_kwargs = {
        "test_name": "sources/unpack_tilize_test.cpp",
        "formats": formats,
        "templates": [
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        "runtimes": [
            generate_input_dim(input_dimensions, input_dimensions),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        "unpack_to_dest": unpack_to_dest,
        **({"dest_acc": dest_acc} if dest_acc is not None else {}),
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    # When num_faces < 4, hardware returns full tiles (1024 elements each) but only
    # the first (num_faces * 256) elements per tile contain valid data
    if num_faces < FACES_PER_TILE and len(res_from_L1) > len(golden_tensor):
        elements_per_valid_face = (
            num_faces * ELEMENTS_PER_FACE
        )  # Valid elements per tile
        full_tile_size = (
            FACES_PER_TILE * ELEMENTS_PER_FACE
        )  # 1024 elements (full tile from hardware)
        res_tensor_list = []
        for tile_idx in range(tile_cnt_A):
            tile_start = tile_idx * full_tile_size
            # Extract only the valid faces from this tile
            tile_data = res_from_L1[tile_start : tile_start + elements_per_valid_face]
            res_tensor_list.extend(tile_data)
        res_from_L1 = res_tensor_list

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    if validate_lossless:
        # Lossless validation
        diff = golden_tensor - res_tensor
        abs_diff = diff.abs()
        assert torch.allclose(golden_tensor, res_tensor, atol=0, rtol=1e-6), (
            f"Float32 tilize lost precision! Input and output differ.\n"
            f"Max difference: {abs_diff.max().item()}\n"
            f"Num different elements: {(abs_diff > 1e-6).sum()}\n"
            f"Expected (golden): {golden_tensor[:10]}\n"
            f"Got (result): {res_tensor[:10]}"
        )
    else:
        # Standard validation with relaxed tolerances
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format, print_errors=False
        ), "Assert against golden failed"
