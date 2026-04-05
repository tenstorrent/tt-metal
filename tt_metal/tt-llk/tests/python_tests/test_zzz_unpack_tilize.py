# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    ELEMENTS_PER_FACE,
    FACES_PER_TILE,
    TILE_DIMENSIONS,
    TilizeGolden,
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
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,  # Unpack Tilize doesn't work for block float formats (Bfp8_b) due to shared exponent at start of input tensor
            DataFormat.Fp8_e4m3,
        ]
    ),
    num_faces=[2, 4],
)
def test_unpack_tilize_float(formats, num_faces, workers_tensix_coordinates):
    if (
        formats.input_format == DataFormat.Fp8_e4m3
        or formats.output_format == DataFormat.Fp8_e4m3
    ) and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "Unpack Tilize does not support Fp8_e4m3 format on non-BLACKHOLE architectures"
        )

    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Unpack Tilize does not support Bfp8_b input format")

    if formats.output_format == DataFormat.Bfp8_b and num_faces != FACES_PER_TILE:
        pytest.skip("Bfp8_b output format only works with num_faces=4")

    unpack_tilize(formats, workers_tensix_coordinates, num_faces=num_faces)


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.Yes],
    num_faces=[2, 4],
)
def test_unpack_tilize_float32_lossless(
    formats, dest_acc, num_faces, workers_tensix_coordinates
):
    unpack_tilize(
        formats,
        workers_tensix_coordinates,
        unpack_to_dest=True,
        validate_lossless=True,
        dest_acc=dest_acc,
        num_faces=num_faces,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_faces=[2, 4],
)
def test_unpack_tilize_int(formats, num_faces, workers_tensix_coordinates):
    unpack_tilize(
        formats, workers_tensix_coordinates, unpack_to_dest=True, num_faces=num_faces
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int8]),
    num_faces=[2, 4],
)
def test_unpack_tilize_int8(formats, num_faces, workers_tensix_coordinates):
    unpack_tilize(
        formats,
        workers_tensix_coordinates,
        unpack_to_dest=False,
        dest_acc=DestAccumulation.Yes,
        num_faces=num_faces,
    )


def unpack_tilize(
    formats,
    workers_tensix_coordinates,
    unpack_to_dest=False,
    validate_lossless=False,
    dest_acc=None,
    num_faces=4,
):
    input_dimensions = [64, 64]
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

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
    )

    configuration = TestConfig(
        "sources/unpack_tilize_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
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
        ),
        unpack_to_dest=unpack_to_dest,
        **({"dest_acc": dest_acc} if dest_acc is not None else {}),
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

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

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

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
