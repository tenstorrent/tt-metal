# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    TILE_COUNT,
)
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,  # Unpack Tilize doesn't work for block float formats (Bfp8_b) due to shared exponent at start of input tensor
        ]
    ),
)
def test_unpack_tilize_float(formats, workers_tensix_coordinates):
    formats = formats[0]
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Unpack Tilize does not support Bfp8_b input format")

    unpack_tilize(formats, workers_tensix_coordinates)


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.Yes],
)
def test_unpack_tilize_float32_lossless(formats, dest_acc, workers_tensix_coordinates):
    unpack_tilize(
        formats,
        workers_tensix_coordinates,
        unpack_to_dest=True,
        validate_lossless=True,
        dest_acc=dest_acc,
    )


@parametrize(formats=input_output_formats([DataFormat.Int32]))
def test_unpack_tilize_int(formats, workers_tensix_coordinates):
    formats = formats[0]
    unpack_tilize(formats, workers_tensix_coordinates, unpack_to_dest=True)


def unpack_tilize(
    formats,
    workers_tensix_coordinates,
    unpack_to_dest=False,
    validate_lossless=False,
    dest_acc=None,
):
    input_dimensions = [64, 64]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(TilizeGolden)
    golden_tensor = generate_golden(src_A, input_dimensions, formats.output_format)

    configuration = TestConfig(
        "sources/unpack_tilize_test.cpp",
        formats,
        templates=[INPUT_DIMENSIONS(input_dimensions, input_dimensions)],
        runtimes=[TILE_COUNT(tile_cnt_A)],
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

    res_from_L1 = configuration.run(workers_tensix_coordinates)

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
            golden_tensor, res_tensor, formats.output_format
        ), "Assert against golden failed"
