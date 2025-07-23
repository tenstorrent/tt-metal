# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import format_dict
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test


@parametrize(
    test_name="unpack_tilize_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,  # Unpack Tilize doesn't work for block float formats (Bfp8_b) due to shared exponent at start of input tensor
        ]
    ),
)
def test_unpack_tilize_float(test_name, formats):
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Unpack Tilize does not support Bfp8_b input format")

    unpack_tilize(test_name, formats)


@parametrize(
    test_name="unpack_tilize_test", formats=input_output_formats([DataFormat.Int32])
)
def test_unpack_tilize_int(test_name, formats):
    unpack_tilize(test_name, formats)


def unpack_tilize(test_name, formats):
    input_dimensions = [64, 64]

    src_A, _, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )
    src_B = torch.full((1024 * tile_cnt,), 0)

    generate_golden = get_golden_generator(TilizeGolden)
    golden_tensor = generate_golden(src_A, input_dimensions, formats.output_format)

    res_address = write_stimuli_to_l1(
        src_A, src_B, formats.input_format, formats.input_format, tile_count=tile_cnt
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "tile_cnt": tile_cnt,
        "input_dimensions": input_dimensions,
        "unpack_to_dest": formats.input_format == DataFormat.Int32,
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
