# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test pack operation with various configurations.

Tests the LLK pack kernel with:
- Different data formats (Float16_b, Float16, Float32, Int32, Bfp8_b)
- Destination accumulation modes
- Variable tile dimensions
- ReLU activation
- Destination sync modes (SyncHalf for double-buffering, SyncFull for single-buffering)
"""

import pytest
import torch
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_dest_indices,
)
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import PackGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DstSync,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test


@parametrize(
    test_name="pack_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Bfp8_b,
        ]
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    input_dimensions=[[32, 32], [64, 64], [32, 64], [64, 32]],
    relu_config=[
        0,
        1,
    ],  # 0 stands for NO_RELU, and 1 for ZERO_RELU. TODO: Use Enum when available.
    dst_sync=[DstSync.SyncHalf, DstSync.SyncFull],
    dest_index=lambda dest_acc, dst_sync, input_dimensions: get_valid_dest_indices(
        dest_sync=dst_sync,
        dest_acc=dest_acc,
        tile_count=(input_dimensions[0] * input_dimensions[1]) // (32 * 32),
    ),
)
def test_pack(
    test_name, formats, dest_acc, input_dimensions, relu_config, dst_sync, dest_index
):

    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        pytest.skip(
            "Pack does not support mixing Int32 with other formats. Check format conversions in packer for more information."
        )

    # Generate test data
    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )

    # Generate golden output
    generate_golden = get_golden_generator(PackGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions=input_dimensions,
        enable_relu=bool(relu_config),
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "tile_cnt": tile_cnt,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "unpack_to_dest": formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.Yes,
        "dest_acc": dest_acc,
        "relu_config": relu_config,
        "dst_sync": dst_sync,
        "dest_index": dest_index,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    run_test(test_config)

    res_from_L1 = collect_results(
        formats,
        tile_count=tile_cnt,
        address=res_address,
    )
    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
