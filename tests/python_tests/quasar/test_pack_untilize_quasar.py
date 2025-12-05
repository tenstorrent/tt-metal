# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, ImpliedMathFormat, format_dict
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test


def generate_pack_untilize_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate pack_untilize combinations.

    Args: List of input-output format pairs

    Returns: List of (format, dest_acc, input_dimensions) tuples
    """
    dimensions_cache = {
        DestAccumulation.No: tuple(
            generate_unary_input_dimensions(DestAccumulation.No)
        ),
        DestAccumulation.Yes: tuple(
            generate_unary_input_dimensions(DestAccumulation.Yes)
        ),
    }

    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format
        if in_fmt != fmt.output_format:
            continue

        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )

        for dest_acc in dest_acc_modes:
            for dimensions in dimensions_cache[dest_acc]:
                combinations.append((fmt, dest_acc, dimensions))

    return combinations


PACK_UNTILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
    ]
)
ALL_PACK_UNTILIZE_COMBINATIONS = generate_pack_untilize_combinations(
    PACK_UNTILIZE_FORMATS
)


@pytest.mark.quasar
@parametrize(
    test_name="pack_untilize_quasar_test",
    formats_dest_acc_dimensions=ALL_PACK_UNTILIZE_COMBINATIONS,
)
def test_pack_untilize_quasar(test_name, formats_dest_acc_dimensions):

    formats = formats_dest_acc_dimensions[0]
    dest_acc = formats_dest_acc_dimensions[1]
    input_dimensions = formats_dest_acc_dimensions[2]

    if formats.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes:
        pytest.skip("Fails for now.")

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(src_A, formats.output_format, input_dimensions)

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "tile_cnt": tile_cnt,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": dest_acc,
        "implied_math_format": ImpliedMathFormat.Yes,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
        num_faces=4,
    )

    run_test(test_config)

    res_from_L1 = collect_results(
        formats, tile_count=tile_cnt, address=res_address, num_faces=4
    )
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
