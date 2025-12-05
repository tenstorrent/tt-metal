# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    TilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, run_test
from helpers.utils import passed_test


def generate_unpack_tilize_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate unpack_tilize combinations.

    Rules:
    1. Tilize 32b data into dest is not yet supported

    Args: List of input-output format pairs

    Returns: List of (format, dest_acc, unpacker_sel, input_dimensions) tuples
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

        if in_fmt.is_32_bit():
            continue  # Tilize 32b data into dest not yet supported

        dest_acc_modes = (DestAccumulation.No, DestAccumulation.Yes)
        unpacker_engines = (UnpackerEngine.UnpA, UnpackerEngine.UnpB)

        for dest_acc in dest_acc_modes:
            for unpacker_sel in unpacker_engines:
                for dimensions in dimensions_cache[dest_acc]:
                    combinations.append((fmt, dest_acc, unpacker_sel, dimensions))

    return combinations


UNPACK_TILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
    ]
)
ALL_UNPACK_TILIZE_COMBINATIONS = generate_unpack_tilize_combinations(
    UNPACK_TILIZE_FORMATS
)


@pytest.mark.quasar
@parametrize(
    test_name="unpack_tilize_quasar_test",
    formats_dest_acc_unpack_sel_dimensions=ALL_UNPACK_TILIZE_COMBINATIONS,
)
def test_unpack_tilize_quasar(
    test_name, formats_dest_acc_unpack_sel_dimensions, boot_mode=BootMode.DEFAULT
):
    formats = formats_dest_acc_unpack_sel_dimensions[0]
    dest_acc = formats_dest_acc_unpack_sel_dimensions[1]
    unpacker_sel = formats_dest_acc_unpack_sel_dimensions[2]
    input_dimensions = formats_dest_acc_unpack_sel_dimensions[3]

    if formats.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes:
        pytest.skip("Fails for now.")

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )

    generate_golden = get_golden_generator(TilizeGolden)
    golden_src = src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    golden_tensor = generate_golden(
        golden_src, input_dimensions, formats.output_format, num_faces=4
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "unpack_to_dest": unpack_to_dest,
        "tile_cnt": tile_cnt,
        "unpacker_engine_sel": unpacker_sel,
        "implied_math_format": ImpliedMathFormat.Yes,
        "data_copy_type": (
            DataCopyType.B2D
            if unpacker_sel == UnpackerEngine.UnpB
            else DataCopyType.A2D
        ),
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

    run_test(test_config, boot_mode=boot_mode)

    res_from_L1 = collect_results(
        formats, tile_count=tile_cnt, address=res_address, num_faces=4
    )

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
