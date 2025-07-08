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
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test

# SUPPORTED FORMATS FOR TEST
supported_formats = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
    DataFormat.Bfp8_b,
]  # Pack Untilize doesn't work for block float formats (Bfp8_b); we only include as input format in our test

#   INPUT-OUTPUT FORMAT SWEEP
#   input_output_formats(supported_formats)

#   FULL FORMAT SWEEP
#   format_combination_sweep(formats=supported_formats, all_same=False, same_src_reg_format=True)

#   SPECIFIC FORMAT COMBINATION
#   generate_combination(
#       [(DataFormat.Float16_b,  # index 0 is for unpack_A_src
#         DataFormat.Float16_b,  # index 1 is for unpack_A_dst
#         DataFormat.Float16_b,  # index 2 is for pack_src (if src registers have same formats)
#         DataFormat.Bfp8_b,  # index 3 is for pack_dst
#         DataFormat.Float16_b,  # index 4 is for math format)])

#   SPECIFIC INPUT-OUTPUT COMBINATION
#   [InputOutputFormat(DataFormat.Float16, DataFormat.Float32)]

test_formats = input_output_formats(supported_formats)
all_params = generate_params(["pack_untilize_test"], test_formats)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize("testname, formats", clean_params(all_params), ids=param_ids)
def test_pack_untilize(testname, formats):
    if formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("Pack Untilize does not support Bfp8_b format")

    input_dimensions = [32, 128]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(src_A, formats.output_format, input_dimensions)

    res_address = write_stimuli_to_l1(
        src_A, src_B, formats.input_format, formats.input_format, tile_count=tile_cnt
    )

    test_config = {
        "formats": formats,
        "testname": testname,
        "tile_cnt": tile_cnt,
        "input_dimensions": input_dimensions,
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
