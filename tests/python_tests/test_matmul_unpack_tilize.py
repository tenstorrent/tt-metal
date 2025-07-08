# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import DestAccumulation, MathFidelity, format_dict
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test

# SUPPORTED FORMATS FOR TEST
supported_formats = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
]  #  Add DataFormat.Bfp8_b only as input when Data format Inference Model 2.0 supports format conversions for > 1 pipeline run with different inputs and outputs.
#  Now tests run by requiring input format to be same as output format.
#  We cannot unpack tilize on Bfp8_b format, so it will be included only as input format.

#   INPUT-OUTPUT FORMAT SWEEP
#   input_output_formats(supported_formats)

#   FULL FORMAT SWEEP
#   format_combination_sweep(formats=supported_formats, all_same=False, same_src_reg_format=True)

#   SPECIFIC FORMAT COMBINATION
#   generate_combination(
#       [(DataFormat.Float16_b,  # index 0 is for unpack_A_src
#         DataFormat.Float16_b,  # index 1 is for unpack_A_dst
#         DataFormat.Float16_b,  # index 2 is for pack_src (if src registers have same formats)
#         DataFormat.Float16_b,  # index 3 is for pack_dst
#         DataFormat.Float16_b,  # index 4 is for math format)])

#   SPECIFIC INPUT-OUTPUT COMBINATION
#   [InputOutputFormat(DataFormat.Float16, DataFormat.Float32)]

test_formats = input_output_formats(supported_formats, same=True)
all_params = generate_params(
    ["matmul_unpack_tilize_test"],
    test_formats,
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, math_fidelity",
    clean_params(all_params),
    ids=param_ids,
)
def test_matmul_unpack_tilize(testname, formats, dest_acc, math_fidelity):

    torch_format = format_dict[formats.output_format]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format
    )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = tilize(
        generate_golden(src_A, src_B, formats.output_format, math_fidelity)
    )
    golden_tensor = golden_tensor.to(torch_format)

    res_address = write_stimuli_to_l1(
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
    )
    buffer_dest_address = 0x1E000
    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
        "L1_to_L1_iterations": 2,
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        test_config.get(
            "L1_to_L1_iterations"  # Needed to calculate accumulated percision loss for fused tests that copy result tensor as input for next runs
        ),
    )
