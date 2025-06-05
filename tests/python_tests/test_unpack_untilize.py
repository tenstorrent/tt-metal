# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from helpers.device import (
    collect_results,
    run_elf_files,
    wait_for_tensix_operations_finished,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import format_dict
from helpers.format_config import DataFormat
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import generate_make_command
from helpers.tilize_untilize import untilize
from helpers.utils import passed_test, run_shell_command


def generate_golden(operand1, data_format):

    A_untilized = untilize(operand1, data_format)
    return A_untilized.flatten()


# SUPPORTED FORMATS FOR TEST
supported_formats = [DataFormat.Float16, DataFormat.Float16_b]

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
all_params = generate_params(["unpack_untilize_test"], test_formats)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize("testname, formats", clean_params(all_params), ids=param_ids)
def test_unpack_untilze(testname, formats):

    src_A, src_B = generate_stimuli(formats.input_format, formats.input_format)
    src_B = torch.full((1024,), 0)

    golden_tensor = generate_golden(src_A, formats.output_format)

    write_stimuli_to_l1(src_A, src_B, formats.input_format, formats.input_format)

    test_config = {
        "formats": formats,
        "testname": testname,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)
    wait_for_tensix_operations_finished()
    res_from_L1 = collect_results(formats, tensor_size=len(src_A))
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
