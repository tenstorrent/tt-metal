# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import (
    collect_results,
    run_elf_files,
    wait_for_tensix_operations_finished,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import generate_make_command
from helpers.utils import passed_test, run_shell_command

# SUPPORTED FORMATS FOR TEST
supported_formats = [DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b]

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
all_params = generate_params(
    ["eltwise_unary_sfpu_test"],
    test_formats,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    approx_mode=[ApproximationMode.No, ApproximationMode.Yes],
    mathop=[
        MathOperation.Abs,
        MathOperation.Cos,
        MathOperation.Log,
        MathOperation.Reciprocal,
        MathOperation.Sin,
        MathOperation.Sqrt,
        MathOperation.Square,
        MathOperation.Celu,
    ],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, approx_mode, mathop",
    clean_params(all_params),
    ids=param_ids,
)
def test_eltwise_unary_sfpu(testname, formats, dest_acc, approx_mode, mathop):
    arch = get_chip_architecture()
    if (
        formats.input_format in [DataFormat.Float32, DataFormat.Int32]
        and dest_acc != DestAccumulation.Yes
    ):
        pytest.skip(
            reason="Skipping test for 32 bit wide data without 32 bit accumulation in Dest"
        )

    if formats.input_format == DataFormat.Float16 and (
        dest_acc == DestAccumulation.No and arch == ChipArchitecture.BLACKHOLE
    ):
        pytest.skip(reason="This combination is not fully implemented in testing")

    src_A, src_B = generate_stimuli(
        formats.input_format,
        formats.input_format,
    )
    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(mathop, src_A, formats.output_format)
    write_stimuli_to_l1(src_A, src_B, formats.input_format, formats.input_format)

    unpack_to_dest = formats.input_format.is_32_bit()
    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "approx_mode": approx_mode,
        "unpack_to_dest": unpack_to_dest,  # This test does a datacopy and unpacks input into dest register
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")
    run_elf_files(testname)

    wait_for_tensix_operations_finished()
    res_from_L1 = collect_results(formats, tensor_size=len(src_A))
    res_from_L1 = res_from_L1[:1024]
    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
