# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 synchronization through owners: complete and migrate, with Dest tensors at fixed rows.

The kernel (llk2_sync_compute_api_test.cpp) runs silu(x) four times, moving the SFPU between the math and the pack
thread and reusing two Dest tensors; every round must give the same result.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation, format_dict
from helpers.param_config import parametrize
from helpers.sfpu_accuracy_budget import accuracy_contract
from helpers.sfpu_domains import exclude_undefined, for_op_pipeline
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter
from helpers.utils import passed_test
from llk2_paths import cpp_source, include_dirs

ROUNDS = 4  # the kernel's rounds, each packed to its own copy of the output


@dataclass
class LLK2_SYNC(TemplateParameter):
    rows: int
    cols: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t ROWS = {self.rows};\nconstexpr std::uint32_t COLS = {self.cols};"


# bf16 in both Dest modes; fp32 in the 32-bit Dest mode, unpacked straight to Dest.
FORMATS_AND_DEST = [
    (InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b), DestAccumulation.No),
    (InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b), DestAccumulation.Yes),
    (InputOutputFormat(DataFormat.Float32, DataFormat.Float32), DestAccumulation.Yes),
]


@parametrize(format_and_dest=FORMATS_AND_DEST, input_dimensions=[[32, 32], [64, 64]])
def test_llk2_sync(format_and_dest, input_dimensions):
    formats, dest_acc = format_and_dest
    mathop = MathOperation.Silu
    torch.manual_seed(0)

    spec_A = exclude_undefined(mathop, for_op_pipeline(mathop, formats.input_format, formats.output_format).spec_A)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )
    golden = get_golden_generator(UnarySFPUGolden)(
        mathop, src_A, formats.output_format, dest_acc, formats.input_format, input_dimensions
    ).repeat(ROUNDS)

    configuration = TestConfig(
        cpp_source("llk2_sync_compute_api_test.cpp"),
        formats,
        templates=[DEST_SYNC(), LLK2_SYNC(input_dimensions[0], input_dimensions[1])],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A * ROUNDS,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(golden), "Result tensor and golden tensor are not of the same length"

    contract = accuracy_contract(
        mathop,
        output_format=formats.output_format,
        input_format=formats.input_format,
        approx_mode=ApproximationMode.No,
        dest_acc=dest_acc,
        arch=TestConfig.CHIP_ARCH,
    )
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert passed_test(golden, res_tensor, formats.output_format, **contract.tolerance_kwargs()), "Assert against golden failed"
