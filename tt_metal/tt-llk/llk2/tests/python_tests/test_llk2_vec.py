# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 SFPU activations (llk2::vec_*) through the Compute API.

One kernel (llk2_vec_compute_api_test.cpp) picks the activation from a compile-time parameter; only the golden
changes per activation. Stimuli domains, goldens and tolerances are the ones test_eltwise_unary_sfpu.py uses.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.sfpu_accuracy_budget import accuracy_contract
from helpers.sfpu_domains import exclude_undefined, for_op_pipeline
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter
from helpers.utils import passed_test
from llk2_paths import cpp_source, include_dirs

# The llk2::VecFunction enumerator (resolver/vec_function.h) for each activation, and the harness op whose golden,
# domain and tolerance it uses.
VEC_FUNCTIONS = {
    "silu": ("Silu", MathOperation.Silu),
    "tanh": ("Tanh", MathOperation.Tanh),
    "gelu": ("Gelu", MathOperation.Gelu),
    "gelu_tanh": ("GeluTanh", MathOperation.GeluTanh),
    "relu_max": ("ReluMax", MathOperation.ReluMax),
    "sigmoid": ("Sigmoid", MathOperation.Sigmoid),
    "hardsigmoid": ("Hardsigmoid", MathOperation.Hardsigmoid),
    "hardtanh": ("Hardtanh", MathOperation.Hardtanh),
    "selu": ("Selu", MathOperation.Selu),
    "softplus": ("Softplus", MathOperation.Softplus),
}

# Activations the target kernel also runs in their fast (approximate) form.
HAS_FAST_FORM = {"tanh", "gelu", "gelu_tanh", "sigmoid"}

# Where the fast form is a different kernel with its own harness op (domain and tolerance), that op.
FAST_FORM_OP = {"sigmoid": MathOperation.SigmoidAppx}


# The llk2::Thread enumerator (tensor/tensor.h) for the thread that drives the SFPU.
VEC_THREADS = {"math": "Math", "pack": "Pack"}


@dataclass
class LLK2_VEC(TemplateParameter):
    function: str
    fast: bool
    thread: str
    rows: int
    cols: int

    def convert_to_cpp(self) -> str:
        return (
            '#include "resolver/vec_function.h"\n'
            f"constexpr auto VEC_FUNCTION = llk2::VecFunction::{self.function};\n"
            f"constexpr bool VEC_FAST = {str(self.fast).lower()};\n"
            f"constexpr auto VEC_THREAD = llk2::Thread::{self.thread};\n"
            f"constexpr std::uint32_t ROWS = {self.rows};\n"
            f"constexpr std::uint32_t COLS = {self.cols};"
        )


def _variants():
    variants = []
    for name in VEC_FUNCTIONS:
        for fast in (False, True) if name in HAS_FAST_FORM else (False,):
            variants.append((name, fast))
    return variants


# bf16 in both Dest modes; fp32 in the 32-bit Dest mode, unpacked straight to Dest.
FORMATS_AND_DEST = [
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
    ),
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.Yes,
    ),
    (InputOutputFormat(DataFormat.Float32, DataFormat.Float32), DestAccumulation.Yes),
]


@parametrize(
    variant=_variants(),
    format_and_dest=FORMATS_AND_DEST,
    thread=list(VEC_THREADS),
    input_dimensions=[[32, 32], [64, 64]],
)
def test_llk2_vec(variant, format_and_dest, thread, input_dimensions):
    name, fast = variant
    formats, dest_acc = format_and_dest
    function, mathop = VEC_FUNCTIONS[name]
    if fast:
        mathop = FAST_FORM_OP.get(name, mathop)
    approx_mode = ApproximationMode.Yes if fast else ApproximationMode.No
    torch.manual_seed(0)

    spec_A = exclude_undefined(
        mathop,
        for_op_pipeline(
            mathop,
            formats.input_format,
            formats.output_format,
            approx_mode=approx_mode,
        ).spec_A,
    )
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )

    golden = get_golden_generator(UnarySFPUGolden)(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    configuration = TestConfig(
        cpp_source("llk2_vec_compute_api_test.cpp"),
        formats,
        templates=[
            DEST_SYNC(),
            LLK2_VEC(
                function,
                fast,
                VEC_THREADS[thread],
                input_dimensions[0],
                input_dimensions[1],
            ),
        ],
        runtimes=[],
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
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.Yes,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    contract = accuracy_contract(
        mathop,
        output_format=formats.output_format,
        input_format=formats.input_format,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=TestConfig.CHIP_ARCH,
    )
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden, res_tensor, formats.output_format, **contract.tolerance_kwargs()
    ), "Assert against golden failed"
