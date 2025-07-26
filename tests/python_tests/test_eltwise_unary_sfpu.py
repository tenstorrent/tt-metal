# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test


@parametrize(
    test_name="eltwise_unary_sfpu_test",
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    approx_mode=[ApproximationMode.No, ApproximationMode.Yes],
    mathop=[
        MathOperation.Abs,
        MathOperation.Atanh,
        MathOperation.Asinh,
        MathOperation.Acosh,
        MathOperation.Cos,
        MathOperation.Log,
        MathOperation.Reciprocal,
        MathOperation.Sin,
        MathOperation.Sqrt,
        MathOperation.Square,
        MathOperation.Celu,
        MathOperation.Silu,
        MathOperation.Gelu,
        MathOperation.Neg,
        MathOperation.Fill,
        MathOperation.Elu,
        MathOperation.Exp,
        MathOperation.Exp2,
        MathOperation.Hardsigmoid,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_unary_sfpu_float(test_name, formats, approx_mode, mathop, dest_acc):
    arch = get_chip_architecture()

    if dest_acc == DestAccumulation.No and arch == ChipArchitecture.BLACKHOLE:
        if formats.input_format == DataFormat.Float16 or formats == InputOutputFormat(
            DataFormat.Float32, DataFormat.Float16
        ):
            pytest.skip(reason="This combination is not supported on BH architecture")

    if (
        approx_mode == ApproximationMode.Yes
        and mathop in [MathOperation.Exp, MathOperation.Exp2, MathOperation.Elu]
        and (
            formats.input_format == DataFormat.Bfp8_b
            or formats.output_format == DataFormat.Bfp8_b
        )
    ):
        pytest.skip(
            reason="Exp-related operations are not supported for bf8_b format in approximation mode."
        )

    eltwise_unary_sfpu(test_name, formats, dest_acc, approx_mode, mathop)


@parametrize(
    test_name="eltwise_unary_sfpu_int",
    formats=input_output_formats([DataFormat.Int32]),
    approx_mode=[ApproximationMode.No, ApproximationMode.Yes],
    mathop=[
        MathOperation.Neg,
        MathOperation.Fill,
    ],
    dest_acc=[DestAccumulation.Yes],
)
def test_eltwise_unary_sfpu_int(test_name, formats, approx_mode, mathop, dest_acc):
    if formats.input_format == DataFormat.Int32:
        pytest.skip(reason=f"Int32 tests break fast tilize, tracked in #495")

    eltwise_unary_sfpu(test_name, formats, dest_acc, approx_mode, mathop)


def eltwise_unary_sfpu(test_name, formats, dest_acc, approx_mode, mathop):
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)
    input_dimensions = [64, 64]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop, src_A, formats.output_format, dest_acc, formats.input_format
    )

    res_address = write_stimuli_to_l1(
        src_A, src_B, formats.input_format, formats.input_format, tile_count=tile_cnt
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit()
        and dest_acc
        == DestAccumulation.Yes  # If dest_acc is off, we unpack Float32 into 16-bit format in src registers (later copied over in dest reg for SFPU op)
    )
    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "approx_mode": approx_mode,
        "unpack_to_dest": unpack_to_dest,
        "tile_cnt": tile_cnt,
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    # res_from_L1 = res_from_L1[:1024]
    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
