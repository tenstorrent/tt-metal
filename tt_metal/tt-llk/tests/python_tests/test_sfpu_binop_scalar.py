# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import ScalarBinopGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    SFPU_BINOP_MODE,
    SFPU_UNARY_SCALAR,
)
from helpers.utils import passed_test


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


_DIVISOR = 2.0
_SCALAR_BITS = {
    MathOperation.ScalarAdd: _bits(2.0),
    MathOperation.ScalarSub: _bits(2.0),
    MathOperation.ScalarMul: _bits(2.0),
    MathOperation.ScalarDiv: _bits(1.0 / _DIVISOR),
    MathOperation.ScalarRsub: _bits(2.0),
}


def _run_sfpu_binop_scalar(formats, dest_acc, mathop, input_dimensions=[32, 32]):
    scalar_bits = _SCALAR_BITS[mathop]

    # Keep inputs small and bounded so the bf16 result stays accurate across all
    # five scalar ops (add/sub/mul/div/rsub) and both dest-accumulation modes.
    spec_a = StimuliSpec.uniform(low=-1.0, high=1.0)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_a,
    )

    generate_golden = get_golden_generator(ScalarBinopGolden)
    golden = generate_golden(mathop, src_A, scalar_bits, formats.output_format)

    configuration = TestConfig(
        "sources/sfpu_binop_scalar_test.cpp",
        formats,
        templates=[
            SFPU_BINOP_MODE(mathop),
            SFPU_UNARY_SCALAR(scalar_bits),
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[:1024]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    golden_tensor = torch.tensor(golden, dtype=torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=[
        MathOperation.ScalarAdd,
        MathOperation.ScalarSub,
        MathOperation.ScalarMul,
        MathOperation.ScalarDiv,
        MathOperation.ScalarRsub,
    ],
)
def test_sfpu_binop_scalar(formats, dest_acc, mathop):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip("Float16_b not supported with DestAccumulation.Yes")

    _run_sfpu_binop_scalar(formats, dest_acc, mathop)
