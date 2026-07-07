# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import TernarySFPUGolden, get_golden_generator
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
    DISABLE_SRC_ZERO_FLAG,
    SFPU_TERNARY_OP,
    SFPU_TERNARY_SCALAR,
)
from helpers.utils import passed_test

_SCALAR_VALUE = 2.0
_SCALAR_VALUE_BITS = struct.unpack("<I", struct.pack("<f", _SCALAR_VALUE))[0]


def _run_sfpu_ternary(formats, dest_acc, mathop, input_dimensions=[32, 32]):

    _divide_by_c = mathop in (MathOperation.SfpuAddcdiv, MathOperation.SfpuSnakeBeta)
    spec_ab = StimuliSpec.uniform(low=-1.0, high=1.0)
    spec_c = (
        StimuliSpec.uniform(low=1.0, high=2.0)
        if _divide_by_c
        else StimuliSpec.uniform(low=-1.0, high=1.0)
    )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_ab,
        spec_B=spec_ab,
    )

    src_C, tile_cnt_C, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_c,
        spec_B=spec_c,
    )

    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden = generate_golden(
        mathop,
        src_A,
        src_B,
        src_C,
        _SCALAR_VALUE_BITS,
        formats.output_format,
    )

    configuration = TestConfig(
        "sources/sfpu_ternary_test.cpp",
        formats,
        templates=[
            SFPU_TERNARY_OP(mathop),
            SFPU_TERNARY_SCALAR(_SCALAR_VALUE_BITS),
            APPROX_MODE(ApproximationMode.No),
            DISABLE_SRC_ZERO_FLAG(True),
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
            buffer_C=src_C.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_cnt_C,
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
            DataFormat.Bfp8_b,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=[
        MathOperation.SfpuAddcmul,
        MathOperation.SfpuAddcdiv,
        MathOperation.SfpuLerp,
        MathOperation.SfpuSnakeBeta,
    ],
)
def test_sfpu_ternary(formats, dest_acc, mathop):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")
    if (
        formats.input_format == DataFormat.Bfp8_b
        and mathop != MathOperation.SfpuAddcmul
    ):
        pytest.skip("Bfp8_b is only supported for addcmul")

    _run_sfpu_ternary(formats, dest_acc, mathop)
