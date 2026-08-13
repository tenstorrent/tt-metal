# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Smoke and invariant tests for SFPI kernels with specialized tile layouts."""

from dataclasses import dataclass
from itertools import product

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    NUM_FACES,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
    TemplateParameter,
)
from helpers.tile_constants import MAX_NUM_FACES, MAX_TILE_ELEMENTS
from helpers.utils import passed_test

_CPP_SOURCE = "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp"


@dataclass(frozen=True)
class SfpiSpecialCase:
    name: str
    kernel: str
    function: str
    init: str
    vector_mode: str
    formats: tuple[tuple[InputOutputFormat, DestAccumulation], ...]
    tile_count: int = 1
    integer: bool = False
    template_args: str = "APPROX_MODE"
    runtime_args: str = ""

    def __repr__(self) -> str:
        return self.name


@dataclass
class SFPI_SPECIAL_OPERATION(TemplateParameter):
    case: SfpiSpecialCase

    def convert_to_cpp(self) -> str:
        op = {
            "alt_complex_rotate90": "alt_complex_rotate90",
            "int_sum_col": "int_sum_col",
            "int_sum_row": "int_sum_row",
            "bitwise_and": "unary_bitwise_and",
            "bitwise_or": "unary_bitwise_or",
            "bitwise_xor": "unary_bitwise_xor",
            "mask_float": "mask",
            "mask_int32": "int_mask",
            "tiled_prod": "tiled_prod",
        }[self.case.name]
        return f"constexpr auto SFPU_UNARY_OPERATION = SfpuType::{op};"


_BF16_FP32 = (
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
    ),
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.Yes,
    ),
    (InputOutputFormat(DataFormat.Float32, DataFormat.Float32), DestAccumulation.Yes),
)
_INT32 = (
    (InputOutputFormat(DataFormat.Int32, DataFormat.Int32), DestAccumulation.Yes),
)

SFPI_SPECIAL_CASES = (
    SfpiSpecialCase(
        "alt_complex_rotate90",
        "alt_complex_rotate90",
        "calculate_alt_complex_rotate90",
        "alt_complex_rotate90_init()",
        "RC",
        _BF16_FP32,
    ),
    SfpiSpecialCase(
        "int_sum_col",
        "int_sum",
        "calculate_sum_int_col",
        "sum_int_init<APPROX_MODE>()",
        "R",
        _INT32,
        integer=True,
    ),
    SfpiSpecialCase(
        "int_sum_row",
        "int_sum",
        "calculate_sum_int_row",
        "sum_int_init<APPROX_MODE>()",
        "C",
        _INT32,
        integer=True,
    ),
    SfpiSpecialCase(
        "bitwise_and",
        "bitwise",
        "calculate_sfpu_unary_bitwise",
        "bitwise_and_init()",
        "RC",
        _INT32,
        integer=True,
        template_args="APPROX_MODE, UnaryBitwiseOp::AND, DataFormat::Int32, SFPU_ITERATIONS",
        runtime_args=", 0x55u",
    ),
    SfpiSpecialCase(
        "bitwise_or",
        "bitwise",
        "calculate_sfpu_unary_bitwise",
        "bitwise_or_init()",
        "RC",
        _INT32,
        integer=True,
        template_args="APPROX_MODE, UnaryBitwiseOp::OR, DataFormat::Int32, SFPU_ITERATIONS",
        runtime_args=", 0x55u",
    ),
    SfpiSpecialCase(
        "bitwise_xor",
        "bitwise",
        "calculate_sfpu_unary_bitwise",
        "bitwise_xor_init()",
        "RC",
        _INT32,
        integer=True,
        template_args="APPROX_MODE, UnaryBitwiseOp::XOR, DataFormat::Int32, SFPU_ITERATIONS",
        runtime_args=", 0x55u",
    ),
    SfpiSpecialCase(
        "mask_float",
        "mask",
        "calculate_mask",
        "mask_init()",
        "RC",
        _BF16_FP32,
        tile_count=2,
    ),
    SfpiSpecialCase(
        "mask_int32",
        "mask",
        "calculate_int_mask",
        "mask_init()",
        "RC",
        _INT32,
        tile_count=2,
        integer=True,
    ),
    SfpiSpecialCase(
        "tiled_prod",
        "tiled_prod",
        "calculate_tiled_prod",
        "tiled_prod_init()",
        "RC",
        _BF16_FP32,
    ),
)


def _invariant_stimuli(case, formats):
    dtype = format_dict[formats.input_format]
    # These values are invariants of the specialized layout transforms.  They
    # let the test validate dispatch and every supported format without baking
    # Blackhole's physical Dest-register layout into a Quasar test.
    fill = 1 if case.name == "tiled_prod" else 0
    if case.name.startswith("bitwise_"):
        source = torch.arange(MAX_TILE_ELEMENTS, dtype=dtype)
        if case.name == "bitwise_and":
            golden = torch.bitwise_and(source, 0x55)
        elif case.name == "bitwise_or":
            golden = torch.bitwise_or(source, 0x55)
        else:
            golden = torch.bitwise_xor(source, 0x55)
    elif case.name.startswith("mask"):
        data = torch.ones(MAX_TILE_ELEMENTS, dtype=dtype)
        mask = torch.zeros(MAX_TILE_ELEMENTS, dtype=dtype)
        source = torch.cat((data, mask))
        golden = torch.zeros_like(source)
    else:
        source = torch.full((case.tile_count * MAX_TILE_ELEMENTS,), fill, dtype=dtype)
        golden = source.clone()
    return source, golden


_VARIANTS = [
    (case, formats, dest_acc, dest_sync, implied_math_format)
    for case in SFPI_SPECIAL_CASES
    for (formats, dest_acc), dest_sync, implied_math_format in product(
        case.formats,
        (DestSync.Half, DestSync.Full),
        (ImpliedMathFormat.No, ImpliedMathFormat.Yes),
    )
]


@pytest.mark.quasar
@pytest.mark.parametrize(
    "case,formats,dest_acc,dest_sync,implied_math_format",
    _VARIANTS,
    ids=lambda value: repr(value) if isinstance(value, SfpiSpecialCase) else None,
)
def test_sfpi_compat_special_quasar(
    case, formats, dest_acc, dest_sync, implied_math_format
):
    source, golden = _invariant_stimuli(case, formats)
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SFPI_SPECIAL_OPERATION(case),
            APPROX_MODE(ApproximationMode.No),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
            TYPECAST_FORMATS(),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            TILE_COUNT(case.tile_count),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            LOOP_FACTOR(1),
        ],
        variant_stimuli=StimuliConfig(
            source,
            formats.input_format,
            torch.zeros_like(source),
            formats.input_format,
            formats.output_format,
            tile_count_A=case.tile_count,
            tile_count_B=case.tile_count,
            tile_count_res=case.tile_count,
            num_faces=MAX_NUM_FACES,
            twos_complement=case.integer,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    result = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    assert passed_test(golden, result, formats.output_format)
