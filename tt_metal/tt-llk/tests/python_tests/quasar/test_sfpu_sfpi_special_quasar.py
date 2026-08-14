# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar coverage for full-tile SFPI rotate, bitwise, int-sum, and product kernels.

These implementations use fixed 32x32 face/register offsets.  This module
therefore intentionally tests only the legal full-tile layout, while sweeping
the supported L1 formats, Dest widths, dvalid synchronization modes, implied
math modes, scalar masks, and the two-tile add path.
"""

from dataclasses import dataclass
from enum import IntEnum

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    RuntimeParameter,
    TemplateParameter,
)
from helpers.tile_shape import construct_tile_shape
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

_CPP_SOURCE = "sources/quasar/sfpu_sfpi_special_quasar_test.cpp"
_FULL_TILE = (32, 32)


class SpecialSfpuOp(IntEnum):
    ROTATE90 = 0
    BIT_AND = 1
    BIT_OR = 2
    BIT_XOR = 3
    SUM_COL = 4
    SUM_ROW = 5
    ADD_OFFSET = 6
    TILED_PROD = 7


_FLOAT_OPS = (SpecialSfpuOp.ROTATE90, SpecialSfpuOp.TILED_PROD)
_BITWISE_OPS = (
    SpecialSfpuOp.BIT_AND,
    SpecialSfpuOp.BIT_OR,
    SpecialSfpuOp.BIT_XOR,
)
_INT_SUM_OPS = (
    SpecialSfpuOp.SUM_COL,
    SpecialSfpuOp.SUM_ROW,
    SpecialSfpuOp.ADD_OFFSET,
)
_FLOAT_FORMATS = (
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Float32,
    DataFormat.Tf32,
    DataFormat.MxFp8R,
    DataFormat.MxFp8P,
    DataFormat.MxFp4,
    DataFormat.MxInt8,
    DataFormat.MxInt4,
    DataFormat.MxInt2,
)
_BITWISE_FORMATS = (DataFormat.Int32, DataFormat.UInt16)
_MASKS = {
    DataFormat.Int32: (0x00000000, 0xFFFFFFFF, 0xAAAAAAAA),
    DataFormat.UInt16: (0x0000, 0xFFFF, 0xAAAA),
}


@dataclass
class SPECIAL_SFPU_OP(TemplateParameter):
    operation: SpecialSfpuOp

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SPECIAL_SFPU_OP = {int(self.operation)}u;"


@dataclass
class SPECIAL_DATA_FORMAT(TemplateParameter):
    data_format: DataFormat

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr auto SPECIAL_DATA_FORMAT = DataFormat::{self.data_format.name};"
        )


@dataclass
class SPECIAL_SFPU_SCALAR(RuntimeParameter):
    scalar: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SPECIAL_SFPU_SCALAR = {self.scalar}u;"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t SPECIAL_SFPU_SCALAR;", "I"


def _float_dest_acc(data_format: DataFormat) -> tuple[DestAccumulation, ...]:
    if data_format in (DataFormat.Float32, DataFormat.Tf32):
        return (DestAccumulation.Yes,)
    if data_format in (DataFormat.Float16, DataFormat.Float16_b):
        return (DestAccumulation.No, DestAccumulation.Yes)
    return (DestAccumulation.No,)


def _generate_cases():
    cases = []

    for operation in _FLOAT_OPS:
        for data_format in _FLOAT_FORMATS:
            formats = InputOutputFormat(data_format, data_format)
            implied_modes = (
                (ImpliedMathFormat.Yes,)
                if data_format.is_mx_format()
                else (ImpliedMathFormat.No, ImpliedMathFormat.Yes)
            )
            for dest_acc in _float_dest_acc(data_format):
                for dest_sync in (DestSync.Half, DestSync.Full):
                    for implied_math in implied_modes:
                        cases.append(
                            (
                                operation,
                                formats,
                                dest_acc,
                                dest_sync,
                                implied_math,
                                runtime(0),
                            )
                        )

    for operation in _BITWISE_OPS:
        for data_format in _BITWISE_FORMATS:
            formats = InputOutputFormat(data_format, data_format)
            dest_acc = (
                DestAccumulation.Yes
                if data_format == DataFormat.Int32
                else DestAccumulation.No
            )
            for scalar in _MASKS[data_format]:
                for dest_sync in (DestSync.Half, DestSync.Full):
                    for implied_math in (
                        ImpliedMathFormat.No,
                        ImpliedMathFormat.Yes,
                    ):
                        cases.append(
                            (
                                operation,
                                formats,
                                dest_acc,
                                dest_sync,
                                implied_math,
                                runtime(scalar),
                            )
                        )

    int32_formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    for operation in _INT_SUM_OPS:
        for dest_sync in (DestSync.Half, DestSync.Full):
            for implied_math in (ImpliedMathFormat.No, ImpliedMathFormat.Yes):
                cases.append(
                    (
                        operation,
                        int32_formats,
                        DestAccumulation.Yes,
                        dest_sync,
                        implied_math,
                        runtime(0),
                    )
                )

    return cases


def _tilized(tensor: torch.Tensor, data_format: DataFormat) -> torch.Tensor:
    return tilize_block(
        tensor.flatten(),
        _FULL_TILE,
        stimuli_format=data_format,
        tile_dimensions=_FULL_TILE,
    ).clone()


def _untilized(tensor: torch.Tensor, data_format: DataFormat) -> torch.Tensor:
    return untilize_block(
        tensor.flatten(),
        stimuli_format=data_format,
        dimensions=_FULL_TILE,
        tile_dimensions=_FULL_TILE,
    ).flatten()


def _float_source(operation: SpecialSfpuOp, data_format: DataFormat) -> torch.Tensor:
    dtype = format_dict[data_format]
    index = torch.arange(32 * 32, dtype=torch.int32)
    if operation == SpecialSfpuOp.ROTATE90:
        # All documented float/MX formats represent -1, 0, and 1 exactly.
        return ((index % 3) - 1).to(dtype)

    # A sparse 0/1 pattern gives a bounded, exactly representable cumulative
    # product while still detecting incorrect face/register traversal.
    return ((index % 37) != 0).to(dtype)


def _int_source(operation: SpecialSfpuOp) -> tuple[torch.Tensor, int]:
    index = torch.arange(32 * 32, dtype=torch.int32)
    if operation == SpecialSfpuOp.ADD_OFFSET:
        lhs = index % 7
        rhs = (index * 3 + 1) % 5
        return torch.cat((lhs, rhs)), 2
    if operation in (SpecialSfpuOp.SUM_COL, SpecialSfpuOp.SUM_ROW):
        return index % 5, 1
    return (index * 73 + 19) & 0x7FFF, 1


def _physical_golden(
    operation: SpecialSfpuOp,
    source: torch.Tensor,
    data_format: DataFormat,
    scalar: int,
) -> torch.Tensor:
    first_tile = source[: 32 * 32]

    if operation == SpecialSfpuOp.ROTATE90:
        # The two physical Dest registers addressed by the SFPI implementation
        # map back to adjacent real/imag elements in each logical row.
        logical = first_tile.reshape(32, 32)
        result = logical.clone()
        result[:, 0::2] = -logical[:, 1::2]
        result[:, 1::2] = logical[:, 0::2]
        return result.flatten()

    if operation == SpecialSfpuOp.SUM_COL:
        # VectorMode::R exposes complete logical rows to each SFPI register.
        # The two face calls reduce eight strided rows into rows 0/1 and 8/9.
        logical = first_tile.reshape(32, 32)
        result = logical.clone()
        offsets = (0, 2, 4, 6, 16, 18, 20, 22)
        for face_base in (0, 8):
            for lane_row in (0, 1):
                output_row = face_base + lane_row
                result[output_row] = sum(
                    logical[output_row + offset] for offset in offsets
                )
        return result.flatten()

    if operation == SpecialSfpuOp.SUM_ROW:
        # In VectorMode::C, registers 0/1 and 8/9 map to the even/odd
        # columns of the upper/lower eight rows in a 16x16 face pair.  The
        # reduction is therefore observable in the even columns of rows 0:8
        # and 16:24; the odd columns and lower eight rows remain untouched.
        logical = first_tile.reshape(32, 32)
        result = logical.clone()
        for face_row_base in (0, 16):
            upper = logical[face_row_base : face_row_base + 8]
            lower = logical[face_row_base + 8 : face_row_base + 16]
            result[face_row_base : face_row_base + 8, 0::2] = (
                upper[:, 0::2] + upper[:, 1::2] + lower[:, 0::2] + lower[:, 1::2]
            )
        return result.flatten()

    if operation == SpecialSfpuOp.TILED_PROD:
        # Quasar's dst_reg++ order walks an adjacent column pair through the
        # even or odd rows, while _inc_dst_addr_<16>() walks the two vertical
        # faces before moving to the right face column.  ITERATIONS=8 writes a
        # ninth register, overlapping the first register of the next vertical
        # face; that makes the product observable as one chain over all 32 rows
        # for each row parity/column pair.  The right face column is a separate
        # chain because the RC face walk resets the accumulator before it.
        logical = first_tile.reshape(32, 32)
        result = logical.clone()
        for face_col_base in (0, 16):
            for row_parity in (0, 1):
                for face_col in range(0, 16, 2):
                    chain = logical[
                        row_parity::2,
                        face_col_base + face_col : face_col_base + face_col + 2,
                    ].flatten()
                    result[
                        row_parity::2,
                        face_col_base + face_col : face_col_base + face_col + 2,
                    ] = torch.cumprod(chain, dim=0).reshape(16, 2)
        return result.flatten()

    registers = _tilized(first_tile, data_format).reshape(32, 32)
    if operation in _BITWISE_OPS:
        if data_format == DataFormat.Int32:
            signed_scalar = scalar if scalar < (1 << 31) else scalar - (1 << 32)
        else:
            signed_scalar = scalar
        scalar_tensor = torch.tensor(signed_scalar, dtype=registers.dtype)
        if operation == SpecialSfpuOp.BIT_AND:
            registers = torch.bitwise_and(registers, scalar_tensor)
        elif operation == SpecialSfpuOp.BIT_OR:
            registers = torch.bitwise_or(registers, scalar_tensor)
        else:
            registers = torch.bitwise_xor(registers, scalar_tensor)
    elif operation == SpecialSfpuOp.ADD_OFFSET:
        rhs_registers = _tilized(source[32 * 32 :], data_format).reshape(32, 32)
        registers = registers + rhs_registers
    else:
        raise AssertionError(f"Unhandled special SFPU operation: {operation}")

    return _untilized(registers, data_format)


def _cases_for(operation: SpecialSfpuOp):
    return [case for case in _generate_cases() if case[0] == operation]


def _run_special_case(special_case):
    (
        operation,
        formats,
        dest_acc,
        dest_sync,
        implied_math,
        scalar,
    ) = special_case[0]

    if operation in _FLOAT_OPS:
        source = _float_source(operation, formats.input_format)
        tile_count = 1
    else:
        source, tile_count = _int_source(operation)

    golden = _physical_golden(operation, source, formats.output_format, scalar).to(
        format_dict[formats.output_format]
    )

    tile_shape = construct_tile_shape(_FULL_TILE)
    # Tf32 is a 32-bit L1 container but is not itself a legal Quasar Dest
    # register format. Route it through SrcA and A2D so the value lands in an
    # FP32 Dest; native Float32 and Int32 retain direct UNPACK-to-Dest.
    unpack_to_dest = (
        formats.input_format.is_32_bit()
        and formats.input_format != DataFormat.Tf32
        and dest_acc == DestAccumulation.Yes
    )
    special_data_format = (
        formats.input_format
        if operation in _BITWISE_OPS or operation in _INT_SUM_OPS
        else DataFormat.Float16_b
    )

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            SPECIAL_SFPU_OP(operation),
            SPECIAL_DATA_FORMAT(special_data_format),
            IMPLIED_MATH_FORMAT(implied_math),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            NUM_FACES(tile_shape.total_num_faces()),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            DEST_INDEX(0),
            SPECIAL_SFPU_SCALAR(scalar),
        ],
        variant_stimuli=StimuliConfig(
            source,
            formats.input_format,
            source[: 32 * 32],
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=_FULL_TILE,
            sfpu=True,
            twos_complement=operation in _BITWISE_OPS or operation in _INT_SUM_OPS,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    result = configuration.run().result
    result_tensor = torch.tensor(
        result, dtype=format_dict[formats.output_format]
    ).flatten()
    assert len(result_tensor) == len(golden)
    assert passed_test(golden, result_tensor, formats.output_format), (
        f"{operation.name} failed for {formats.input_format}, scalar={scalar:#x}, "
        f"dest_acc={dest_acc}, dest_sync={dest_sync}, implied_math={implied_math}; "
        "this kernel family is intentionally full-tile-only"
    )


@pytest.mark.quasar
@parametrize(special_case=_cases_for(SpecialSfpuOp.ROTATE90))
def test_sfpu_alt_complex_rotate90_quasar(special_case):
    """Rotate complex pairs using the exact ported full-tile SFPI header."""
    _run_special_case(special_case)


@pytest.mark.quasar
@parametrize(
    special_case=(
        _cases_for(SpecialSfpuOp.BIT_AND)
        + _cases_for(SpecialSfpuOp.BIT_OR)
        + _cases_for(SpecialSfpuOp.BIT_XOR)
    )
)
def test_sfpu_unary_bitwise_quasar(special_case):
    """Sweep AND/OR/XOR, Int32/UInt16, and scalar mask edge cases."""
    _run_special_case(special_case)


@pytest.mark.quasar
@parametrize(special_case=_cases_for(SpecialSfpuOp.SUM_COL))
def test_sfpu_int_sum_col_quasar(special_case):
    """Exercise fixed-register integer column reduction."""
    _run_special_case(special_case)


@pytest.mark.quasar
@parametrize(special_case=_cases_for(SpecialSfpuOp.SUM_ROW))
def test_sfpu_int_sum_row_quasar(special_case):
    """Exercise fixed-register integer row reduction."""
    _run_special_case(special_case)


@pytest.mark.quasar
@parametrize(special_case=_cases_for(SpecialSfpuOp.ADD_OFFSET))
def test_sfpu_int_sum_add_quasar(special_case):
    """Exercise the only legal add_int offset: one complete Dest tile."""
    _run_special_case(special_case)


@pytest.mark.quasar
@parametrize(special_case=_cases_for(SpecialSfpuOp.TILED_PROD))
def test_sfpu_tiled_prod_quasar(special_case):
    """Exercise bounded cumulative products in physical Dest register order."""
    _run_special_case(special_case)
