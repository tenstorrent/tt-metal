# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar functional coverage for the SFPI addcdiv, lerp, and snake_beta kernels."""

import struct
from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TernarySFPUGolden,
    get_golden_generator,
    quantize_mx_stimuli,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
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
from helpers.utils import passed_test

_CPP_SOURCE = "sources/quasar/sfpu_ternary_sfpi_quasar_test.cpp"
_OPERATIONS = (
    MathOperation.SfpuAddcdiv,
    MathOperation.SfpuLerp,
    MathOperation.SfpuSnakeBeta,
)
_L1_FORMATS = (
    # Complete Python-representable subset of the Tensix Formats page. The
    # current enum has no Fp8R/Fp8P, MxFp6R/MxFp6P, or MxFp4 2x encodings.
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
_BF16_FORMAT = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


_DEFAULT_SCALAR_BITS = _float_bits(2.0)


@dataclass
class QUASAR_TERNARY_SFPI_OP(TemplateParameter):
    """Select one exact ported header without pulling the other two into a build."""

    operation: MathOperation

    def convert_to_cpp(self) -> str:
        return f"#define QUASAR_SFPI_TERNARY_{self.operation.cpp_enum_value.upper()}"


@dataclass
class QUASAR_TERNARY_SCALAR_BITS(RuntimeParameter):
    """Raw fp32 scalar consumed by addcdiv; runtime-only for ELF reuse."""

    scalar_bits: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TERNARY_SCALAR_BITS = {self.scalar_bits}u;"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t TERNARY_SCALAR_BITS;", "I"


def _canonical_dest_acc(data_format: DataFormat) -> DestAccumulation:
    if data_format in (DataFormat.Float32, DataFormat.Tf32):
        return DestAccumulation.Yes
    return DestAccumulation.No


def _append_case(cases, seen, case):
    def key_value(value):
        if isinstance(value, InputOutputFormat):
            return value.input_format, value.output_format
        if type(value).__name__ == "_RuntimeMarker":
            return value.value
        return value

    key = tuple(key_value(value) for value in case)
    if key not in seen:
        seen.add(key)
        cases.append(case)


def _generate_orthogonal_cases():
    """Cover each independent axis without taking their full Cartesian product."""
    cases = []
    seen = set()

    for operation in _OPERATIONS:
        # L1 format axis. MX formats unpack to Float16_b; Tf32 uses a 32-bit Dest.
        for data_format in _L1_FORMATS:
            fmt = InputOutputFormat(data_format, data_format)
            _append_case(
                cases,
                seen,
                (
                    operation,
                    fmt,
                    _canonical_dest_acc(data_format),
                    DestSync.Half,
                    ImpliedMathFormat.Yes,
                    runtime(_DEFAULT_SCALAR_BITS),
                    runtime((32, 32)),
                    runtime("signed"),
                ),
            )

        # Dest width, dvalid synchronization, and implied-math axes.
        for dest_acc, dest_sync, implied_math in (
            (DestAccumulation.No, DestSync.Half, ImpliedMathFormat.No),
            (DestAccumulation.No, DestSync.Full, ImpliedMathFormat.Yes),
            (DestAccumulation.Yes, DestSync.Half, ImpliedMathFormat.Yes),
            (DestAccumulation.Yes, DestSync.Full, ImpliedMathFormat.No),
        ):
            _append_case(
                cases,
                seen,
                (
                    operation,
                    _BF16_FORMAT,
                    dest_acc,
                    dest_sync,
                    implied_math,
                    runtime(_DEFAULT_SCALAR_BITS),
                    runtime((32, 32)),
                    runtime("signed"),
                ),
            )

    # Runtime-value coverage: addcdiv's scalar multiplier and deterministic
    # edge/extrapolation profiles for the two no-scalar kernels.
    for scalar in (-2.0, 0.5, 2.0):
        _append_case(
            cases,
            seen,
            (
                MathOperation.SfpuAddcdiv,
                InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
                DestAccumulation.Yes,
                DestSync.Half,
                ImpliedMathFormat.Yes,
                runtime(_float_bits(scalar)),
                runtime((32, 32)),
                runtime("signed"),
            ),
        )
    for operation in (MathOperation.SfpuLerp, MathOperation.SfpuSnakeBeta):
        _append_case(
            cases,
            seen,
            (
                operation,
                _BF16_FORMAT,
                DestAccumulation.No,
                DestSync.Half,
                ImpliedMathFormat.Yes,
                runtime(_DEFAULT_SCALAR_BITS),
                runtime((32, 32)),
                runtime("edge"),
            ),
        )

    return cases


def _alternating_sign(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.clone().flatten()
    result[::2] = -result[::2]
    return result


def _prepare_operands(operation, data_format, tile_dimensions, profile):
    torch.manual_seed(42)
    src_a, tile_count, src_b, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=tile_dimensions,
        stimuli_format_B=data_format,
        input_dimensions_B=tile_dimensions,
        spec_A=StimuliSpec.uniform(low=-1.5, high=1.5),
        spec_B=StimuliSpec.uniform(low=-2.0, high=2.0),
        tile_dimensions=tile_dimensions,
    )

    torch.manual_seed(43)
    if operation == MathOperation.SfpuLerp:
        src_c, _, _, _ = generate_stimuli(
            stimuli_format_A=data_format,
            input_dimensions_A=tile_dimensions,
            stimuli_format_B=data_format,
            input_dimensions_B=tile_dimensions,
            spec_A=StimuliSpec.uniform(low=-1.0, high=2.0),
            spec_B=StimuliSpec.uniform(low=-1.0, high=2.0),
            tile_dimensions=tile_dimensions,
        )
        if profile == "edge":
            edge_values = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=src_c.dtype)
            src_c = edge_values.repeat((src_c.numel() + 4) // 5)[: src_c.numel()]
    else:
        # Keep divisors away from zero but exercise both signs in every tile.
        src_c, _, _, _ = generate_stimuli(
            stimuli_format_A=data_format,
            input_dimensions_A=tile_dimensions,
            stimuli_format_B=data_format,
            input_dimensions_B=tile_dimensions,
            spec_A=StimuliSpec.uniform(low=1.0, high=2.0),
            spec_B=StimuliSpec.uniform(low=1.0, high=2.0),
            tile_dimensions=tile_dimensions,
        )
        src_c = _alternating_sign(src_c)
        if profile == "edge" and operation == MathOperation.SfpuSnakeBeta:
            beta_values = torch.tensor([-2.0, -1.0, 0.5, 2.0], dtype=src_c.dtype)
            src_c = beta_values.repeat((src_c.numel() + 3) // 4)[: src_c.numel()]

    if operation == MathOperation.SfpuAddcdiv and data_format == DataFormat.MxFp8R:
        # Avoid an output exactly on the MxFp8R shared-scale rounding tie. The
        # packer and the software quantizer legitimately select opposite
        # adjacent values there, obscuring the arithmetic being tested. These
        # signed, exactly representable patterns still exercise add/divide and
        # both divisor signs, with exact results {0, -1, -3, 1}.
        src_a_values = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=src_a.dtype)
        src_b_values = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=src_b.dtype)
        src_c_values = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=src_c.dtype)
        repetitions = (src_a.numel() + 3) // 4
        src_a = src_a_values.repeat(repetitions)[: src_a.numel()]
        src_b = src_b_values.repeat(repetitions)[: src_b.numel()]
        src_c = src_c_values.repeat(repetitions)[: src_c.numel()]

    return src_a.flatten(), src_b.flatten(), src_c.flatten(), tile_count


def _quantize_l1_values(tensor, data_format, num_faces):
    if data_format.is_mx_format():
        return quantize_mx_stimuli(tensor.flatten(), data_format, num_faces)
    return tensor.flatten()


@pytest.mark.quasar
@parametrize(ternary_case=_generate_orthogonal_cases())
def test_sfpu_ternary_sfpi_quasar(ternary_case):
    """Exercise the three newly ported Quasar ternary SFPI implementations."""
    (
        operation,
        formats,
        dest_acc,
        dest_sync,
        implied_math_format,
        scalar_bits,
        tile_dimensions,
        profile,
    ) = ternary_case[0]

    tile_shape = construct_tile_shape(tile_dimensions)
    num_faces = tile_shape.total_num_faces()
    src_a, src_b, src_c, tile_count = _prepare_operands(
        operation, formats.input_format, tile_dimensions, profile
    )
    assert (
        tile_count == 1
    ), "Orthogonal ternary cases stage exactly one tile per operand"

    golden_a = _quantize_l1_values(src_a, formats.input_format, num_faces)
    golden_b = _quantize_l1_values(src_b, formats.input_format, num_faces)
    golden_c = _quantize_l1_values(src_c, formats.input_format, num_faces)
    generate_golden = get_golden_generator(TernarySFPUGolden)
    golden_tensor = generate_golden(
        operation,
        golden_a,
        golden_b,
        golden_c,
        scalar_bits,
        formats.output_format,
    )
    if formats.output_format.is_mx_format():
        golden_tensor = quantize_mx_stimuli(
            golden_tensor.flatten(), formats.output_format, num_faces
        )

    buffer_a = torch.cat([src_a, src_b, src_c])
    # Tf32 does not have a legal direct UNPACK-to-Dest conversion on Quasar;
    # it and every narrow/MX L1 format use A2D before SFPU.
    unpack_to_dest = formats.input_format == DataFormat.Float32
    torch_format_out = format_dict[formats.output_format]

    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            QUASAR_TERNARY_SFPI_OP(operation),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(3),
            NUM_FACES(num_faces),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            DEST_INDEX(0),
            QUASAR_TERNARY_SCALAR_BITS(scalar_bits),
        ],
        variant_stimuli=StimuliConfig(
            buffer_a,
            formats.input_format,
            src_b,
            formats.input_format,
            formats.output_format,
            tile_count_A=3,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=tile_dimensions != (32, 32),
            sfpu=True,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    result = configuration.run().result
    assert len(result) == len(golden_tensor)
    result_tensor = torch.tensor(result, dtype=torch_format_out).flatten()
    golden_tensor = golden_tensor.to(torch_format_out).flatten()
    assert passed_test(golden_tensor, result_tensor, formats.output_format), (
        f"{operation.name} failed for {formats.input_format}, dest_acc={dest_acc}, "
        f"dest_sync={dest_sync}, implied_math={implied_math_format}, "
        f"tile={tile_dimensions}, profile={profile}"
    )
