# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Unified Quasar SFPU test file.  Replaces the individual per-op test files:
#   test_sfpu_abs_quasar.py, test_sfpu_nonlinear_quasar.py,
#   test_sfpu_rsqrt_quasar.py, test_sfpu_square_quasar.py,
#   test_sfpu_fill_quasar.py, test_sfpu_swiglu_quasar.py,
#   test_sfpu_binary_quasar.py, test_sfpu_binary_float_quasar.py,
#   test_sfpu_where_quasar.py
#
# Also provides the first Python-level test for sfpu_binary_max_min.
#
# Source file routing:
#   unpack_to_dest=True  → sources/quasar/sfpu_quasar_dest_test.cpp
#   unpack_to_dest=False → sources/quasar/sfpu_quasar_srca_test.cpp

import math
from enum import Enum
from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig, InputOutputFormat
from helpers.golden_generators import (
    BinarySFPUGolden,
    UnarySFPUGolden,
    WhereGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    VectorMode,
    format_dict,
)
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    FILL_INT_FORMAT,
    IMPLIED_MATH_FORMAT,
    IS_MAX_OP,
    MATH_OP,
    NUM_FACES,
    SFPU_INT_OP,
    SFPU_IS_BINARY_INT_OP,
    SFPU_IS_BINARY_MAX_MIN_OP,
    SFPU_IS_FILL_OP,
    SFPU_IS_WHERE_OP,
    SFPU_TILE_INDICES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    VECTOR_MODE,
)
from helpers.utils import passed_test

# ---------------------------------------------------------------------------
# Source file router
# ---------------------------------------------------------------------------


def _sfpu_source(unpack_to_dest: bool) -> str:
    if unpack_to_dest:
        return "sources/quasar/sfpu_quasar_dest_test.cpp"
    return "sources/quasar/sfpu_quasar_srca_test.cpp"


def _unpacker_engine(unpack_to_dest: bool) -> UnpackerEngine:
    return UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA


# ---------------------------------------------------------------------------
# Shared format / combination helpers
# ---------------------------------------------------------------------------

_STANDARD_FORMATS = input_output_formats(
    [DataFormat.Float16, DataFormat.Float32, DataFormat.Float16_b]
)


def _generate_unary_combinations(formats_list: List[FormatConfig]):
    combinations = []
    dest_sync_modes = (DestSync.Half, DestSync.Full)
    for fmt in formats_list:
        in_fmt = fmt.input_format
        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )
        for dest_acc in dest_acc_modes:
            if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                continue
            for dest_sync in dest_sync_modes:
                for implied_math_format in [
                    ImpliedMathFormat.No,
                    ImpliedMathFormat.Yes,
                ]:
                    if (
                        in_fmt.is_mx_format()
                        and implied_math_format == ImpliedMathFormat.No
                    ):
                        continue
                    for input_dimensions in [[32, 32], [64, 64]]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                dest_sync,
                                implied_math_format,
                                input_dimensions,
                            )
                        )
    return combinations


# generate_sfpu_square_combinations / SFPU_SQUARE_FORMATS exported for
# test_sfpu_square_trisc3_quasar.py which imports them.
def generate_sfpu_square_combinations(formats_list: List[FormatConfig]):
    combinations = []
    dest_sync_modes = (DestSync.Half, DestSync.Full)
    for fmt in formats_list:
        in_fmt = fmt.input_format
        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )
        for dest_acc in dest_acc_modes:
            if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                continue
            for dest_sync in dest_sync_modes:
                for implied_math_format in [
                    ImpliedMathFormat.No,
                    ImpliedMathFormat.Yes,
                ]:
                    for input_dimensions in [[32, 32], [64, 64], [32, 64]]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                dest_sync,
                                implied_math_format,
                                input_dimensions,
                            )
                        )
    return combinations


SFPU_SQUARE_FORMATS = _STANDARD_FORMATS


# ---------------------------------------------------------------------------
# Stimuli preparation helpers
# ---------------------------------------------------------------------------


def prepare_abs_inputs(src_A, src_B, input_format, output_format):
    input_torch_format = format_dict[input_format]
    output_torch_format = format_dict[output_format]
    input_finfo = torch.finfo(input_torch_format)
    output_finfo = torch.finfo(output_torch_format)
    max_safe_value = min(input_finfo.max, output_finfo.max) * 0.9
    if input_torch_format == torch.bfloat16:
        max_safe_value = min(max_safe_value, 1e4)
    else:
        max_safe_value = min(max_safe_value, input_finfo.max * 0.9)
    min_magnitude = max(1e-6, input_finfo.tiny * 100)
    src_A_float = src_A.to(torch.float32)
    src_B_float = src_B.to(torch.float32)
    src_A_min, src_A_max = src_A_float.min(), src_A_float.max()
    src_A_normalized = (
        (src_A_float - src_A_min) / (src_A_max - src_A_min)
        if src_A_max > src_A_min
        else torch.zeros_like(src_A_float)
    )
    log_min = torch.log(torch.tensor(min_magnitude, dtype=torch.float32))
    log_max = torch.log(torch.tensor(max_safe_value, dtype=torch.float32))
    magnitudes = torch.exp(log_min + src_A_normalized * (log_max - log_min))
    src_B_min, src_B_max = src_B_float.min(), src_B_float.max()
    src_B_normalized = (
        (src_B_float - src_B_min) / (src_B_max - src_B_min)
        if src_B_max > src_B_min
        else torch.zeros_like(src_B_float)
    )
    signs = torch.where(src_B_normalized < 0.5, -1.0, 1.0)
    src_A_values = torch.clamp(signs * magnitudes, -max_safe_value, max_safe_value)
    return src_A_values.to(input_torch_format)


def prepare_square_inputs(src_A, src_B, input_format, output_format):
    input_torch_format = format_dict[input_format]
    output_torch_format = format_dict[output_format]
    input_finfo = torch.finfo(input_torch_format)
    output_finfo = torch.finfo(output_torch_format)
    max_safe_value = math.sqrt(output_finfo.max) * 0.9
    if input_torch_format == torch.bfloat16:
        max_safe_value = min(max_safe_value, 1e4)
    else:
        max_safe_value = min(max_safe_value, math.sqrt(input_finfo.max) * 0.9)
    min_magnitude = max(1e-6, input_finfo.tiny * 100)
    src_A_float = src_A.to(torch.float32)
    src_B_float = src_B.to(torch.float32)
    src_A_min, src_A_max = src_A_float.min(), src_A_float.max()
    src_A_normalized = (
        (src_A_float - src_A_min) / (src_A_max - src_A_min)
        if src_A_max > src_A_min
        else torch.zeros_like(src_A_float)
    )
    log_min = torch.log(torch.tensor(min_magnitude, dtype=torch.float32))
    log_max = torch.log(torch.tensor(max_safe_value, dtype=torch.float32))
    magnitudes = torch.exp(log_min + src_A_normalized * (log_max - log_min))
    src_B_min, src_B_max = src_B_float.min(), src_B_float.max()
    src_B_normalized = (
        (src_B_float - src_B_min) / (src_B_max - src_B_min)
        if src_B_max > src_B_min
        else torch.zeros_like(src_B_float)
    )
    signs = torch.where(src_B_normalized < 0.5, -1.0, 1.0)
    src_A_values = torch.clamp(signs * magnitudes, -max_safe_value, max_safe_value)
    return src_A_values.to(input_torch_format)


def prepare_inputs_for_operation(src_A, mathop, input_format, output_format=None):
    torch_format = format_dict[input_format]
    if mathop == MathOperation.Exp:
        src_A = (-10.0 + src_A.to(torch.float32) * 20.0).to(torch_format)
    elif mathop == MathOperation.Gelu:
        src_A = torch.empty_like(src_A, dtype=torch.float32).uniform_(-10.0, 10.0)
    elif mathop == MathOperation.Relu:
        finfo = torch.finfo(torch_format)
        min_val, max_val = finfo.min / 2, finfo.max / 2
        src_A = (min_val + src_A.to(torch.float32) * (max_val - min_val)).to(
            torch_format
        )
    elif mathop in (MathOperation.Sqrt, MathOperation.Rsqrt):
        finfo = torch.finfo(torch_format)
        min_val = max(1e-6, finfo.tiny * 100)
        if output_format and output_format in (
            DataFormat.Float16,
            DataFormat.Float16_b,
        ):
            out_finfo = torch.finfo(format_dict[output_format])
            max_val = out_finfo.max * 0.8
        elif torch_format in (torch.float16, torch.bfloat16):
            max_val = min(finfo.max, 1e4)
        else:
            max_val = finfo.max
        log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
        log_max = torch.log(torch.tensor(float(max_val), dtype=torch.float32))
        src_A = torch.exp(log_min + src_A.to(torch.float32) * (log_max - log_min)).to(
            torch_format
        )
    elif mathop == MathOperation.Reciprocal:
        finfo = torch.finfo(torch_format)
        min_val = max(1e-6, finfo.tiny * 100)
        max_val = finfo.max / 2
        log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
        log_max = torch.log(torch.tensor(float(max_val), dtype=torch.float32))
        vals = torch.exp(log_min + src_A.to(torch.float32) * (log_max - log_min))
        src_A = torch.where(vals.abs() < min_val, torch.sign(vals) * min_val, vals).to(
            torch_format
        )
    elif mathop in (MathOperation.Tanh, MathOperation.Sigmoid, MathOperation.Silu):
        src_A = (-10.0 + src_A.to(torch.float32) * 20.0).to(torch_format)
    return src_A


# ---------------------------------------------------------------------------
# Standard unary test (abs, exp, gelu, relu, reciprocal, sqrt, tanh,
#                      sigmoid, silu, rsqrt, square)
# ---------------------------------------------------------------------------

_UNARY_OPS = [
    MathOperation.Abs,
    MathOperation.Exp,
    MathOperation.Gelu,
    MathOperation.Relu,
    MathOperation.Reciprocal,
    MathOperation.Sqrt,
    MathOperation.Tanh,
    MathOperation.Sigmoid,
    MathOperation.Silu,
    MathOperation.Rsqrt,
    MathOperation.Square,
]

_UNARY_COMBINATIONS = [
    (fmt, dest_acc, dest_sync, implied_math, dims, mathop)
    for (fmt, dest_acc, dest_sync, implied_math, dims) in _generate_unary_combinations(
        _STANDARD_FORMATS
    )
    for mathop in _UNARY_OPS
]


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_implied_math_dims_mathop=_UNARY_COMBINATIONS,
)
def test_sfpu_unary_quasar(formats_dest_acc_sync_implied_math_dims_mathop):
    """Test standard unary SFPU ops on Quasar (abs, exp, gelu, relu, reciprocal,
    sqrt, tanh, sigmoid, silu, rsqrt, square)."""
    (formats, dest_acc, dest_sync, implied_math_format, input_dimensions, mathop) = (
        formats_dest_acc_sync_implied_math_dims_mathop[0]
    )

    torch.manual_seed(42)
    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    if mathop == MathOperation.Abs:
        src_A = prepare_abs_inputs(
            src_A, src_B, formats.input_format, formats.output_format
        )
    elif mathop == MathOperation.Square:
        src_A = prepare_square_inputs(
            src_A, src_B, formats.input_format, formats.output_format
        )
    else:
        src_A = prepare_inputs_for_operation(
            src_A, mathop, formats.input_format, formats.output_format
        )

    num_faces = 4
    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    configuration = TestConfig(
        _sfpu_source(unpack_to_dest),
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(_unpacker_engine(unpack_to_dest)),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# ---------------------------------------------------------------------------
# Fill test
# ---------------------------------------------------------------------------

_FILL_FLOAT_INPUTS = [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32]
_FILL_FLOAT_OUTPUTS = _FILL_FLOAT_INPUTS + [DataFormat.MxFp8R, DataFormat.MxFp8P]
_FILL_FLOAT_FORMATS = [
    InputOutputFormat(in_fmt, out_fmt)
    for in_fmt in _FILL_FLOAT_INPUTS
    for out_fmt in _FILL_FLOAT_OUTPUTS
]
_FILL_INT_FORMATS = input_output_formats(
    [DataFormat.Int32, DataFormat.Int16, DataFormat.Int8, DataFormat.UInt8], same=True
)


def _generate_fill_combinations(float_formats, int_formats):
    combinations = []
    for fmt in float_formats:
        in_fmt = fmt.input_format
        dest_acc = DestAccumulation.Yes if in_fmt.is_32_bit() else DestAccumulation.No
        if (
            in_fmt != DataFormat.Float32
            and fmt.output_format == DataFormat.Float32
            and dest_acc == DestAccumulation.No
        ):
            continue
        for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
            for input_dimensions in [[32, 32], [64, 64]]:
                combinations.append(
                    (fmt, dest_acc, implied_math_format, input_dimensions)
                )
    for fmt in int_formats:
        in_fmt = fmt.input_format
        dest_acc = DestAccumulation.Yes if in_fmt.is_32_bit() else DestAccumulation.No
        for input_dimensions in [[32, 32], [64, 64]]:
            combinations.append((fmt, dest_acc, ImpliedMathFormat.No, input_dimensions))
    return combinations


# Exported alias used by test_sfpu_fill_quasar.py (if still running)
generate_sfpu_fill_combinations = _generate_fill_combinations


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_input_dims=_generate_fill_combinations(
        _FILL_FLOAT_FORMATS, _FILL_INT_FORMATS
    ),
)
def test_sfpu_fill_quasar(formats_dest_acc_implied_math_input_dims):
    """Test fill SFPU operation on Quasar. Always uses unpack-to-dest path."""
    (formats, dest_acc, implied_math_format, input_dimensions) = (
        formats_dest_acc_implied_math_input_dims[0]
    )
    is_int_fill = formats.input_format.is_integer()
    torch.manual_seed(42)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    num_faces = 4

    if is_int_fill:
        FILL_INT_VALUE = 5
        golden_tensor = torch.full(
            (src_A.numel(),), FILL_INT_VALUE, dtype=format_dict[formats.output_format]
        )
    else:
        FILL_CONST_VALUE = 5.0
        generate_golden = get_golden_generator(UnarySFPUGolden)
        golden_tensor = generate_golden(
            MathOperation.Fill,
            src_A,
            formats.output_format,
            dest_acc,
            formats.input_format,
            input_dimensions,
            fill_const_value=FILL_CONST_VALUE,
        )

    configuration = TestConfig(
        _sfpu_source(True),  # fill always uses unpack-to-dest path
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Fill),
            SFPU_IS_FILL_OP(),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
            FILL_INT_FORMAT(formats.input_format if is_int_fill else DataFormat.Int32),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# ---------------------------------------------------------------------------
# Swiglu test
# ---------------------------------------------------------------------------

_SWIGLU_ALPHA = 1.702
_SWIGLU_CLAMP_LIMIT = 7.0

_SWIGLU_CORNER_CASES = (
    (+7.0, +7.0),
    (-7.0, -7.0),
    (+7.0, -7.0),
    (-7.0, +7.0),
    (+6.99, +6.99),
    (+7.01, +7.01),
    (-6.99, -6.99),
    (-7.01, -7.01),
    (+6.999, +7.001),
    (-7.001, -6.999),
    (+8.0, +8.0),
    (-8.0, -8.0),
    (+50.0, -50.0),
    (-50.0, +50.0),
    (+1000.0, +1000.0),
    (-1000.0, -1000.0),
    (0.0, 0.0),
    (0.0, -1.0),
    (+5.0, -1.0),
    (-5.0, -1.0),
    (+7.0, -1.0),
    (+0.001, +0.001),
    (-0.001, -0.001),
    (+0.0001, -0.0001),
    (+5.0, -3.0),
    (-5.0, +3.0),
    (+3.5, -5.5),
    (+50.0, +0.5),
    (+0.5, +50.0),
    (-50.0, +0.5),
    (+0.5, -50.0),
    (+1.0, +0.5),
    (+2.0, -0.5),
)


class _StimulusDistribution(Enum):
    LOG_UNIFORM = "log_uniform"
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    LARGE_ONLY = "large_only"
    TINY_ONLY = "tiny_only"


def _swiglu_golden(gate, up, output_format):
    torch_out_dtype = format_dict[output_format]
    gate_c = torch.minimum(gate.to(torch.float32), torch.tensor(_SWIGLU_CLAMP_LIMIT))
    up_c = torch.clamp(up.to(torch.float32), -_SWIGLU_CLAMP_LIMIT, _SWIGLU_CLAMP_LIMIT)
    sig = torch.sigmoid(_SWIGLU_ALPHA * gate_c)
    return ((up_c + 1.0) * gate_c * sig).to(torch_out_dtype)


def _normalize_to_unit(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min) if x_max > x_min else torch.zeros_like(x)


def _prepare_swiglu_inputs(
    src_A, src_B, input_format, distribution=_StimulusDistribution.LOG_UNIFORM
):
    torch_dtype = format_dict[input_format]
    src_A_u = _normalize_to_unit(src_A.to(torch.float32))
    signs = torch.where(_normalize_to_unit(src_B.to(torch.float32)) < 0.5, -1.0, 1.0)
    if distribution == _StimulusDistribution.LOG_UNIFORM:
        log_low, log_high = torch.log(torch.tensor(0.1)), torch.log(torch.tensor(9.0))
        gate_up = signs * torch.exp(log_low + src_A_u * (log_high - log_low))
    elif distribution == _StimulusDistribution.UNIFORM:
        gate_up = (src_A_u * 2.0 - 1.0) * 9.0
    elif distribution == _StimulusDistribution.GAUSSIAN:
        eps = torch.tensor(1e-7)
        u1 = torch.clamp(src_A_u, min=eps, max=1.0 - eps.item())
        u2 = torch.clamp(
            _normalize_to_unit(src_B.to(torch.float32)), min=eps, max=1.0 - eps.item()
        )
        gate_up = (
            torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * torch.pi * u2) * 3.0
        )
    elif distribution == _StimulusDistribution.LARGE_ONLY:
        gate_up = signs * (7.5 + src_A_u * 42.5)
    else:  # TINY_ONLY
        gate_up = signs * (src_A_u * 0.5)
    gate_up = gate_up.to(torch.float32)
    n = gate_up.numel() // 2
    gate, up = gate_up[:n].clone(), gate_up[n:].clone()
    num_corner = min(len(_SWIGLU_CORNER_CASES), gate.numel(), up.numel())
    for i in range(num_corner):
        gate[i], up[i] = _SWIGLU_CORNER_CASES[i]
    return gate.to(torch_dtype), up.to(torch_dtype)


_SWIGLU_FORMATS = input_output_formats(
    [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
)


def _generate_swiglu_combinations(formats_list):
    combinations = []
    for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats_list):
        for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
            if (
                fmt.input_format.is_mx_format()
                and implied_math_format == ImpliedMathFormat.No
            ):
                continue
            combinations.append((fmt, dest_acc, implied_math_format))
    return combinations


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math=_generate_swiglu_combinations(_SWIGLU_FORMATS),
    distribution=list(_StimulusDistribution),
)
def test_sfpu_swiglu_quasar(formats_dest_acc_implied_math, distribution):
    """Test swiglu SFPU kernel on Quasar."""
    (formats, dest_acc, implied_math_format) = formats_dest_acc_implied_math
    torch.manual_seed(42)
    num_faces = 4
    input_dimensions = [64, 32]  # 2 tiles: gate + up

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    gate, up = _prepare_swiglu_inputs(src_A, src_B, formats.input_format, distribution)
    combined_input = torch.cat([gate, up]).to(format_dict[formats.input_format])
    golden_tensor = _swiglu_golden(gate, up, formats.output_format).flatten()

    unpack_to_dest = formats.input_format.is_32_bit() == (
        dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        _sfpu_source(unpack_to_dest),
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuSwiGLU),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(_unpacker_engine(unpack_to_dest)),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            combined_input,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# ---------------------------------------------------------------------------
# Binary integer tests (add, mul, compare)
# ---------------------------------------------------------------------------

_SRC0_IDX, _SRC1_IDX, _DST_IDX = 0, 1, 0

_COMP_OPS = [
    ("GT", MathOperation.SfpuGtInt),
    ("LT", MathOperation.SfpuLtInt),
    ("LE", MathOperation.SfpuLeInt),
    ("GE", MathOperation.SfpuGeInt),
]


def _run_sfpu_binary_int(
    data_format,
    dest_acc,
    src0_idx,
    src1_idx,
    dst_idx,
    mathop,
    sfpu_int_op="",
    clamp_inputs=None,
    unpack_to_dest=True,
):
    num_tiles_needed = max(src0_idx, src1_idx, dst_idx) + 1
    formats = InputOutputFormat(input_format=data_format, output_format=data_format)
    input_dimensions = [num_tiles_needed * 32, 32]

    if data_format.is_integer():
        iinfo = torch.iinfo(format_dict[data_format])
        spec = StimuliSpec.uniform(low=float(iinfo.min), high=float(iinfo.max - 1))
    else:
        spec = StimuliSpec.uniform(low=-1.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=data_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    if clamp_inputs is not None:
        src_A = torch.clamp(src_A, -clamp_inputs, clamp_inputs)
        src_B = torch.clamp(src_B, -clamp_inputs, clamp_inputs)

    num_faces = 4
    elements_per_tile = 1024
    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_full = generate_golden(
        mathop,
        src_A,
        src0_idx,
        src1_idx,
        dst_idx,
        32,
        input_dimensions,
        data_format,
    ).flatten()
    dst_start = dst_idx * elements_per_tile
    golden_tensor = golden_full[dst_start : dst_start + elements_per_tile]

    templates = [
        MATH_OP(mathop=mathop),
        SFPU_IS_BINARY_INT_OP(),
        IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
        UNPACKER_ENGINE_SEL(_unpacker_engine(unpack_to_dest)),
        DEST_SYNC(),
    ]
    if sfpu_int_op:
        templates.insert(2, SFPU_INT_OP(sfpu_int_op))

    configuration = TestConfig(
        _sfpu_source(unpack_to_dest),
        formats,
        templates=templates,
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            data_format,
            src_B,
            data_format,
            data_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=1,
            num_faces=num_faces,
            twos_complement=data_format.is_integer(),
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[data_format])
    assert passed_test(
        golden_tensor, res_tensor, data_format
    ), "Assert against golden failed"


@pytest.mark.quasar
@pytest.mark.parametrize(
    "data_format,dest_acc",
    [
        (DataFormat.Int32, DestAccumulation.Yes),
    ],
)
def test_sfpu_binary_add_quasar(data_format, dest_acc):
    """Binary SFPU ADD (Int32) — unpack-to-dest path."""
    _run_sfpu_binary_int(
        data_format,
        dest_acc,
        _SRC0_IDX,
        _SRC1_IDX,
        _DST_IDX,
        mathop=MathOperation.SfpuElwadd,
        unpack_to_dest=True,
    )


@pytest.mark.quasar
@pytest.mark.parametrize(
    "data_format,dest_acc",
    [
        (DataFormat.Int32, DestAccumulation.Yes),
    ],
)
def test_sfpu_binary_mul_int_quasar(data_format, dest_acc):
    """Binary SFPU MUL_INT (Int32) — SrcA/FPU path (tests the FPU datacopy path for int ops)."""
    _run_sfpu_binary_int(
        data_format,
        dest_acc,
        _SRC0_IDX,
        _SRC1_IDX,
        _DST_IDX,
        mathop=MathOperation.SfpuElwmulInt,
        sfpu_int_op="MUL",
        clamp_inputs=1000,
        unpack_to_dest=False,
    )


@pytest.mark.quasar
@pytest.mark.parametrize("comp_op,mathop", _COMP_OPS, ids=[op for op, _ in _COMP_OPS])
@pytest.mark.parametrize(
    "data_format,dest_acc",
    [
        (DataFormat.Int32, DestAccumulation.Yes),
    ],
)
def test_sfpu_binary_comp_int_quasar(comp_op, mathop, data_format, dest_acc):
    """Binary SFPU integer comparison (GT/LT/LE/GE) — unpack-to-dest path."""
    _run_sfpu_binary_int(
        data_format,
        dest_acc,
        _SRC0_IDX,
        _SRC1_IDX,
        _DST_IDX,
        mathop=mathop,
        sfpu_int_op=comp_op,
        unpack_to_dest=True,
    )


# ---------------------------------------------------------------------------
# Binary float test (div, mul)
# ---------------------------------------------------------------------------

_BINARY_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
_TILE_INDEX_VARIANTS = [(0, 1, 0), (0, 1, 1), (2, 3, 0)]
_ELEMENTS_PER_TILE = 1024
_SPECIAL_CASE_LANES = [
    (0, 0.0, 0.0, "nan"),
    (1, 1.5, 0.0, "pos_inf"),
    (2, -1.5, 0.0, "neg_inf"),
    (3, 2.7, 2.7, "one"),
    (4, -3.3, -3.3, "one"),
]


def _get_binary_float_valid_formats_dest_acc():
    fmts = input_output_formats(_BINARY_FLOAT_FORMATS)
    return [
        (fmt, dest_acc)
        for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(fmts)
        if not (
            fmt.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes
        )
    ]


def _get_binary_float_implied_math(fmt):
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _prepare_binary_float_inputs(src_A, data_format, src0_idx, src1_idx, mathop):
    torch_format = format_dict[data_format]
    if mathop == MathOperation.SfpuElwdiv:
        scaled = (src_A.to(torch.float32) - 0.5) * 8.0
        sign = torch.where(scaled >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        scaled = sign * torch.maximum(scaled.abs(), torch.tensor(0.25))
        scaled = scaled.to(torch_format)
        flat = scaled.flatten()
        for lane, dividend, divisor, _ in _SPECIAL_CASE_LANES:
            flat[src0_idx * _ELEMENTS_PER_TILE + lane] = dividend
            flat[src1_idx * _ELEMENTS_PER_TILE + lane] = divisor
        return flat.reshape(scaled.shape)
    elif mathop == MathOperation.SfpuElwmul:
        return ((src_A.to(torch.float32) - 0.5) * 500.0).to(torch_format)
    return src_A


def _run_sfpu_binary_float(formats_dest_acc, implied_math_format, tile_indices, mathop):
    formats, dest_acc = formats_dest_acc
    src0_idx, src1_idx, dst_idx = tile_indices
    num_tiles_needed = max(src0_idx, src1_idx, dst_idx) + 1
    input_dimensions = [num_tiles_needed * 32, 32]
    torch.manual_seed(42)

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    src_A = _prepare_binary_float_inputs(
        src_A, formats.input_format, src0_idx, src1_idx, mathop
    )
    num_faces = 4

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_full = generate_golden(
        mathop,
        src_A,
        src0_idx,
        src1_idx,
        dst_idx,
        32,
        input_dimensions,
        formats.input_format,
    ).flatten()
    golden_tensor = golden_full[
        dst_idx * _ELEMENTS_PER_TILE : (dst_idx + 1) * _ELEMENTS_PER_TILE
    ]
    golden_tensor = golden_tensor.to(format_dict[formats.output_format])

    configuration = TestConfig(
        _sfpu_source(True),  # binary float always unpack-to-dest
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"

    if mathop == MathOperation.SfpuElwdiv:
        for lane, _, _, kind in _SPECIAL_CASE_LANES:
            if kind == "one":
                assert (
                    res_tensor[lane].item() == 1.0
                ), f"x/x at lane {lane}: expected 1.0"


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_binary_float_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_binary_float_implied_math(
        formats_dest_acc[0]
    ),
    tile_indices=_TILE_INDEX_VARIANTS,
)
def test_sfpu_binary_div_quasar(formats_dest_acc, implied_math_format, tile_indices):
    _run_sfpu_binary_float(
        formats_dest_acc, implied_math_format, tile_indices, MathOperation.SfpuElwdiv
    )


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_binary_float_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_binary_float_implied_math(
        formats_dest_acc[0]
    ),
    tile_indices=_TILE_INDEX_VARIANTS,
)
def test_sfpu_binary_float_mul_quasar(
    formats_dest_acc, implied_math_format, tile_indices
):
    _run_sfpu_binary_float(
        formats_dest_acc, implied_math_format, tile_indices, MathOperation.SfpuElwmul
    )


# ---------------------------------------------------------------------------
# Binary max / min test
# ---------------------------------------------------------------------------

_MAX_MIN_FORMATS = input_output_formats(
    [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32, DataFormat.Int32],
    same=True,
)


def _generate_binary_max_min_combinations(formats_list):
    combinations = []
    for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats_list):
        if not (
            fmt.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes
        ):
            for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
                if (
                    fmt.input_format.is_mx_format()
                    and implied_math_format == ImpliedMathFormat.No
                ):
                    continue
                combinations.append((fmt, dest_acc, implied_math_format))
    return combinations


def _binary_max_min_golden(in0, in1, is_max, output_format):
    torch_out = format_dict[output_format]
    in0_f = in0.to(torch.float32)
    in1_f = in1.to(torch.float32)
    result = torch.maximum(in0_f, in1_f) if is_max else torch.minimum(in0_f, in1_f)
    return result.to(torch_out)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math=_generate_binary_max_min_combinations(
        _MAX_MIN_FORMATS
    ),
    is_max=[True, False],
)
def test_sfpu_binary_max_min_quasar(formats_dest_acc_implied_math, is_max):
    """Test binary max/min SFPU on Quasar. 2 input tiles, output at DST_INDEX+2."""
    (formats, dest_acc, implied_math_format) = formats_dest_acc_implied_math
    input_dimensions = [64, 32]  # 2 input tiles (32x32 each)
    torch.manual_seed(42)

    if formats.input_format.is_integer():
        iinfo = torch.iinfo(format_dict[formats.input_format])
        spec = StimuliSpec.uniform(low=float(iinfo.min), high=float(iinfo.max - 1))
    else:
        spec = StimuliSpec.uniform(low=-10.0, high=10.0)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    num_faces = 4
    elements_per_tile = num_faces * 16 * 16
    in0 = src_A[:elements_per_tile]
    in1 = src_A[elements_per_tile : 2 * elements_per_tile]
    golden_tensor = _binary_max_min_golden(in0, in1, is_max, formats.output_format)

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        _sfpu_source(unpack_to_dest),
        formats,
        templates=[
            IS_MAX_OP(is_max),
            SFPU_IS_BINARY_MAX_MIN_OP(),
            IMPLIED_MATH_FORMAT(implied_math_format),
            UNPACKER_ENGINE_SEL(_unpacker_engine(unpack_to_dest)),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=1,
            num_faces=num_faces,
            twos_complement=formats.input_format.is_integer(),
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# ---------------------------------------------------------------------------
# Where (ternary) test
# ---------------------------------------------------------------------------

_WHERE_FORMATS = input_output_formats(
    [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
)
_FACE_SIZE = 16 * 16
_PROCESSED_FACES = {
    VectorMode.None_: (0,),
    VectorMode.R: (0, 1),
    VectorMode.C: (0, 2),
    VectorMode.RC: (0, 1, 2, 3),
}


def _processed_face_mask(vector_mode, num_faces):
    mask = torch.zeros(num_faces * _FACE_SIZE, dtype=torch.bool)
    for face in _PROCESSED_FACES[vector_mode]:
        mask[face * _FACE_SIZE : (face + 1) * _FACE_SIZE] = True
    return mask


def _get_where_valid_formats_dest_acc():
    return [
        (fmt, dest_acc)
        for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(_WHERE_FORMATS)
        if not (
            fmt.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes
        )
    ]


def _get_where_implied_math(fmt):
    if fmt.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def _build_condition(base, input_format, test_case):
    torch_format = format_dict[input_format]
    if test_case == "all_ones":
        return torch.ones_like(base, dtype=torch_format)
    if test_case == "all_zeros":
        return torch.zeros_like(base, dtype=torch_format)
    return base.to(torch_format)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=_get_where_valid_formats_dest_acc(),
    implied_math_format=lambda formats_dest_acc: _get_where_implied_math(
        formats_dest_acc[0]
    ),
    test_case=["mixed", "all_ones", "all_zeros"],
    vector_mode=[VectorMode.None_, VectorMode.R, VectorMode.C, VectorMode.RC],
)
def test_sfpu_where_quasar(
    formats_dest_acc, implied_math_format, test_case, vector_mode
):
    """Test ternary where(cond, true_val, false_val) on Quasar."""
    formats, dest_acc = formats_dest_acc
    input_dimensions = [32, 32]
    torch_format_in = format_dict[formats.input_format]
    torch.manual_seed(42)

    src_cond_raw, tile_cnt_single, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    torch.manual_seed(43)
    src_true_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    torch.manual_seed(44)
    src_false_raw, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    condition = _build_condition(src_cond_raw, formats.input_format, test_case)
    true_val = src_true_raw.to(torch_format_in)
    false_val = src_false_raw.to(torch_format_in)
    src_A = torch.cat([condition, true_val, false_val])
    tile_cnt_A = tile_cnt_single * 3
    num_faces = 4

    generate_golden = get_golden_generator(WhereGolden)
    golden_tensor = generate_golden(condition, true_val, false_val).to(
        format_dict[formats.output_format]
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    src_B_dummy = torch.zeros_like(condition)

    configuration = TestConfig(
        _sfpu_source(unpack_to_dest),
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuWhere),
            SFPU_IS_WHERE_OP(),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(_unpacker_engine(unpack_to_dest)),
            DEST_SYNC(),
            VECTOR_MODE(vector_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B_dummy,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_single,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    mask = _processed_face_mask(vector_mode, num_faces)
    assert passed_test(
        golden_tensor[mask], res_tensor[mask], formats.output_format
    ), "Assert against golden failed"
