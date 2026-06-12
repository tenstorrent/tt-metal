# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Consolidated Quasar unary-SFPU test.

Replaces the four standalone Quasar unary-SFPU test pairs (abs, square, rsqrt,
nonlinear) with a single python driver + single cpp source
(`sources/quasar/eltwise_unary_sfpu_quasar_test.cpp`). The operation is selected
at compile time via the `SFPU_UNARY_OPERATION = SfpuType::<op>` constant emitted
by `MATH_OP`, mirroring how Blackhole's `test_zzz_eltwise_unary_sfpu.py` drives a
single `eltwise_unary_sfpu_test.cpp`.

Per-operation sweeps (input dimensions, dest-sync modes, input-stimuli ranges)
are preserved exactly from the original standalone files so coverage is
unchanged — see OP_CONFIGS below.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
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
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

# Formats swept by every op (none are MX formats, so the implied-math-format
# guard below is a no-op for this list — kept for forward-compatibility).
SFPU_UNARY_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)


# ---------------------------------------------------------------------------
# Per-operation input preparation (folded verbatim from the standalone files)
# ---------------------------------------------------------------------------
def prepare_abs_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for absolute value operation with safe value ranges.

    Generates a mix of positive and negative values across the representable
    range. Abs does not change magnitude, so the only constraint is that values
    fit in the input/output format.
    """
    input_torch_format = format_dict[input_format]
    output_torch_format = format_dict[output_format]
    input_finfo = torch.finfo(input_torch_format)
    output_finfo = torch.finfo(output_torch_format)

    # For abs, output magnitude equals input magnitude, so we need values that
    # fit in BOTH input and output formats
    max_safe_value = min(input_finfo.max, output_finfo.max) * 0.9

    # Special handling for bfloat16: limit to reasonable bounds to avoid
    # precision issues at extreme values
    if input_torch_format == torch.bfloat16:
        max_safe_value = min(max_safe_value, 1e4)
    else:
        max_safe_value = min(max_safe_value, input_finfo.max * 0.9)

    min_magnitude = max(1e-6, input_finfo.tiny * 100)  # Avoid denormals

    # Ensure src_A and src_B don't contain inf/nan before normalization
    src_A_float = src_A.to(torch.float32)
    src_B_float = src_B.to(torch.float32)

    # Normalize src_A to [0, 1] range for log-uniform distribution
    src_A_min = src_A_float.min()
    src_A_max = src_A_float.max()
    src_A_normalized = (
        (src_A_float - src_A_min) / (src_A_max - src_A_min)
        if src_A_max > src_A_min
        else torch.zeros_like(src_A_float)
    )

    # Use log-uniform distribution for magnitudes to test across orders of magnitude
    log_min = torch.log(torch.tensor(min_magnitude, dtype=torch.float32))
    log_max = torch.log(torch.tensor(max_safe_value, dtype=torch.float32))
    magnitudes = torch.exp(log_min + src_A_normalized * (log_max - log_min))

    # Randomly assign signs to get both positive and negative values
    src_B_min = src_B_float.min()
    src_B_max = src_B_float.max()
    src_B_normalized = (
        (src_B_float - src_B_min) / (src_B_max - src_B_min)
        if src_B_max > src_B_min
        else torch.zeros_like(src_B_float)
    )
    signs = torch.where(src_B_normalized < 0.5, -1.0, 1.0)

    # Apply signs and clamp to safe range BEFORE converting to input format
    src_A_values = signs * magnitudes
    src_A_values = torch.clamp(src_A_values, -max_safe_value, max_safe_value)
    return src_A_values.to(input_torch_format)


def prepare_square_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for square operation with safe value ranges.

    Applies a log-uniform distribution and clamps values so that the input fits
    the input format and x² fits the output format, across orders of magnitude.
    """
    input_torch_format = format_dict[input_format]
    output_torch_format = format_dict[output_format]
    input_finfo = torch.finfo(input_torch_format)
    output_finfo = torch.finfo(output_torch_format)

    # For squaring, x² must fit in the OUTPUT format
    max_safe_value = math.sqrt(output_finfo.max) * 0.9

    # Special handling for bfloat16: wide range but limited precision
    if input_torch_format == torch.bfloat16:
        max_safe_value = min(max_safe_value, 1e4)  # 10000² = 1e8 fits comfortably
    else:
        # For Float16, ensure input itself fits in input format
        max_safe_value = min(max_safe_value, math.sqrt(input_finfo.max) * 0.9)

    min_magnitude = max(1e-6, input_finfo.tiny * 100)  # Avoid denormals

    # Ensure src_A and src_B don't contain inf/nan before normalization
    src_A_float = src_A.to(torch.float32)
    src_B_float = src_B.to(torch.float32)

    # Normalize src_A to [0, 1] range for log-uniform distribution
    src_A_min = src_A_float.min()
    src_A_max = src_A_float.max()
    src_A_normalized = (
        (src_A_float - src_A_min) / (src_A_max - src_A_min)
        if src_A_max > src_A_min
        else torch.zeros_like(src_A_float)
    )

    # Use log-uniform distribution for magnitudes to test across orders of magnitude
    log_min = torch.log(torch.tensor(min_magnitude, dtype=torch.float32))
    log_max = torch.log(torch.tensor(max_safe_value, dtype=torch.float32))
    magnitudes = torch.exp(log_min + src_A_normalized * (log_max - log_min))

    # Randomly assign signs to get both positive and negative values
    src_B_min = src_B_float.min()
    src_B_max = src_B_float.max()
    src_B_normalized = (
        (src_B_float - src_B_min) / (src_B_max - src_B_min)
        if src_B_max > src_B_min
        else torch.zeros_like(src_B_float)
    )
    signs = torch.where(src_B_normalized < 0.5, -1.0, 1.0)

    # Apply signs and clamp to safe range BEFORE converting to input format
    src_A_values = signs * magnitudes
    src_A_values = torch.clamp(src_A_values, -max_safe_value, max_safe_value)
    return src_A_values.to(input_torch_format)


def prepare_rsqrt_inputs(
    src_A: torch.Tensor,
    input_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for rsqrt across the full representable range using a
    log-uniform distribution (rsqrt accepts only positive inputs).
    """
    torch_format = format_dict[input_format]
    finfo = torch.finfo(torch_format)
    min_val = max(1e-6, finfo.tiny * 100)
    max_val = finfo.max

    # Transform uniform [0,1) to log-uniform [min_val, max_val]
    log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
    log_max = torch.log(torch.tensor(float(max_val), dtype=torch.float32))
    return torch.exp(log_min + src_A.to(torch.float32) * (log_max - log_min)).to(
        torch_format
    )


def prepare_inputs_for_operation(
    src_A: torch.Tensor,
    mathop: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat = None,
) -> torch.Tensor:
    """
    Prepare input tensor for the nonlinear ops (exp, gelu, relu, reciprocal,
    sqrt, tanh, sigmoid, silu) with operation-specific safe value ranges.
    """
    torch_format = format_dict[input_format]

    if mathop == MathOperation.Exp:
        # Scale to range [-10, 10] for exp - avoids overflow while testing meaningful range
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.Gelu:
        # Scale to range [-10, 10] for gelu - balanced negative/near-zero/positive coverage
        min_val = -10.0
        max_val = 10.0
        src_A = torch.empty_like(src_A, dtype=torch.float32).uniform_(min_val, max_val)
    elif mathop == MathOperation.Relu:
        # Scale to range including negative and positive values for ReLU testing
        finfo = torch.finfo(torch_format)
        min_val = finfo.min / 2  # Use half range to avoid extremes
        max_val = finfo.max / 2
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.Sqrt:
        # Scale to positive range using log-uniform distribution.
        # CRITICAL: golden converts input -> output format FIRST, then computes sqrt,
        # so the input must fit in the output format when converted.
        finfo = torch.finfo(torch_format)
        min_val = max(1e-6, finfo.tiny * 100)
        if output_format:
            output_torch_format = format_dict[output_format]
            output_finfo = torch.finfo(output_torch_format)
            if output_torch_format in (torch.float16, torch.bfloat16):
                max_input_for_format = output_finfo.max  # Input must fit in output
                max_safe_sqrt = output_finfo.max * 0.95  # Leave 5% headroom
                max_input_for_sqrt = max_safe_sqrt**2  # Max input so sqrt fits
                max_val = min(finfo.max, max_input_for_format, max_input_for_sqrt)
                max_val = min(max_val, output_finfo.max * 0.8)  # extra safety
            else:
                max_val = finfo.max
        else:
            if torch_format in (torch.float16, torch.bfloat16):
                max_val = min(finfo.max, 1e4)  # sqrt(1e4) = 100, safe for 16-bit
            else:
                max_val = finfo.max  # Float32 can handle larger values
        # Transform uniform [0,1) to log-uniform [min_val, max_val]
        log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
        log_max = torch.log(torch.tensor(float(max_val), dtype=torch.float32))
        src_A_float32 = torch.exp(
            log_min + src_A.to(torch.float32) * (log_max - log_min)
        )
        src_A_float32 = torch.clamp(src_A_float32, min_val, max_val)

        # Final safety: ensure values fit in output format when converted
        if output_format and output_format in (
            DataFormat.Float16,
            DataFormat.Float16_b,
        ):
            output_torch_format = format_dict[output_format]
            output_finfo = torch.finfo(output_torch_format)
            src_A_converted = src_A_float32.to(output_torch_format)
            if torch.any(torch.isinf(src_A_converted)):
                max_safe_input = output_finfo.max * 0.8
                src_A_float32 = torch.clamp(src_A_float32, min_val, max_safe_input)

        src_A = src_A_float32.to(torch_format)

        # After converting to input format, re-verify values still fit in output format
        if output_format and output_format in (
            DataFormat.Float16,
            DataFormat.Float16_b,
        ):
            output_torch_format = format_dict[output_format]
            output_finfo = torch.finfo(output_torch_format)
            src_A_converted = src_A.to(output_torch_format)
            if torch.any(torch.isinf(src_A_converted)):
                max_safe_input = output_finfo.max * 0.75  # Very conservative
                src_A_float32 = src_A.to(torch.float32)
                src_A_float32 = torch.clamp(src_A_float32, min_val, max_safe_input)
                src_A = src_A_float32.to(torch_format)
    elif mathop == MathOperation.Reciprocal:
        # Scale to range avoiding zero to prevent division by zero
        finfo = torch.finfo(torch_format)
        min_val = max(1e-6, finfo.tiny * 100)
        max_val = finfo.max / 2  # Avoid very large values that might underflow
        log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
        log_max = torch.log(torch.tensor(float(max_val), dtype=torch.float32))
        src_A_float32 = torch.exp(
            log_min + src_A.to(torch.float32) * (log_max - log_min)
        )
        src_A_float32 = torch.where(
            torch.abs(src_A_float32) < min_val,
            torch.sign(src_A_float32) * min_val,
            src_A_float32,
        )
        src_A = src_A_float32.to(torch_format)
    elif mathop == MathOperation.Tanh:
        # Scale to range [-10, 10] for tanh
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.Sigmoid:
        # Scale to range [-10, 10] for sigmoid
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.Silu:
        # Scale to range [-10, 10] for SiLU (avoid overflow with negative exponential)
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    # else: keep src_A as-is

    return src_A


def prepare_unary_inputs(
    mathop: MathOperation,
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """Dispatch to the op-specific input-preparation routine."""
    if mathop == MathOperation.Abs:
        return prepare_abs_inputs(src_A, src_B, input_format, output_format)
    if mathop == MathOperation.Square:
        return prepare_square_inputs(src_A, src_B, input_format, output_format)
    if mathop == MathOperation.Rsqrt:
        return prepare_rsqrt_inputs(src_A, input_format)
    return prepare_inputs_for_operation(src_A, mathop, input_format, output_format)


# ---------------------------------------------------------------------------
# Per-operation sweep configuration (preserves the original standalone coverage)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpConfig:
    mathop: MathOperation
    input_dims: tuple  # list of [H, W] dimensions
    dest_sync_modes: tuple  # DestSync values to sweep
    # abs/square seed the RNG (default stimuli spec); rsqrt/nonlinear use a
    # uniform [0, 1) stimuli spec with no explicit seed — preserved per-op.
    seed: Optional[int] = None
    uniform_spec: bool = False


_THREE_DIMS = ([32, 32], [64, 64], [32, 64])
_TWO_DIMS = ([32, 32], [64, 64])
_HALF_FULL = (DestSync.Half, DestSync.Full)
_HALF_ONLY = (DestSync.Half,)  # abs did not sweep dest_sync (DEST_SYNC() default)

OP_CONFIGS = [
    OpConfig(MathOperation.Abs, _THREE_DIMS, _HALF_ONLY, seed=42),
    OpConfig(MathOperation.Square, _THREE_DIMS, _HALF_FULL, seed=42),
    OpConfig(MathOperation.Rsqrt, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
    # Nonlinear ops: identical [32,32]/[64,64] × Half/Full × uniform-spec sweep.
    OpConfig(MathOperation.Exp, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
    OpConfig(MathOperation.Gelu, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
    OpConfig(MathOperation.Relu, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
    OpConfig(MathOperation.Reciprocal, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
    OpConfig(MathOperation.Sqrt, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
    OpConfig(MathOperation.Tanh, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
    OpConfig(MathOperation.Sigmoid, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
    OpConfig(MathOperation.Silu, _TWO_DIMS, _HALF_FULL, uniform_spec=True),
]

OP_CONFIG_BY_MATHOP = {cfg.mathop: cfg for cfg in OP_CONFIGS}


def generate_sfpu_unary_combinations(formats_list: List[FormatConfig]):
    """
    Build the full unary-SFPU sweep across all ops, preserving each op's exact
    original coverage.

    Returns: list of (mathop, fmt, dest_acc, dest_sync, implied_math_format,
    input_dimensions) tuples.
    """
    combinations = []
    for cfg in OP_CONFIGS:
        for fmt in formats_list:
            in_fmt = fmt.input_format

            dest_acc_modes = (
                (DestAccumulation.Yes,)
                if in_fmt.is_32_bit()
                else (DestAccumulation.No, DestAccumulation.Yes)
            )
            for dest_acc in dest_acc_modes:
                # Skip invalid format combinations for Quasar
                if is_invalid_quasar_sfpu_format_combination(fmt, dest_acc):
                    continue

                for dest_sync in cfg.dest_sync_modes:
                    for implied_math_format in [
                        ImpliedMathFormat.No,
                        ImpliedMathFormat.Yes,
                    ]:
                        # MX formats require implied_math_format=Yes
                        if (
                            in_fmt.is_mx_format()
                            and implied_math_format == ImpliedMathFormat.No
                        ):
                            continue

                        for input_dimensions in cfg.input_dims:
                            combinations.append(
                                (
                                    cfg.mathop,
                                    fmt,
                                    dest_acc,
                                    dest_sync,
                                    implied_math_format,
                                    input_dimensions,
                                )
                            )

    return combinations


@pytest.mark.quasar
@parametrize(
    mathop_formats_dest_acc_sync_implied_math_input_dims=generate_sfpu_unary_combinations(
        SFPU_UNARY_FORMATS
    ),
)
def test_eltwise_unary_sfpu_quasar(
    mathop_formats_dest_acc_sync_implied_math_input_dims,
):
    """
    Consolidated unary-SFPU test on Quasar. One compile-time-selected op per
    variant (abs, exp, gelu, relu, reciprocal, sqrt, tanh, sigmoid, silu, rsqrt,
    square), validated against the UnarySFPUGolden reference.
    """
    (mathop, formats, dest_acc, dest_sync, implied_math_format, input_dimensions) = (
        mathop_formats_dest_acc_sync_implied_math_input_dims[0]
    )

    cfg = OP_CONFIG_BY_MATHOP[mathop]
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    spec = StimuliSpec.uniform(low=0.0, high=1.0) if cfg.uniform_spec else None
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    # Prepare inputs with operation-specific ranges
    src_A = prepare_unary_inputs(
        mathop, src_A, src_B, formats.input_format, formats.output_format
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
        "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
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

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
