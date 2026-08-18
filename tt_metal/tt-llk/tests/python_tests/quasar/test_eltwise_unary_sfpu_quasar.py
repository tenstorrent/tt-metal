# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.sfpu_domains import exclude_undefined, for_op_pipeline
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    StimuliSpec,
    apply_log_uniform_magnitudes,
    compute_safe_input_magnitude_range,
    format_elem_max,
    generate_stimuli,
)
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
    TemplateParameter,
)
from helpers.tile_constants import (
    MX_SUPPORTED_TILE_SIZES,
    SUPPORTED_TILE_SIZES,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed the RNG once per test so stimuli are deterministic across runs."""
    torch.manual_seed(42)


# Formats swept by every op (none are MX formats, so the implied-math-format
# guard below is a no-op for this list — kept for forward-compatibility).
SFPU_UNARY_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)

# Extra L1 storage formats from the Tensix Formats specification. SFPU never
# computes in an MX encoding: the unpacker converts each of these to a native
# Float16_b register value, and the packer converts the result back to L1.
# Tf32 similarly occupies Float32 containers in Dest, so it is forced to the
# 32-bit Dest mode by generate_sfpu_unary_combinations().
# The specification also lists standalone Fp8R/Fp8P, MxFp6R/MxFp6P, and
# Quasar's MxFp4 2x modes; the Python DataFormat model does not expose those
# encodings yet, so this is the complete currently representable set.
SFPU_PARITY_L1_FORMATS = input_output_formats(
    [
        DataFormat.Tf32,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
    same=True,
)
SFPU_PARITY_FLOAT_FORMATS = SFPU_UNARY_FORMATS + SFPU_PARITY_L1_FORMATS

# The trigonometry / inverse-hyperbolic transcendentals. Float-only (they share the
# SFPU_UNARY_FORMATS set), each with its own safe input domain (see prepare_trig_inputs).
TRIGONOMETRY_OPS = [
    MathOperation.Sin,
    MathOperation.Cos,
    MathOperation.Tan,
    MathOperation.Atan,
    MathOperation.Asin,
    MathOperation.Acos,
    MathOperation.Sinh,
    MathOperation.Cosh,
    MathOperation.Acosh,
    MathOperation.Asinh,
    MathOperation.Atanh,
]

# The six comparison-to-zero modes. These run integer formats too (and UInt16 via
# the Int16 container), so the sweep adds them to the float formats above.
COMP_OPS = [
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
]

SFPI_PARITY_SHIFT_OPS = (
    MathOperation.LeftShift,
    MathOperation.RightShift,
)

# Extra (integer) formats only the comp family sweeps. Int32/Int16/Int8 (signed) and UInt8
# (unsigned) use their native Quasar dest format. UInt16 is the exception: it has no native Quasar
# dest format, so the inference routes its data path through Int16 and sets FormatConfig.sfpu_math=
# UInt16, the only stage the comp kernel reads as uint16.
SFPU_COMP_EXTRA_FORMATS = input_output_formats(
    [
        DataFormat.Int32,
        DataFormat.Int16,
        DataFormat.Int8,
        DataFormat.UInt16,
        DataFormat.UInt8,
    ],
    same=True,
)


# ---------------------------------------------------------------------------
# Per-operation input preparation (folded verbatim from the standalone files)
# ---------------------------------------------------------------------------
def _log_uniform_signed_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    max_safe_value: float,
) -> torch.Tensor:
    """
    Shared input builder for abs/square.

    Produces a log-uniform magnitude distribution across orders of magnitude
    with random signs, clamped to ``max_safe_value`` and converted to
    ``input_format``. ``src_A`` seeds the magnitudes and ``src_B`` the signs;
    callers supply the op-specific ``max_safe_value`` ceiling.
    """
    input_torch_format = format_dict[input_format]
    input_finfo = torch.finfo(input_torch_format)

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


def prepare_abs_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for absolute value operation with safe value ranges.

    Abs preserves magnitude, so values only need to fit in BOTH the input and
    output formats; the shared log-uniform builder handles the distribution.
    """
    input_torch_format = format_dict[input_format]
    input_finfo = torch.finfo(input_torch_format)
    output_finfo = torch.finfo(format_dict[output_format])

    # For abs, output magnitude equals input magnitude, so values must fit in
    # BOTH input and output formats.
    max_safe_value = min(input_finfo.max, output_finfo.max) * 0.9
    # Special handling for bfloat16: limit to reasonable bounds to avoid
    # precision issues at extreme values.
    if input_torch_format == torch.bfloat16:
        max_safe_value = min(max_safe_value, 1e4)
    else:
        max_safe_value = min(max_safe_value, input_finfo.max * 0.9)

    return _log_uniform_signed_inputs(src_A, src_B, input_format, max_safe_value)


def prepare_square_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for square operation with safe value ranges.

    For squaring, x² must fit in the OUTPUT format, so the magnitude ceiling is
    derived from sqrt(output_max); the shared log-uniform builder handles the
    distribution.
    """
    input_torch_format = format_dict[input_format]
    input_finfo = torch.finfo(input_torch_format)
    output_finfo = torch.finfo(format_dict[output_format])

    # For squaring, x² must fit in the OUTPUT format.
    max_safe_value = math.sqrt(output_finfo.max) * 0.9
    # Special handling for bfloat16: wide range but limited precision.
    if input_torch_format == torch.bfloat16:
        max_safe_value = min(max_safe_value, 1e4)  # 10000² = 1e8 fits comfortably
    else:
        # For Float16, ensure the input itself fits in the input format.
        max_safe_value = min(max_safe_value, math.sqrt(input_finfo.max) * 0.9)

    return _log_uniform_signed_inputs(src_A, src_B, input_format, max_safe_value)


def prepare_inputs_for_operation(
    src_A: torch.Tensor,
    mathop: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat = None,
) -> torch.Tensor:
    """
    Prepare input tensor for the nonlinear ops (exp, gelu, relu, reciprocal,
    sqrt, rsqrt, tanh, sigmoid, silu) with operation-specific safe value ranges.
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
    elif mathop in (MathOperation.Sqrt, MathOperation.SqrtCustom):
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
                if output_torch_format == torch.bfloat16:
                    # BF16's enormous exponent range otherwise dominates this
                    # log sweep with 1e30-scale inputs. SqrtCustom's contract is
                    # an approximate square root, and its existing tolerance is
                    # meaningful over this still-four-decade domain rather than
                    # at the final BF16 exponent where a one-ULP result shift is
                    # itself enormous.
                    max_val = min(max_val, 1e4)
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
    elif mathop == MathOperation.Log:
        # Log's valid domain is positive. Log-uniform magnitudes exercise the
        # exponent-reduction path rather than clustering around one decade.
        finfo = torch.finfo(torch_format)
        min_val = max(1e-6, finfo.tiny * 100)
        max_val = min(float(finfo.max), 1e6)
        log_min = math.log(min_val)
        log_max = math.log(max_val)
        src_A = torch.exp(
            torch.tensor(log_min, dtype=torch.float32)
            + src_A.to(torch.float32) * (log_max - log_min)
        ).to(torch_format)
    elif mathop == MathOperation.Log1p:
        # Cover the high-curvature region around -1, zero, and a broad positive
        # tail while staying inside log1p's real-valued domain.
        src_A = (-0.99 + src_A.to(torch.float32) * 100.99).to(torch_format)
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
    elif mathop == MathOperation.Rsqrt:
        # Full representable range via log-uniform distribution
        # (rsqrt accepts only positive inputs).
        finfo = torch.finfo(torch_format)
        min_val = max(1e-6, finfo.tiny * 100)
        max_val = finfo.max
        log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
        log_max = torch.log(torch.tensor(float(max_val), dtype=torch.float32))
        src_A = torch.exp(log_min + src_A.to(torch.float32) * (log_max - log_min)).to(
            torch_format
        )
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
    elif mathop == MathOperation.Clamp:
        # Clamp bounds are fixed to [-1, 1]; span past both to exercise the lower/upper/pass-through
        # cases (mirrors sfpu_domains' Clamp spec).
        min_val = -2.0
        max_val = 2.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.Neg:
        # Negation is exact for any representable value; span both signs (mirrors sfpu_domains' Neg spec).
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.Softplus:
        # Span both signs and past the linear threshold (20) so the kernel's polynomial region, the
        # negative saturation region, and the linear passthrough (t > threshold -> softplus ~= x) are
        # all covered (mirrors sfpu_domains' Softplus spec).
        min_val = -8.0
        max_val = 30.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    # else: keep src_A as-is

    return src_A


def prepare_trig_inputs(
    src_A: torch.Tensor,
    mathop: MathOperation,
    input_format: DataFormat,
) -> torch.Tensor:
    """
    Map the uniform [0, 1] stimulus into each op's safe domain so the Quasar kernel
    stays in its accurate range:
      sin / cos — [-pi, pi] (argument reduction is valid far wider, but a small domain
                  keeps the Maclaurin polynomial precise).
      asinh    — [-10, 10] (log polynomial stable away from overflow).
      acosh    — [1.1, 50]  (x >= 1 domain; avoid near-1 where the 3rd-order log loses
                  precision).
      atanh    — [-0.9, 0.9] (|x| < 1 domain with margin so RECIP does not blow up).
    """
    torch_format = format_dict[input_format]
    u = src_A.to(torch.float32)  # uniform [0, 1] from the uniform stimuli spec

    if mathop in (MathOperation.Sin, MathOperation.Cos):
        lo, hi = -math.pi, math.pi
    elif mathop == MathOperation.Tan:
        # Stay away from odd pi/2 poles; range reduction and both signs remain covered.
        lo, hi = -math.pi / 3.0, math.pi / 3.0
    elif mathop == MathOperation.Atan:
        lo, hi = -100.0, 100.0
    elif mathop in (MathOperation.Asin, MathOperation.Acos):
        lo, hi = -1.0, 1.0
    elif mathop in (MathOperation.Sinh, MathOperation.Cosh):
        # Keeps fp16 outputs finite while covering the exp-based tails.
        lo, hi = -8.0, 8.0
    elif mathop == MathOperation.Asinh:
        lo, hi = -10.0, 10.0
    elif mathop == MathOperation.Acosh:
        lo, hi = 1.1, 50.0
    elif mathop == MathOperation.Atanh:
        lo, hi = -0.9, 0.9
    else:
        return src_A

    return (lo + u * (hi - lo)).to(torch_format)


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
    if mathop in TRIGONOMETRY_OPS:
        return prepare_trig_inputs(src_A, mathop, input_format)
    if mathop in COMP_OPS:
        # Unsigned formats need non-negative stimuli (a signed split would wrap under the unsigned
        # dtype); signed formats use the sign-vs-magnitude builder.
        if input_format in (DataFormat.UInt16, DataFormat.UInt8):
            return prepare_comp_inputs_uint(src_A, src_B, input_format)
        return prepare_comp_inputs(src_A, src_B, input_format, output_format)
    if mathop in (
        MathOperation.UnaryGt,
        MathOperation.UnaryLt,
        MathOperation.UnaryGe,
        MathOperation.UnaryLe,
        MathOperation.UnaryEq,
        MathOperation.UnaryNe,
    ):
        # These ported kernels compare against the compile-time scalar 0.5.
        # Include exact equal and values on both sides so the Quasar workaround
        # is exercised for all affected predicates (<, >=, ==, !=), rather
        # than relying on a random draw to land exactly on the boundary.
        pattern = torch.tensor(
            [-2.0, -0.0, 0.0, 0.25, 0.5, 0.75, 2.0],
            dtype=format_dict[input_format],
        )
        return pattern.repeat((src_A.numel() + pattern.numel() - 1) // pattern.numel())[
            : src_A.numel()
        ]
    if mathop in SFPI_PARITY_SHIFT_OPS:
        # The public unary-shift entry points use the fixed immediate 3. Match
        # the existing BH coverage contract and keep left shifts inside the
        # positive Int32 range so wraparound is not mistaken for a port error.
        pattern = torch.tensor(
            [0, 1, 2, 7, 8, 31, 255, 65_535, 1_000_000], dtype=torch.int32
        )
        return pattern.repeat((src_A.numel() + pattern.numel() - 1) // pattern.numel())[
            : src_A.numel()
        ]
    return prepare_inputs_for_operation(src_A, mathop, input_format, output_format)


def prepare_comp_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for comparison-to-zero operations.

    Mixes positive, negative, exact +0.0/-0.0, and small-magnitude values so the
    sign-vs-magnitude split (ltz/gtz are sign tests; eqz/nez are magnitude tests)
    is exercised. Avoids NaN/subnormal stimuli, which SFPSETCC does not special-case
    and on which Quasar and an IEEE golden could disagree.
    """
    input_torch_format = format_dict[input_format]

    # Integer formats (Int32 / Int16, both signed): comparison-to-zero only depends on sign and
    # zero-ness, not magnitude. The default integer stimuli are non-negative, so split src_B about
    # its median to sign roughly half the lanes negative (spread across every face), then seed a few
    # exact zeros/extremes to exercise all six modes.
    if not input_torch_format.is_floating_point:
        big = torch.iinfo(input_torch_format).max // 8
        src_B_float = src_B.to(torch.float32)
        signs = torch.where(src_B_float < src_B_float.median(), -1, 1)
        values = (src_A.to(torch.int64) % big) * signs

        flat = values.flatten()
        for i, seed in enumerate([0, 1, -1, big, -big, 2]):
            if i < flat.numel():
                flat[i] = seed
        return flat.reshape(values.shape).to(input_torch_format)

    src_A_float = src_A.to(torch.float32)
    src_B_float = src_B.to(torch.float32)

    # Magnitudes in a comfortably-representable range, signed by src_B. src_B is non-negative under
    # the default spec, so split it about its median to sign roughly half the lanes negative.
    magnitudes = torch.clamp(torch.abs(src_A_float) * 0.5 + 0.5, 0.1, 100.0)
    signs = torch.where(src_B_float < src_B_float.median(), -1.0, 1.0)
    values = signs * magnitudes

    flat = values.flatten()
    # Seed exact zeros of both signs and a few small-magnitude values to pin
    # down the sign-vs-magnitude behaviour at the origin.
    if flat.numel() >= 8:
        flat[0] = 0.0  # +0.0
        flat[1] = -0.0  # -0.0
        flat[2] = 1.0
        flat[3] = -1.0
        flat[4] = 0.5
        flat[5] = -0.5
        flat[6] = 2.0
        flat[7] = -2.0
    values = flat.reshape(values.shape)

    return values.to(input_torch_format)


def prepare_comp_inputs_uint(
    src_A: torch.Tensor, src_B: torch.Tensor, input_format: DataFormat
) -> torch.Tensor:
    """
    Non-negative stimuli for an unsigned comp path (UInt8 / UInt16).

    UInt16 rides the Int16/SMAG16 container, so its values are kept in [0, 32767] where the bit
    pattern is identical read as signed or unsigned. UInt8 uses its native UINT8 dest, so it spans
    the full [0, 255] range (bit 7 set is exercised). Seeds exact zero and a couple of extremes so
    every comparison mode is hit; the signed and unsigned goldens coincide on non-negative inputs.
    """
    # Signed-safe magnitude ceiling: half-range for UInt16 (Int16 container), full range for UInt8.
    hi = 32767 if input_format == DataFormat.UInt16 else 255
    values = (src_A.to(torch.int64).abs() % (hi + 1)) | (
        src_B.to(torch.int64).abs() % 256
    )  # mix in low bits from B for variety, stays non-negative
    values = values % (hi + 1)

    flat = values.flatten()
    for i, seed in enumerate([0, 1, 2, hi, 100, 0]):
        if i < flat.numel():
            flat[i] = seed
    return flat.reshape(values.shape).to(format_dict[input_format])


# ---------------------------------------------------------------------------
# Typecast: a *conversion* op whose applicability is per (src, dst) format pair,
# not per single format, so it cannot register in the generic unary-SFPU format
# matrix above. It is folded in here as a Typecast-aware OpConfig that carries its
# own pair sweep (TYPECAST_CASES) and input builder.
#
# The full reference matrix of SFPU arithmetic casts is swept (both directions of
# every reference-list pair), excluding two families: block-float (Bfp8_b / Bfp4_b),
# which are a pure unpack/pack gasket datacopy — not an SFPU op — and UInt32, which
# Quasar's DataFormat enum does not define. Each cast is one of:
#   float<->float : widen (store) or RNE narrow to fp16 (round-nearest-even)
#   float<->int32 : SFPCAST
#   float->narrow int : clamp negatives (unsigned) + RNE narrow
#   int->float : SFPCAST (+ fp16 narrow if the dst is fp16)
#   int<->int : store sfpmem mode (widen/equal) or RNE narrow to 8-bit
#
# The functor `calculate_typecast<IN_FMT, OUT_FMT>` needs the format pair at
# COMPILE time, but the unified dispatcher only carries `SfpuType` at compile time
# and formats at runtime. We bridge that with the `TYPECAST_FORMATS` template param,
# which bakes the pair as `constexpr DataFormat TYPECAST_IN_FORMAT / TYPECAST_OUT_FORMAT`
# per build variant.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TypecastCase:
    src: DataFormat
    dst: DataFormat


_TYPECAST_PAIRS = (
    (DataFormat.Float16_b, DataFormat.Float32),
    (DataFormat.Float16_b, DataFormat.Int32),
    (DataFormat.Float16_b, DataFormat.UInt8),
    (DataFormat.Float16_b, DataFormat.UInt16),
    (DataFormat.Float32, DataFormat.Int32),
    (DataFormat.Float32, DataFormat.UInt8),
    (DataFormat.Float32, DataFormat.UInt16),
    (DataFormat.UInt16, DataFormat.Int32),
    (DataFormat.UInt16, DataFormat.UInt8),
    # Int16 (signed 16-bit) — not in the ttnn typecast matrix, but the kernel handles it on every
    # path (float<->int16 via SFPCAST + 16-bit store-narrow, int16<->int via the int->int path), so
    # it is swept here too. Mirrors the UInt16 set; Int16 has a native Quasar dest format.
    (DataFormat.Float16_b, DataFormat.Int16),
    (DataFormat.Float32, DataFormat.Int16),
    (DataFormat.Int16, DataFormat.Int32),
    (DataFormat.Int16, DataFormat.UInt8),
)

# Expand each unordered pair into both cast directions.
TYPECAST_CASES = tuple(
    TypecastCase(a, b)
    for src, dst in _TYPECAST_PAIRS
    for a, b in ((src, dst), (dst, src))
)

_RANGE_SAFETY_FACTOR = 0.9


def _prepare_typecast_input(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    src_format: DataFormat,
    dst_format: DataFormat,
) -> torch.Tensor:
    """Pick stimuli that round-trip cleanly through both endpoints, so the identity
    golden matches the hardware conversion element-for-element."""
    if src_format.is_integer() or dst_format.is_integer():
        # At least one integer endpoint. Constrain the raw stimulus (which spans the full
        # format range) to an integer-valued band that BOTH endpoints represent exactly, so
        # the hardware's round-nearest-even and the golden's torch cast agree everywhere:
        #  - non-negative if either endpoint is unsigned (the hardware clamps negatives to 0);
        #  - capped to the narrowest endpoint: UInt8 -> 255; a Float16_b/Float16 (bf16/fp16)
        #    endpoint is integer-exact only up to 256, so cap there; otherwise a wide band.
        # Normalising first makes this independent of the raw stimulus range (otherwise
        # scaling a full-range int32 overflows to INT32_MIN).
        formats = (src_format, dst_format)
        has_unsigned = any(f in (DataFormat.UInt8, DataFormat.UInt16) for f in formats)
        if DataFormat.UInt8 in formats:
            cap = 255.0
        elif any(f in (DataFormat.Float16_b, DataFormat.Float16) for f in formats):
            cap = 200.0  # bf16/fp16 is integer-exact only to 256
        else:
            cap = 1000.0
        lo = 0.0 if has_unsigned else -cap

        af = src_A.to(torch.float32)
        span = af.max() - af.min()
        norm = (af - af.min()) / span if span > 0 else torch.zeros_like(af)
        vals = lo + norm * (cap - lo)
        return vals.round().to(format_dict[src_format])

    # Float endpoints: log-uniform magnitudes inside both formats' representable ranges,
    # so values stay accurate through the narrowing cast.
    input_cap = format_elem_max(src_format) * _RANGE_SAFETY_FACTOR
    output_cap = format_elem_max(dst_format) * _RANGE_SAFETY_FACTOR
    min_magnitude, max_magnitude = compute_safe_input_magnitude_range(
        src_format,
        dst_format,
        input_magnitude_cap=input_cap,
        output_magnitude_cap=output_cap,
    )
    return apply_log_uniform_magnitudes(
        src_A,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        cast_to_format=src_format,
        sign_source=src_B,
    )


# ---------------------------------------------------------------------------
# Per-operation sweep configuration.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpConfig:
    mathop: MathOperation
    tile_cases: tuple  # ((input H/W, tile H/W), ...)
    dest_sync_modes: tuple  # DestSync values to sweep
    uniform_spec: bool = False
    orthogonal: bool = False


@dataclass
class QUASAR_SFPU_UNARY_OP(TemplateParameter):
    """Select a unary op without requiring a production Quasar SfpuType entry."""

    mathop: MathOperation

    def convert_to_cpp(self) -> str:
        return (
            "constexpr auto SFPU_UNARY_OPERATION = "
            f"QuasarSfpuTestOperation::{self.mathop.cpp_enum_value};"
        )


DEFAULT_TILE_CASES = (
    ((32, 32), (32, 32)),
    ((64, 64), (32, 32)),
)
PARITY_TILE_CASES = tuple((shape, shape) for shape in SUPPORTED_TILE_SIZES) + (
    ((64, 64), (32, 32)),
)
DEST_SYNC_MODES = (DestSync.Half, DestSync.Full)

OP_CONFIGS = [
    OpConfig(MathOperation.Abs, DEFAULT_TILE_CASES, DEST_SYNC_MODES),
    OpConfig(MathOperation.Square, DEFAULT_TILE_CASES, DEST_SYNC_MODES),
    OpConfig(
        MathOperation.Rsqrt,
        DEFAULT_TILE_CASES,
        DEST_SYNC_MODES,
        uniform_spec=True,
    ),
    # Nonlinear ops: identical [32,32]/[64,64] × Half/Full × uniform-spec sweep.
    OpConfig(MathOperation.Exp, DEFAULT_TILE_CASES, DEST_SYNC_MODES, uniform_spec=True),
    # Accurate FP32 GELU directly exercises ckernel_sfpu_piecewise_rational.h.
    OpConfig(
        MathOperation.Gelu,
        PARITY_TILE_CASES,
        DEST_SYNC_MODES,
        uniform_spec=True,
        orthogonal=True,
    ),
    OpConfig(
        MathOperation.Relu, DEFAULT_TILE_CASES, DEST_SYNC_MODES, uniform_spec=True
    ),
    OpConfig(
        MathOperation.Reciprocal,
        DEFAULT_TILE_CASES,
        DEST_SYNC_MODES,
        uniform_spec=True,
    ),
    OpConfig(
        MathOperation.Sqrt, DEFAULT_TILE_CASES, DEST_SYNC_MODES, uniform_spec=True
    ),
    OpConfig(
        MathOperation.SqrtCustom,
        PARITY_TILE_CASES,
        DEST_SYNC_MODES,
        uniform_spec=True,
        orthogonal=True,
    ),
    OpConfig(
        MathOperation.Log,
        PARITY_TILE_CASES,
        DEST_SYNC_MODES,
        uniform_spec=True,
        orthogonal=True,
    ),
    OpConfig(
        MathOperation.Log1p,
        PARITY_TILE_CASES,
        DEST_SYNC_MODES,
        uniform_spec=True,
        orthogonal=True,
    ),
    OpConfig(
        MathOperation.Tanh, DEFAULT_TILE_CASES, DEST_SYNC_MODES, uniform_spec=True
    ),
    OpConfig(
        MathOperation.Sigmoid, DEFAULT_TILE_CASES, DEST_SYNC_MODES, uniform_spec=True
    ),
    OpConfig(
        MathOperation.Silu, DEFAULT_TILE_CASES, DEST_SYNC_MODES, uniform_spec=True
    ),
    OpConfig(
        MathOperation.Clamp,
        PARITY_TILE_CASES,
        DEST_SYNC_MODES,
        uniform_spec=True,
        orthogonal=True,
    ),
    OpConfig(MathOperation.Neg, DEFAULT_TILE_CASES, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(
        MathOperation.Softplus,
        PARITY_TILE_CASES,
        DEST_SYNC_MODES,
        uniform_spec=True,
        orthogonal=True,
    ),
    OpConfig(MathOperation.Typecast, DEFAULT_TILE_CASES, DEST_SYNC_MODES),
    # Trigonometry / inverse-hyperbolic ops: same matrix as the other transcendentals,
    # fed a uniform [0, 1] stimulus that prepare_trig_inputs maps into each op's domain.
    *[
        OpConfig(
            op, PARITY_TILE_CASES, DEST_SYNC_MODES, uniform_spec=True, orthogonal=True
        )
        for op in TRIGONOMETRY_OPS
    ],
] + [OpConfig(op, DEFAULT_TILE_CASES, DEST_SYNC_MODES) for op in COMP_OPS]

# Conventional unary leaves from the 57-family Blackhole SFPI parity list that
# were not already present in the Quasar sweep above. Layout-sensitive integer
# operations (bitwise), reductions, and multi-input kernels use dedicated
# harnesses instead of being forced through this float-unary path. Unary shifts
# are Int32 elementwise operations and use this harness's exact integer path.
SFPI_PARITY_NEW_UNARY_OPS = (
    MathOperation.Hardsigmoid,
    MathOperation.Add1,
    MathOperation.CastFp32ToFp16a,
    MathOperation.Cbrt,
    MathOperation.Celu,
    MathOperation.Digamma,
    MathOperation.Elu,
    MathOperation.Erf,
    MathOperation.Erfc,
    MathOperation.Erfinv,
    MathOperation.Exp2,
    MathOperation.Expm1,
    MathOperation.Hardmish,
    MathOperation.Hardshrink,
    MathOperation.Hardtanh,
    MathOperation.Heaviside,
    MathOperation.I0,
    MathOperation.I1,
    MathOperation.Identity,
    MathOperation.Lgamma,
    MathOperation.Polygamma,
    MathOperation.Prelu,
    MathOperation.Rdiv,
    MathOperation.Rpow,
    MathOperation.Selu,
    MathOperation.Sign,
    MathOperation.Softshrink,
    MathOperation.Softsign,
    MathOperation.Tanhshrink,
    MathOperation.UnaryGt,
    MathOperation.UnaryLt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLe,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
    MathOperation.UnaryPower,
    MathOperation.Xielu,
)

OP_CONFIGS.extend(
    OpConfig(op, PARITY_TILE_CASES, DEST_SYNC_MODES, orthogonal=True)
    for op in SFPI_PARITY_NEW_UNARY_OPS
)
OP_CONFIGS.extend(
    OpConfig(op, PARITY_TILE_CASES, DEST_SYNC_MODES, orthogonal=True)
    for op in SFPI_PARITY_SHIFT_OPS
)

OP_CONFIG_BY_MATHOP = {cfg.mathop: cfg for cfg in OP_CONFIGS}


def formats_for_op(cfg: OpConfig) -> List[InputOutputFormat]:
    """Float formats for every op, plus the integer/UInt16 formats only comp sweeps."""
    if cfg.mathop == MathOperation.Typecast:
        return [InputOutputFormat(case.src, case.dst) for case in TYPECAST_CASES]
    if cfg.mathop == MathOperation.CastFp32ToFp16a:
        return [InputOutputFormat(DataFormat.Float32, DataFormat.Float16)]
    if cfg.mathop in SFPI_PARITY_SHIFT_OPS:
        return [InputOutputFormat(DataFormat.Int32, DataFormat.Int32)]
    if cfg.mathop in COMP_OPS:
        return SFPU_UNARY_FORMATS + SFPU_COMP_EXTRA_FORMATS
    if cfg.mathop in {
        MathOperation.Gelu,
        MathOperation.SqrtCustom,
        MathOperation.Log,
        MathOperation.Log1p,
        MathOperation.Clamp,
        MathOperation.Softplus,
        *SFPI_PARITY_NEW_UNARY_OPS,
        *TRIGONOMETRY_OPS,
    }:
        return SFPU_PARITY_FLOAT_FORMATS
    return SFPU_UNARY_FORMATS


def quasar_unpack_to_dest(formats, dest_acc, is_typecast):
    """Whether the input is written straight to Dest via UNPACR_DEST (vs the FPU SrcA→A2D datacopy).

    Typecast routes every 32-bit-Dest case (EITHER endpoint 32-bit) through unpack-to-Dest, because a
    narrow input cannot be FPU-datacopied into a 32-bit Dest (the int datacopy lands all-zeros). Other
    unary ops only use unpack-to-Dest for a 32-bit input with dest_acc=Yes.
    """
    if is_typecast:
        return formats.input_format.is_32_bit() or formats.output_format.is_32_bit()
    return formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


def _typecast_pack_src_format(
    output_format: DataFormat, dest_acc: DestAccumulation
) -> DataFormat:
    """Format the packer must read Dest in for a typecast op.

    The typecast SFPU op writes its OUTPUT format into Dest, so the packer must read Dest in the
    output register format. Format inference derives pack_src from the input side (it assumes the
    dest format equals the unpacked format), which is wrong for a format-converting op: e.g.
    Int32->Float32 infers pack_src=Int32 and Float32->Int32 infers pack_src=Float32, both reading
    the SFPU result in the wrong format. This returns the Dest register form of the output:
     - 32-bit Dest (dest_acc=Yes, a 32-bit endpoint): Int32 for an integer output, Float32
       otherwise; the pack gasket then narrows (e.g. Float32->Float16_b, Int32->UInt8).
     - 16-bit Dest (dest_acc=No, both endpoints <=16-bit): the output sits in Dest in its own format.
    """
    if output_format.is_integer():
        # Integer output: the packer reads the narrow int the SFPU stored, in its own format
        # (NOT a 32-bit container, even in a 32-bit Dest). UInt16 has no Quasar packer encoding,
        # so it is read as Int16 (non-negative values share the bit pattern -> golden matches).
        return DataFormat.Int16 if output_format == DataFormat.UInt16 else output_format
    if dest_acc == DestAccumulation.Yes:
        # Float output in a 32-bit Dest: the value sits as Float32; the pack gasket narrows it
        # to the final output (e.g. Float32 -> Float16_b).
        return DataFormat.Float32
    return output_format


def generate_sfpu_unary_combinations(*, is_perf=False):
    """
    Build the unary-SFPU sweep across all operations and their format matrices.

    Functional mode sweeps dest-sync, implied-math, tensor/tile shape, and the
    L1 formats supported by each operation. Performance mode intentionally keeps
    the complete op, format, dest_acc, and approximation coverage while pinning
    those runtime axes to DestSync.Half, ImpliedMathFormat.Yes, and one 32x32 tile.

    Returns: list of (mathop, fmt, dest_acc, dest_sync, implied_math_format,
    approx_mode, input_dimensions, tile_dimensions) tuples.
    """
    combinations = []

    def dest_acc_modes_for(fmt, is_typecast):
        in_fmt = fmt.input_format
        if (
            in_fmt.is_32_bit()
            or in_fmt == DataFormat.Tf32
            or (is_typecast and fmt.output_format.is_32_bit())
        ):
            return (DestAccumulation.Yes,)
        if is_typecast or in_fmt.is_mx_format():
            return (DestAccumulation.No,)
        return (DestAccumulation.No, DestAccumulation.Yes)

    def append_case(cfg, fmt, dest_acc, dest_sync, implied, approx, tile_case):
        is_typecast = cfg.mathop == MathOperation.Typecast
        if is_invalid_quasar_sfpu_format_combination(
            fmt, dest_acc, quasar_unpack_to_dest(fmt, dest_acc, is_typecast)
        ):
            return
        input_dimensions, tile_dimensions = tile_case
        if (
            fmt.input_format.is_mx_format()
            and tile_dimensions not in MX_SUPPORTED_TILE_SIZES
        ):
            return
        candidate = (
            cfg.mathop,
            fmt,
            dest_acc,
            dest_sync,
            implied,
            approx,
            runtime(input_dimensions),
            runtime(tile_dimensions),
        )
        key = (
            cfg.mathop,
            fmt.input_format,
            fmt.output_format,
            dest_acc,
            dest_sync,
            implied,
            approx,
            input_dimensions,
            tile_dimensions,
        )
        if key not in seen:
            combinations.append(candidate)
            seen.add(key)

    seen = set()
    for cfg in OP_CONFIGS:
        # Ops that expose both a non-approximate and an approximate kernel are swept over both
        # ApproximationMode values; every other op has a single implementation (ApproximationMode.No).
        approx_modes = (
            (ApproximationMode.No, ApproximationMode.Yes)
            if cfg.mathop
            in (
                MathOperation.Exp,
                MathOperation.Gelu,
                MathOperation.Reciprocal,
                MathOperation.Rsqrt,
                MathOperation.Sin,
                MathOperation.Cos,
                MathOperation.Tan,
            )
            else (ApproximationMode.No,)
        )
        if cfg.orthogonal and not is_perf:
            fmts = formats_for_op(cfg)
            baseline = next(
                (
                    fmt
                    for fmt in fmts
                    if fmt.input_format == DataFormat.Float32
                    and fmt.output_format == DataFormat.Float32
                ),
                fmts[0],
            )
            baseline_dest = dest_acc_modes_for(baseline, False)[0]
            baseline_case = ((32, 32), (32, 32))

            # Format axis: every supported input/output pair at one canonical geometry.
            for fmt in fmts:
                dest = dest_acc_modes_for(fmt, False)[0]
                append_case(
                    cfg,
                    fmt,
                    dest,
                    DestSync.Half,
                    ImpliedMathFormat.Yes,
                    ApproximationMode.No,
                    baseline_case,
                )

            # Tile axis: every supported tile shape in both FP32 and BF16, plus 64x64 multi-tile.
            tile_fmts = [baseline]
            tile_fmts.extend(
                fmt
                for fmt in fmts
                if fmt.input_format == DataFormat.Float16_b
                and fmt.output_format == DataFormat.Float16_b
            )
            for fmt in tile_fmts:
                for tile_case in cfg.tile_cases:
                    append_case(
                        cfg,
                        fmt,
                        dest_acc_modes_for(fmt, False)[0],
                        DestSync.Half,
                        ImpliedMathFormat.Yes,
                        ApproximationMode.No,
                        tile_case,
                    )

            # Independent control axes, kept at the baseline format/shape so the
            # matrix does not explode while every relevant value is still exercised.
            for dest in dest_acc_modes_for(baseline, False):
                append_case(
                    cfg,
                    baseline,
                    dest,
                    DestSync.Half,
                    ImpliedMathFormat.Yes,
                    ApproximationMode.No,
                    baseline_case,
                )
            for sync in cfg.dest_sync_modes:
                append_case(
                    cfg,
                    baseline,
                    baseline_dest,
                    sync,
                    ImpliedMathFormat.Yes,
                    ApproximationMode.No,
                    baseline_case,
                )
            for implied in (ImpliedMathFormat.No, ImpliedMathFormat.Yes):
                append_case(
                    cfg,
                    baseline,
                    baseline_dest,
                    DestSync.Half,
                    implied,
                    ApproximationMode.No,
                    baseline_case,
                )
            for approx in approx_modes:
                append_case(
                    cfg,
                    baseline,
                    baseline_dest,
                    DestSync.Half,
                    ImpliedMathFormat.Yes,
                    approx,
                    baseline_case,
                )
            continue

        for fmt in formats_for_op(cfg):
            in_fmt = fmt.input_format

            # Typecast's dest width is determined by the format pair, not swept: a 32-bit
            # endpoint (either side) forces a 32-bit dest, every other pair runs in 16-bit
            # dest. Every other op sweeps both dest_acc modes for non-32-bit inputs.
            is_typecast = cfg.mathop == MathOperation.Typecast
            dest_acc_modes = dest_acc_modes_for(fmt, is_typecast)
            for dest_acc in dest_acc_modes:
                # Skip invalid format combinations for Quasar
                if is_invalid_quasar_sfpu_format_combination(
                    fmt, dest_acc, quasar_unpack_to_dest(fmt, dest_acc, is_typecast)
                ):
                    continue

                dest_sync_modes = (DestSync.Half,) if is_perf else cfg.dest_sync_modes
                implied_math_formats = (
                    (ImpliedMathFormat.Yes,)
                    if is_perf or in_fmt.is_mx_format()
                    else (ImpliedMathFormat.No, ImpliedMathFormat.Yes)
                )
                tile_cases = (((32, 32), (32, 32)),) if is_perf else cfg.tile_cases
                for dest_sync in dest_sync_modes:
                    for implied_math_format in implied_math_formats:
                        for approx_mode in approx_modes:
                            for input_dimensions, tile_dimensions in tile_cases:
                                if (
                                    in_fmt.is_mx_format()
                                    and tile_dimensions not in MX_SUPPORTED_TILE_SIZES
                                ):
                                    continue
                                append_case(
                                    cfg,
                                    fmt,
                                    dest_acc,
                                    dest_sync,
                                    implied_math_format,
                                    approx_mode,
                                    (input_dimensions, tile_dimensions),
                                )

    return combinations


@pytest.mark.quasar
@parametrize(
    mathop_formats_dest_acc_sync_implied_math_dims=generate_sfpu_unary_combinations(),
)
def test_eltwise_unary_sfpu_quasar(
    mathop_formats_dest_acc_sync_implied_math_dims,
    *,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor=1,
    is_perf=False,
    perf_report=None,
):
    """
    Consolidated unary-SFPU test on Quasar. One compile-time-selected op per
    variant (abs, exp, gelu, relu, reciprocal, sqrt, tanh, sigmoid, silu, rsqrt,
    square, typecast, and the six compare-to-zero modes), validated against the
    UnarySFPUGolden reference. Typecast sweeps explicit (src, dst) format pairs;
    every other op sweeps the shared format matrix.
    """
    (
        mathop,
        formats,
        dest_acc,
        dest_sync,
        implied_math_format,
        approx_mode,
        input_dimensions,
        tile_dimensions,
    ) = mathop_formats_dest_acc_sync_implied_math_dims[0]

    is_typecast = mathop == MathOperation.Typecast

    cfg = OP_CONFIG_BY_MATHOP[mathop]
    if mathop in (MathOperation.UnaryEq, MathOperation.UnaryNe):
        # Equality/inequality compare against the kernel's fixed 0.5 threshold.
        # A generic random domain almost never lands on the equal branch, so use
        # exact threshold/straddle values in every face.  These two operations
        # intentionally have no generic sfpu_domains entry for that reason.
        spec = StimuliSpec.custom(values=[-1.0, 0.0, 0.5, 1.0], seed=0)
    elif mathop in SFPI_PARITY_NEW_UNARY_OPS:
        spec = exclude_undefined(
            mathop,
            for_op_pipeline(mathop, formats.input_format, formats.output_format).spec_A,
        )
    else:
        spec = (
            StimuliSpec.uniform(low=0.0, high=1.0)
            if (cfg.uniform_spec and not is_typecast)
            else None
        )
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
        tile_dimensions=tile_dimensions,
    )

    # Prepare inputs with operation-specific ranges
    if is_typecast:
        src_A = _prepare_typecast_input(
            src_A, src_B, formats.input_format, formats.output_format
        )
    else:
        src_A = prepare_unary_inputs(
            mathop, src_A, src_B, formats.input_format, formats.output_format
        )

    tile_shape = construct_tile_shape(tile_dimensions)
    num_faces = tile_shape.total_num_faces()

    if not is_perf:
        if format_dict[formats.input_format].is_floating_point:
            generate_golden = get_golden_generator(UnarySFPUGolden)
            golden_tensor = generate_golden(
                mathop,
                src_A,
                formats.output_format,
                dest_acc,
                formats.input_format,
                input_dimensions,
                skip_tilize=tile_dimensions != (32, 32),
            )
        else:
            # Integer-input ops (Int32/Int16/UInt16 — currently only the comp family): apply the
            # UnarySFPUGolden op element-wise instead of through its __call__. __call__ runs a
            # float-only pipeline (float dst, tilize, FTZ) that would mangle integer values; applying
            # the op per element keeps integers intact, and for an element-wise op row-major order
            # already matches the packed result. A non-element-wise integer op would need its own path.
            ops = UnarySFPUGolden().ops
            op_res = [ops[mathop](x) for x in src_A.flatten().tolist()]
            golden_tensor = torch.tensor(
                op_res, dtype=format_dict[formats.output_format]
            )

    unpack_to_dest = quasar_unpack_to_dest(formats, dest_acc, is_typecast)
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
        "formats": formats,
        "templates": [
            QUASAR_SFPU_UNARY_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
            # Typecast bakes the (input, output) pair so the compile-time functor can pick
            # the right conversion; every other op defaults it. The typecast dispatcher branch
            # in the shared C++ source references TYPECAST_IN_FORMAT/TYPECAST_OUT_FORMAT, so
            # every build must define them.
            (
                TYPECAST_FORMATS(
                    input_format=formats.input_format,
                    output_format=formats.output_format,
                )
                if is_typecast
                else TYPECAST_FORMATS()
            ),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            DEST_INDEX(0),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=tile_dimensions != (32, 32),
        ),
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )

    if is_typecast:
        pack_src_for_output = _typecast_pack_src_format(formats.output_format, dest_acc)
        for fc in configuration.formats_config:
            fc.pack_src = pack_src_for_output
            fc.pack_S_src = pack_src_for_output

    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), "Assert against golden failed"


# Compact emulator gate for the 57-family SFPI parity work.  The exhaustive
# test above remains the source of format/tile/control-axis coverage; this list
# deliberately chooses one legal, full-tile case per conventional parity-unary
# operation so an emulator run can validate every ported dispatch in one pass.
# CastFp32ToFp16a is the sole format exception because its public contract is
# specifically Float32 -> Float16_a.  Accurate GELU is the other intentional
# exception: the FP32-Dest path is what consumes piecewise_rational, while the
# BF16 approximate case consumes the Quasar LUT configuration from issue 51346.
SFPI_PARITY_UNARY_EMULATOR_SMOKE_OPS = (
    *SFPI_PARITY_NEW_UNARY_OPS,
    *SFPI_PARITY_SHIFT_OPS,
    MathOperation.Clamp,
    MathOperation.Log,
    MathOperation.Log1p,
    MathOperation.Softplus,
    MathOperation.SqrtCustom,
    *TRIGONOMETRY_OPS,
)


def _sfpi_parity_unary_emulator_smoke_cases():
    bf16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    fp32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    int32 = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    fp32_to_fp16a = InputOutputFormat(DataFormat.Float32, DataFormat.Float16)
    full_tile = (32, 32)
    cases = []

    assert len(SFPI_PARITY_UNARY_EMULATOR_SMOKE_OPS) == len(
        set(SFPI_PARITY_UNARY_EMULATOR_SMOKE_OPS)
    )
    assert set(SFPI_PARITY_UNARY_EMULATOR_SMOKE_OPS) <= set(OP_CONFIG_BY_MATHOP)

    for operation in SFPI_PARITY_UNARY_EMULATOR_SMOKE_OPS:
        is_fp32_to_fp16a = operation == MathOperation.CastFp32ToFp16a
        is_shift = operation in SFPI_PARITY_SHIFT_OPS
        cases.append(
            (
                operation,
                (fp32_to_fp16a if is_fp32_to_fp16a else int32 if is_shift else bf16),
                (
                    DestAccumulation.Yes
                    if is_fp32_to_fp16a or is_shift
                    else DestAccumulation.No
                ),
                DestSync.Half,
                ImpliedMathFormat.Yes,
                # The accurate I1 instantiation currently exceeds the Quasar
                # compiler's reload budget at -O3.  Approximation mode changes
                # only the reciprocal refinement count and exercises the same
                # ported I1 body without weakening its functional golden.
                (
                    ApproximationMode.Yes
                    if operation == MathOperation.I1
                    else ApproximationMode.No
                ),
                runtime(full_tile),
                runtime(full_tile),
            )
        )

    cases.extend(
        (
            MathOperation.Gelu,
            formats,
            dest_acc,
            DestSync.Half,
            ImpliedMathFormat.Yes,
            approximation_mode,
            runtime(full_tile),
            runtime(full_tile),
        )
        for formats, dest_acc, approximation_mode in (
            # Accurate FP32 calls ckernel_sfpu_piecewise_rational.h.
            (fp32, DestAccumulation.Yes, ApproximationMode.No),
            # Approximate BF16 calls the six-segment Quasar LUT path.
            (bf16, DestAccumulation.No, ApproximationMode.Yes),
        )
    )
    return cases


@pytest.mark.quasar
@parametrize(
    smoke_case=_sfpi_parity_unary_emulator_smoke_cases(),
)
def test_sfpi_parity_unary_emulator_smoke(smoke_case):
    """Run one canonical full-tile emulator case through each parity-unary port."""
    test_eltwise_unary_sfpu_quasar(smoke_case)
