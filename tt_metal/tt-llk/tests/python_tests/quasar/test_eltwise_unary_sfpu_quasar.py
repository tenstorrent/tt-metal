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
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    is_invalid_quasar_sfpu_format_combination,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    StimuliSpec,
    apply_log_uniform_magnitudes,
    compute_safe_input_magnitude_range,
    format_elem_max,
    generate_stimuli,
)
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES
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
        min_val = -5.0
        max_val = 30.0
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
    if mathop in COMP_OPS:
        # Unsigned formats need non-negative stimuli (a signed split would wrap under the unsigned
        # dtype); signed formats use the sign-vs-magnitude builder.
        if input_format in (DataFormat.UInt16, DataFormat.UInt8):
            return prepare_comp_inputs_uint(src_A, src_B, input_format)
        return prepare_comp_inputs(src_A, src_B, input_format, output_format)
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
    input_dims: tuple  # list of [H, W] dimensions
    dest_sync_modes: tuple  # DestSync values to sweep
    uniform_spec: bool = False


TENSOR_DIMS = ([32, 32], [64, 64])
DEST_SYNC_MODES = (DestSync.Half, DestSync.Full)

OP_CONFIGS = [
    OpConfig(MathOperation.Abs, TENSOR_DIMS, DEST_SYNC_MODES),
    OpConfig(MathOperation.Square, TENSOR_DIMS, DEST_SYNC_MODES),
    OpConfig(MathOperation.Rsqrt, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    # Nonlinear ops: identical [32,32]/[64,64] × Half/Full × uniform-spec sweep.
    OpConfig(MathOperation.Exp, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Gelu, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Relu, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Reciprocal, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Sqrt, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Tanh, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Sigmoid, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Silu, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Clamp, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Neg, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Softplus, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Typecast, TENSOR_DIMS, DEST_SYNC_MODES),
] + [OpConfig(op, TENSOR_DIMS, DEST_SYNC_MODES) for op in COMP_OPS]

OP_CONFIG_BY_MATHOP = {cfg.mathop: cfg for cfg in OP_CONFIGS}


def formats_for_op(cfg: OpConfig) -> List[InputOutputFormat]:
    """Float formats for every op, plus the integer/UInt16 formats only comp sweeps."""
    if cfg.mathop == MathOperation.Typecast:
        return [InputOutputFormat(case.src, case.dst) for case in TYPECAST_CASES]
    if cfg.mathop in COMP_OPS:
        return SFPU_UNARY_FORMATS + SFPU_COMP_EXTRA_FORMATS
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


def generate_sfpu_unary_combinations():
    """
    Build the full unary-SFPU sweep across all ops: per op, a
    formats × dest_acc × dest-sync × implied-math × {[32, 32], [64, 64]} matrix.

    Every op runs the same matrix over its own format set (from formats_for_op).
    32-bit inputs always pair with dest_acc=Yes; 16-bit inputs sweep both dest_acc
    modes. Invalid format/dest_acc combinations are dropped via the shared filter.

    Reciprocal, Exp, and Rsqrt each expose both a non-approximate kernel
    (Newton-Raphson refined reciprocal / fp32-accurate exp / SQRT_23-bits rsqrt
    ported from Blackhole) and an approximate one (raw ``approx_recip`` /
    ``approx_exp`` / ``approx_recip(approx_sqrt)`` via the HW LUT), so they are
    swept over both ApproximationMode values; every other op only has a single
    implementation and is swept with ApproximationMode.No.

    Returns: list of (mathop, fmt, dest_acc, dest_sync, implied_math_format,
    approx_mode, input_dimensions) tuples.
    """
    combinations = []
    for cfg in OP_CONFIGS:
        approx_modes = (
            (ApproximationMode.No, ApproximationMode.Yes)
            if cfg.mathop
            in (MathOperation.Reciprocal, MathOperation.Exp, MathOperation.Rsqrt)
            else (ApproximationMode.No,)
        )
        for fmt in formats_for_op(cfg):
            in_fmt = fmt.input_format

            # Typecast's dest width is determined by the format pair, not swept: a 32-bit
            # endpoint (either side) forces a 32-bit dest, every other pair runs in 16-bit
            # dest. Every other op sweeps both dest_acc modes for non-32-bit inputs.
            is_typecast = cfg.mathop == MathOperation.Typecast
            dest_acc_modes = (
                (DestAccumulation.Yes,)
                if in_fmt.is_32_bit() or (is_typecast and fmt.output_format.is_32_bit())
                else (
                    (DestAccumulation.No,)
                    if is_typecast
                    else (DestAccumulation.No, DestAccumulation.Yes)
                )
            )
            for dest_acc in dest_acc_modes:
                # Skip invalid format combinations for Quasar
                if is_invalid_quasar_sfpu_format_combination(
                    fmt, dest_acc, quasar_unpack_to_dest(fmt, dest_acc, is_typecast)
                ):
                    continue

                for dest_sync in cfg.dest_sync_modes:
                    for implied_math_format in [
                        ImpliedMathFormat.No,
                        ImpliedMathFormat.Yes,
                    ]:
                        for approx_mode in approx_modes:
                            for input_dimensions in cfg.input_dims:
                                combinations.append(
                                    (
                                        cfg.mathop,
                                        fmt,
                                        dest_acc,
                                        dest_sync,
                                        implied_math_format,
                                        approx_mode,
                                        runtime(input_dimensions),
                                    )
                                )

    return combinations


@pytest.mark.quasar
@parametrize(
    mathop_formats_dest_acc_sync_implied_math_input_dims=generate_sfpu_unary_combinations(),
)
def test_eltwise_unary_sfpu_quasar(
    mathop_formats_dest_acc_sync_implied_math_input_dims,
):
    """
    Consolidated unary-SFPU test on Quasar. One compile-time-selected op per
    variant (abs, exp, gelu, relu, reciprocal, sqrt, tanh, sigmoid, silu, rsqrt,
    square, clamp, typecast, and the six compare-to-zero modes), validated against
    the UnarySFPUGolden reference. Typecast sweeps explicit (src, dst) format pairs;
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
    ) = mathop_formats_dest_acc_sync_implied_math_input_dims[0]

    is_typecast = mathop == MathOperation.Typecast

    cfg = OP_CONFIG_BY_MATHOP[mathop]
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

    num_faces = MAX_NUM_FACES

    if format_dict[formats.input_format].is_floating_point:
        generate_golden = get_golden_generator(UnarySFPUGolden)
        golden_tensor = generate_golden(
            mathop,
            src_A,
            formats.output_format,
            dest_acc,
            formats.input_format,
            input_dimensions,
        )
    else:
        # Integer-input ops (Int32/Int16/UInt16 — currently only the comp family): apply the
        # UnarySFPUGolden op element-wise instead of through its __call__. __call__ runs a
        # float-only pipeline (float dst, tilize, FTZ) that would mangle integer values; applying
        # the op per element keeps integers intact, and for an element-wise op row-major order
        # already matches the packed result. A non-element-wise integer op would need its own path.
        ops = UnarySFPUGolden().ops
        op_res = [ops[mathop](x) for x in src_A.flatten().tolist()]
        golden_tensor = torch.tensor(op_res, dtype=format_dict[formats.output_format])

    unpack_to_dest = quasar_unpack_to_dest(formats, dest_acc, is_typecast)
    configuration = TestConfig(
        "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
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

    if is_typecast:
        pack_src_for_output = _typecast_pack_src_format(formats.output_format, dest_acc)
        for fc in configuration.formats_config:
            fc.pack_src = pack_src_for_output
            fc.pack_S_src = pack_src_for_output

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
