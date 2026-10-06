# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import struct
from dataclasses import dataclass
from typing import List

import pytest
import torch
from helpers.constraints import is_valid_quasar_fpu_path
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TilizeGolden,
    UnarySFPUGolden,
    UntilizeGolden,
    get_golden_generator,
)
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
    QuasarSfpuVariant,
    generate_quasar_sfpu_format_variants,
    input_output_formats,
    parametrize,
    runtime,
    select_perf_input_dimensions,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.sfpu_dispatch_constants import (
    RELU_MAX_THRESHOLD,
    RELU_MIN_THRESHOLD,
)
from helpers.sfpu_domains import op_edge_points
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
    MATH_OP,
    NUM_FACES,
    RAND_RANGE,
    RAND_SEED,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
)
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

# The trigonometry / inverse-hyperbolic transcendentals. Float-only (they share the
# SFPU_UNARY_FORMATS set), each with its own safe input domain (see prepare_trig_inputs).
TRIGONOMETRY_OPS = [
    MathOperation.Sin,
    MathOperation.Cos,
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

# Ops that sweep the comp format set (float + integer formats) and build on the comp stimuli.
# Signbit is a pure sign-bit test: it differs from an IEEE `< 0` at -0.0 and negative NaN, and from
# Quasar's less_than_zero (itself a bit-31 test with a magnitude-nonzero guard) only at -0.0. The
# comp stimuli seed +/-0.0 at a single datum and keep UInt16 below bit 15, so signbit adds its own
# per-face seeds on top of them (see prepare_signbit_inputs).
COMP_FORMAT_OPS = COMP_OPS + [MathOperation.Signbit]

RELU_CC_OPS = [
    MathOperation.Lrelu,
    MathOperation.ReluMin,
    MathOperation.ReluMax,
]

# Float rounding family. Domain [-10, 10] spans both signs so floor/ceil differ from trunc.
# Exact knees (round-half-to-even ties, integer boundaries) come from op_edge_points().
ROUNDING_OPS = [
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Frac,
    MathOperation.Round,
]

# Extra (integer) formats only COMP_FORMAT_OPS (the comp family + signbit) sweep. Int32/Int16/Int8
# (signed) and UInt8 (unsigned) use their native Quasar dest format. UInt16 is the exception: it has no native Quasar
# dest format, so the inference routes its data path through Int16 and sets FormatConfig.sfpu_src=
# UInt16, the only stage the comp/signbit kernels read as uint16.
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
    elif mathop == MathOperation.Lrelu:
        min_val = -5.0
        max_val = 5.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.ReluMin:
        min_val = -5.0
        max_val = 2.0 * RELU_MIN_THRESHOLD
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.ReluMax:
        min_val = -5.0
        max_val = 2.0 * RELU_MAX_THRESHOLD
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
        min_val = -8.0
        max_val = 30.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop in ROUNDING_OPS:
        # [-10, 10] spans both signs so floor/ceil differ from trunc. Overlay op_edge_points()
        # so ties and integer knees are exact; a uniform draw does not hit them.
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        edges = op_edge_points(mathop)
        if edges:
            flat = src_A.flatten()
            n = min(len(edges), flat.numel())
            flat[:n] = torch.tensor(edges[:n], dtype=flat.dtype)
            src_A = flat.view(src_A.shape)
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
    elif mathop == MathOperation.Asinh:
        lo, hi = -10.0, 10.0
    elif mathop == MathOperation.Acosh:
        lo, hi = 1.1, 50.0
    elif mathop == MathOperation.Atanh:
        lo, hi = -0.9, 0.9
    else:
        return src_A

    return (lo + u * (hi - lo)).to(torch_format)


def prepare_cumsum_inputs(
    src_A: torch.Tensor,
    input_format: DataFormat,
) -> torch.Tensor:
    """
    Map the uniform [0, 1] stimulus into [-1, 1] for the column-wise cumulative sum.

    A column total chains up to 32 adds, so |x| <= 1 keeps every partial sum inside +-32 —
    well inside Float16's range, and clear of the subnormal magnitudes the SFPU MAD
    datapath flushes to zero. Both signs are covered so the chain sees cancellation as
    well as growth.
    """
    u = src_A.to(torch.float32)  # uniform [0, 1] from the uniform stimuli spec
    return (-1.0 + 2.0 * u).to(format_dict[input_format])


# Ops whose result depends on where in the tile a datum sits, so L1 has to hold a real tilized tile.
# Every other op in this suite is element-wise and cannot tell a tilized buffer from a row-major one,
# which is why the suite has always written the latter.
LAYOUT_SENSITIVE_OPS = (MathOperation.Cumsum,)


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
    if mathop == MathOperation.Cumsum:
        return prepare_cumsum_inputs(src_A, input_format)
    if mathop in TRIGONOMETRY_OPS:
        return prepare_trig_inputs(src_A, mathop, input_format)
    if mathop == MathOperation.Signbit:
        return prepare_signbit_inputs(src_A, src_B, input_format, output_format)
    if mathop in COMP_FORMAT_OPS:
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


# Raw bit patterns of a quiet NaN of either sign, per float input format, as (view dtype, +NaN,
# -NaN). Written through a signed integer view, so -NaN is the two's-complement spelling of
# 0xFFC00000 / 0xFFC0 / 0xFE00, and no float conversion can canonicalise the NaN's sign.
_SIGNED_NAN_BITS = {
    DataFormat.Float32: (torch.int32, 0x7FC00000, -0x00400000),
    DataFormat.Float16_b: (torch.int16, 0x7FC0, -0x0040),
    DataFormat.Float16: (torch.int16, 0x7E00, -0x0200),
}

# Elements per 16x16 face. The suite writes row-major data that the hardware reads as tiles, so
# every consecutive FACE_ELEMS-element chunk of the flat input lands in its own face.
FACE_ELEMS = MAX_FACE_R_DIM * FACE_C_DIM


def prepare_signbit_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Comp stimuli plus the inputs on which a sign-bit test differs from a compare.

    Floats: every face gets +0.0, -0.0, +NaN and -NaN, so a compare-based kernel (wrong at -0.0 and
    -NaN) or a datacopy that drops the sign of -0.0 fails in every face, not at one datum. UInt16: the
    comp builder stays below bit 15, so seed 0x8000 / 0xFFFF per face to catch a sign-extending load
    (signbit of an unsigned value is always 0). Other integer formats keep the comp stimuli, which
    already sign about half the lanes across every face.
    """
    if input_format in (DataFormat.UInt16, DataFormat.UInt8):
        values = prepare_comp_inputs_uint(src_A, src_B, input_format)
        if input_format == DataFormat.UInt16:
            flat = values.flatten()
            for base in range(0, flat.numel(), FACE_ELEMS):
                flat[base + 8] = 0x8000
                flat[base + 9] = 0xFFFF
            values = flat.reshape(values.shape)
        return values

    values = prepare_comp_inputs(src_A, src_B, input_format, output_format)
    if input_format not in _SIGNED_NAN_BITS:
        return values

    int_dtype, pos_nan, neg_nan = _SIGNED_NAN_BITS[input_format]
    flat = values.flatten().clone()
    bits = flat.view(int_dtype)
    for base in range(0, flat.numel(), FACE_ELEMS):
        flat[base + 8] = 0.0
        flat[base + 9] = -0.0
        bits[base + 10] = pos_nan
        bits[base + 11] = neg_nan
    return flat.reshape(values.shape)


def _drops_zero_and_nan_sign(mathop: MathOperation, variant: QuasarSfpuVariant) -> bool:
    """
    Whether this signbit route reaches Dest through Quasar's 32-bit-Dest FPU datacopy.

    That datacopy is ELWADD (SrcA + SrcB, SrcB zero; llk_math_eltwise_unary_datacopy.h), and the add
    returns +0.0 for -0.0 + 0.0 and a positive NaN for a negative one, so the sign bit of -0.0 and
    -NaN never reaches the SFPU. Measured on emu-quasar-1x3 for every Float16/Float16_b ELWADD route;
    the 16-bit-Dest MOVA2D route and Unpack-to-Dest keep both signs. Only signbit can see the
    difference (its result is the raw sign bit), so only signbit models it.
    """
    return (
        mathop == MathOperation.Signbit
        and variant.uses_fpu
        and variant.dest_acc == DestAccumulation.Yes
        and format_dict[variant.formats.input_format].is_floating_point
    )


def _elwadd_datacopy_dest_image(src: torch.Tensor) -> torch.Tensor:
    """
    Signbit golden input for an ELWADD route: -0.0 and every NaN reach the SFPU sign-clear.

    Both are spelled +0.0, which has the same sign bit as the positive NaN the datacopy really leaves
    in Dest: the golden's bf16-input / 32-bit-Dest path re-canonicalises a positive NaN to a negative
    one (torch's fp32 -> bf16 cast), which would put the sign bit back.
    """
    return torch.where((src == 0) | torch.isnan(src), torch.zeros_like(src), src)


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
# Int16 (signed 16-bit) is not in the ttnn typecast matrix, but the kernel handles
# it on every path (float<->int16 via SFPCAST + 16-bit store-narrow, int16<->int
# via the int->int path), so it is swept here too. Mirrors the UInt16 set; Int16
# has a native Quasar dest format.
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


TYPECAST_CASES = (
    # float <-> float: widen on store, RNE narrow to fp16
    TypecastCase(DataFormat.Float16_b, DataFormat.Float32),
    TypecastCase(DataFormat.Float32, DataFormat.Float16_b),
    # float <-> int32: SFPCAST (+ fp16 narrow when the dst is fp16)
    TypecastCase(DataFormat.Float16_b, DataFormat.Int32),
    TypecastCase(DataFormat.Int32, DataFormat.Float16_b),
    TypecastCase(DataFormat.Float32, DataFormat.Int32),
    TypecastCase(DataFormat.Int32, DataFormat.Float32),
    # float <-> int16
    TypecastCase(DataFormat.Float16_b, DataFormat.Int16),
    TypecastCase(DataFormat.Int16, DataFormat.Float16_b),
    TypecastCase(DataFormat.Float32, DataFormat.Int16),
    TypecastCase(DataFormat.Int16, DataFormat.Float32),
    # float <-> uint16
    TypecastCase(DataFormat.Float16_b, DataFormat.UInt16),
    TypecastCase(DataFormat.UInt16, DataFormat.Float16_b),
    TypecastCase(DataFormat.Float32, DataFormat.UInt16),
    TypecastCase(DataFormat.UInt16, DataFormat.Float32),
    # float <-> uint8: clamps negatives, then RNE narrows
    TypecastCase(DataFormat.Float16_b, DataFormat.UInt8),
    TypecastCase(DataFormat.UInt8, DataFormat.Float16_b),
    TypecastCase(DataFormat.Float32, DataFormat.UInt8),
    TypecastCase(DataFormat.UInt8, DataFormat.Float32),
    # int <-> int: store sfpmem mode (widen/equal) or RNE narrow to 8-bit
    TypecastCase(DataFormat.UInt16, DataFormat.Int32),
    TypecastCase(DataFormat.Int32, DataFormat.UInt16),
    TypecastCase(DataFormat.UInt16, DataFormat.UInt8),
    TypecastCase(DataFormat.UInt8, DataFormat.UInt16),
    TypecastCase(DataFormat.Int16, DataFormat.Int32),
    TypecastCase(DataFormat.Int32, DataFormat.Int16),
    TypecastCase(DataFormat.Int16, DataFormat.UInt8),
    TypecastCase(DataFormat.UInt8, DataFormat.Int16),
)

_RANGE_SAFETY_FACTOR = 0.9

# Integers around the bf16 ulp=2 boundary (256). 255/256/258 are exact; 257 and 259
# are halfway cases that round-nearest-even must resolve (257→256, 259→260).
_INT32_TO_FP16B_RNE_BOUNDARIES = (255, 256, 257, 258, 259)


def _is_int32_to_fp16b(src_format: DataFormat, dst_format: DataFormat) -> bool:
    return src_format == DataFormat.Int32 and dst_format == DataFormat.Float16_b


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
        result = vals.round().to(format_dict[src_format])

        # Int32 → Float16_b: plant bf16 spacing-boundary integers (and their negatives)
        # so FP32_TO_FP16B nearest-even is actually exercised. The random band above
        # stays in [-200, 200], which is integer-exact in bf16 and would not catch a
        # missing or wrong rounding step.
        if _is_int32_to_fp16b(src_format, dst_format):
            flat = result.flatten()
            seeds = list(_INT32_TO_FP16B_RNE_BOUNDARIES) + [
                -v for v in _INT32_TO_FP16B_RNE_BOUNDARIES
            ]
            for i, seed in enumerate(seeds):
                if i < flat.numel():
                    flat[i] = seed
            result = flat.reshape(result.shape)
        return result

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
    *[
        OpConfig(op, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True)
        for op in RELU_CC_OPS
    ],
    OpConfig(MathOperation.Reciprocal, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Sqrt, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Tanh, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Sigmoid, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Silu, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Clamp, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Neg, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Softplus, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    # Column-wise cumulative sum: a whole-tile op (VectorMode::RC_custom, one call per
    # tile) whose running total lives in LREG4-7 between calls. Every tile is swept with
    # first=true, which zeroes that carry, so tiles are independent; covering the
    # cross-tile carry (first=false) needs the shared C++ source to thread
    # `first = (i == 0)` through its tile loop, so it is a follow-on.
    OpConfig(MathOperation.Cumsum, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True),
    OpConfig(MathOperation.Typecast, TENSOR_DIMS, DEST_SYNC_MODES),
    # Trigonometry / inverse-hyperbolic ops: same matrix as the other transcendentals,
    # fed a uniform [0, 1] stimulus that prepare_trig_inputs maps into each op's domain.
    *[
        OpConfig(op, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True)
        for op in TRIGONOMETRY_OPS
    ],
    *[
        OpConfig(op, TENSOR_DIMS, DEST_SYNC_MODES, uniform_spec=True)
        for op in ROUNDING_OPS
    ],
] + [OpConfig(op, TENSOR_DIMS, DEST_SYNC_MODES) for op in COMP_FORMAT_OPS]

OP_CONFIG_BY_MATHOP = {cfg.mathop: cfg for cfg in OP_CONFIGS}

# Float formats whose 16-bit Dest route gets an explicit MOVA2D (FPU datacopy) signbit variant.
SIGNBIT_MOVA2D_FORMATS = (DataFormat.Float16, DataFormat.Float16_b)


def signbit_fpu_route_variants() -> List[QuasarSfpuVariant]:
    """
    Signbit variants that reach Dest through the FPU datacopy instead of Unpack-to-Dest.

    The default dedup keeps Unpack-to-Dest for every signbit state, so -0.0 (and -NaN) never cross the
    FPU datacopy, the route a default copy_tile takes and the one that flushed -0.0 on WH/BH. This adds
    every FPU route of the full format-route sweep (ELWADD into a 32-bit Dest) plus a MOVA2D route into
    a 16-bit Dest, which the resolver never picks because a direct unpack is always valid there. The
    ELWADD routes do drop the sign of -0.0 and -NaN; see _drops_zero_and_nan_sign.
    """
    variants = [
        variant
        for variant in generate_quasar_sfpu_format_variants(
            MathOperation.Signbit,
            formats_for_op(OP_CONFIG_BY_MATHOP[MathOperation.Signbit]),
            full_format_route_sweep=True,
        )
        if variant.uses_fpu
    ]
    for fmt in SIGNBIT_MOVA2D_FORMATS:
        assert is_valid_quasar_fpu_path(fmt, fmt, fmt, DestAccumulation.No)
        variants.append(
            QuasarSfpuVariant(
                formats=InputOutputFormat(fmt, fmt),
                dest_acc=DestAccumulation.No,
                unpack_to_dest=False,
                unpack_dst=fmt,
                sfpu_src=fmt,
                sfpu_dst=fmt,
                pack_src=fmt,
            )
        )
    return variants


def formats_for_op(cfg: OpConfig) -> List[InputOutputFormat]:
    """Float formats for every op, plus the integer/UInt16 formats only COMP_FORMAT_OPS sweep."""
    if cfg.mathop == MathOperation.Typecast:
        return [InputOutputFormat(case.src, case.dst) for case in TYPECAST_CASES]
    if cfg.mathop in COMP_FORMAT_OPS:
        return SFPU_UNARY_FORMATS + SFPU_COMP_EXTRA_FORMATS
    return SFPU_UNARY_FORMATS


def generate_sfpu_unary_combinations(*, is_perf=False):
    """
    Build the unary-SFPU sweep across all operations and their format matrices.

    Functional mode sweeps dest-sync, implied-math, and both [32, 32]/[64, 64]
    dimensions. Performance mode intentionally keeps the complete op, format,
    dest_acc, and approximation coverage while pinning those three axes to
    DestSync.Half, ImpliedMathFormat.Yes, and the largest functional matrix
    because none of the preferred perf matrices are supported.

    Returns: list of (mathop, resolved format variant, dest_sync,
    implied_math_format, approx_mode, input_dimensions) tuples.
    """
    combinations = []
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
            )
            else (ApproximationMode.No,)
        )
        format_variants = [
            (variant, False)
            for variant in generate_quasar_sfpu_format_variants(
                cfg.mathop, formats_for_op(cfg)
            )
        ]
        if cfg.mathop == MathOperation.Signbit and not is_perf:
            # The extra FPU routes only check the datacopy keeps the sign bit, so they run at one
            # dest-sync / size point (still both implied-math modes) instead of the full matrix.
            format_variants += [(v, True) for v in signbit_fpu_route_variants()]
        for variant, route_only in format_variants:
            dest_sync_modes = (
                (DestSync.Half,) if (is_perf or route_only) else cfg.dest_sync_modes
            )
            if cfg.mathop == MathOperation.Typecast:
                implied_math_formats = (ImpliedMathFormat.No,)
            elif is_perf:
                implied_math_formats = (ImpliedMathFormat.Yes,)
            else:
                implied_math_formats = (ImpliedMathFormat.No, ImpliedMathFormat.Yes)
            if is_perf:
                input_dims = select_perf_input_dimensions(cfg.input_dims)
            elif route_only:
                input_dims = cfg.input_dims[:1]
            else:
                input_dims = cfg.input_dims
            for dest_sync in dest_sync_modes:
                for implied_math_format in implied_math_formats:
                    for approx_mode in approx_modes:
                        for input_dimensions in input_dims:
                            combinations.append(
                                (
                                    cfg.mathop,
                                    variant,
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
    *,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor=1,
    is_perf=False,
    perf_report=None,
):
    """
    Consolidated unary-SFPU test on Quasar. One compile-time-selected op per
    variant (abs, exp, gelu, relu, lrelu, relu_min, relu_max, reciprocal, sqrt,
    tanh, sigmoid, silu, rsqrt, square, cumsum, typecast,
    floor/ceil/trunc/frac/round, the six
    compare-to-zero modes, and signbit), validated against the UnarySFPUGolden reference.
    Typecast sweeps explicit (src, dst) format pairs; every other op sweeps the
    shared format matrix.
    """
    (
        mathop,
        format_variant,
        dest_sync,
        implied_math_format,
        approx_mode,
        input_dimensions,
    ) = mathop_formats_dest_acc_sync_implied_math_input_dims[0]

    assert isinstance(format_variant, QuasarSfpuVariant)
    formats = format_variant.formats
    dest_acc = format_variant.dest_acc
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

    if not is_perf:
        if format_dict[formats.input_format].is_floating_point:
            generate_golden = get_golden_generator(UnarySFPUGolden)
            golden_tensor = generate_golden(
                mathop,
                (
                    _elwadd_datacopy_dest_image(src_A)
                    if _drops_zero_and_nan_sign(mathop, format_variant)
                    else src_A
                ),
                formats.output_format,
                dest_acc,
                formats.input_format,
                input_dimensions,
            )
        else:
            # Integer-input ops (Int32/Int16/UInt16 — currently COMP_FORMAT_OPS): apply the
            # UnarySFPUGolden op element-wise instead of through its __call__. __call__ runs a
            # float-only pipeline (float dst, tilize, FTZ) that would mangle integer values; applying
            # the op per element keeps integers intact, and for an element-wise op row-major order
            # already matches the packed result. A non-element-wise integer op would need its own path.
            if _is_int32_to_fp16b(formats.input_format, formats.output_format):
                # Explicit fp32 → bf16 RNE so planted 257/259 (and negatives) check the
                # kernel's FP32_TO_FP16B nearest-even step, not an identity bit copy.
                golden_tensor = src_A.to(torch.float32).to(torch.bfloat16)
            else:
                ops = UnarySFPUGolden().ops
                op_res = [ops[mathop](x) for x in src_A.flatten().tolist()]
                golden_tensor = torch.tensor(
                    op_res, dtype=format_dict[formats.output_format]
                )

    # A layout-sensitive op reads the tile's face structure, so it gets the tilized buffer tt-metal
    # would feed it, and its result is read back through the matching untilize. UnarySFPUGolden
    # already models the logical -> Dest -> logical round trip, so the golden above stays on the
    # logical tensor and only what crosses to L1 and back is converted.
    is_layout_sensitive = mathop in LAYOUT_SENSITIVE_OPS
    device_src_A = (
        get_golden_generator(TilizeGolden)(
            src_A, input_dimensions, formats.input_format
        )
        if is_layout_sensitive
        else src_A
    )

    unpack_to_dest = format_variant.unpack_to_dest
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
        "formats": formats,
        "templates": [
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
                    input_format=format_variant.sfpu_src,
                    output_format=format_variant.sfpu_dst,
                )
                if is_typecast
                else TYPECAST_FORMATS()
            ),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            device_src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            # Unpack-to-Dest copies Int32 L1 as two's-complement. Only Int32 → Float16_b
            # converts 2SC → SM in the kernel; other integer typecasts still pack SM.
            twos_complement=_is_int32_to_fp16b(
                formats.input_format, formats.output_format
            ),
        ),
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )

    format_variant.apply_formats(configuration.formats_config)

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

    if is_layout_sensitive:
        res_tensor = get_golden_generator(UntilizeGolden)(
            res_tensor, formats.output_format, input_dimensions
        )

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), "Assert against golden failed"


# ---------------------------------------------------------------------------
# Cumsum layout / face-boundary detector.
#
# The random sweep above proves cumsum against UnarySFPUGolden; this case proves the two
# properties that golden cannot name on its own, against an oracle that is exact rather
# than tolerance-bounded:
#   1. the kernel addresses Dest as the TILE-layout tile `copy_tile` puts there, not as an
#      untilized row-major buffer, and
#   2. the running total crosses the face-pair boundary (tile row 15 -> 16) intact.
# Pinned to the 16-bit float formats on purpose: a Float32 input would be written straight
# to Dest by UNPACR_DEST, bypassing the SrcA -> A2D datacopy that is `copy_tile`.
# ---------------------------------------------------------------------------
CUMSUM_DETECTOR_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16, DataFormat.Float16),
]

# The tile row the accumulation chain crosses from face pair 0/1 into face pair 2/3.
CUMSUM_FACE_BOUNDARY_ROW = MAX_FACE_R_DIM


def _cumsum_detector_stimulus() -> torch.Tensor:
    """
    A one-tile stimulus of small integers that pins every axis the kernel could get wrong.

    ``in(r, c) = 1 + (c % 2) + 3 * (r % 2)`` — values 1, 2, 4, 5:
      - it varies with ``r``, so the tilized L1 tile differs element-for-element from the
        row-major one the rest of this suite writes; an untilized-Dest kernel reads the
        wrong rows and cannot produce the expected column sums;
      - it varies with ``c``'s parity, so the even-column and odd-column halves of each
        SFPLOAD carry different data and a swapped parity shows up immediately;
      - every partial sum is an integer <= 112, so every comparison against the oracle is
        exact in Float16, Float16_b and Float32 alike.
    """
    rows = torch.arange(DEFAULT_TILE_R_DIM).unsqueeze(1)
    cols = torch.arange(DEFAULT_TILE_C_DIM).unsqueeze(0)
    return (1 + (cols % 2) + 3 * (rows % 2)).to(torch.float32)


@pytest.mark.quasar
@parametrize(
    cumsum_formats_dest_acc=[
        (fmt, dest_acc)
        for fmt in CUMSUM_DETECTOR_FORMATS
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
    ],
)
def test_cumsum_tilized_dest_quasar(cumsum_formats_dest_acc):
    """
    Deterministic proof that Quasar cumsum walks the tilized Dest layout and carries its
    running total across the face-pair boundary.

    The stimulus reaches Dest exactly as TTNN delivers it: a Layout::TILE tile in L1,
    unpacked to SrcA and datacopied into Dest by A2D — the LLK decomposition of
    `copy_tile`. The result is packed back in tile layout and untilized before comparison,
    so both the input and the output permutation are load-bearing: an untilized row-major
    reading of Dest cannot satisfy this oracle.
    """
    (formats, dest_acc) = cumsum_formats_dest_acc[0]

    input_dimensions = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    src_A = _cumsum_detector_stimulus().to(format_dict[formats.input_format])
    # src_B is unused by a unary op, but StimuliConfig requires an operand-B buffer.
    src_B = torch.zeros_like(src_A)

    # The oracle: column-wise prefix sum of the logical tile, accumulated in float32 the
    # way the SFPU accumulates in an FP32 LREG. Every value is a small exact integer, so
    # the Dest-format rounding the hardware applies on each store is a no-op here.
    logical_src_A = src_A.to(torch.float32)
    expected = torch.cumsum(logical_src_A, dim=0)

    # Layout::TILE in L1 — the discriminator. Feeding the row-major tensor instead would
    # make an untilized-Dest kernel pass.
    device_src_A = get_golden_generator(TilizeGolden)(
        src_A, input_dimensions, formats.input_format
    )

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs={
            "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
            "formats": formats,
            "templates": [
                MATH_OP(mathop=MathOperation.Cumsum),
                APPROX_MODE(ApproximationMode.No),
                IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
                DEST_SYNC(DestSync.Half),
                TYPECAST_FORMATS(),
            ],
            "runtimes": [
                TILE_COUNT(1),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                device_src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=1,
                tile_count_B=1,
                tile_count_res=1,
                num_faces=MAX_NUM_FACES,
            ),
            "unpack_to_dest": False,
            "dest_acc": dest_acc,
        },
    )

    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = get_golden_generator(UntilizeGolden)(
        res_tensor, formats.output_format, input_dimensions
    )
    result = res_tensor.to(torch.float32).reshape(
        DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM
    )

    # Row 0 must be the raw input: the carry bank was zeroed by first == true.
    assert torch.allclose(
        result[0], expected[0], rtol=0, atol=1e-3
    ), f"row 0 is not the input row — carry bank not zeroed: got {result[0].tolist()}"

    # The face-pair boundary, asserted on its own so a failure names the boundary instead
    # of showing a wall of mismatches. out(16) - out(15) == in(16) is exactly the statement
    # that the total through tile row 15 (face pair 0/1) reached tile row 16 (face pair 2/3).
    boundary = CUMSUM_FACE_BOUNDARY_ROW
    assert torch.allclose(
        result[boundary - 1], expected[boundary - 1], rtol=0, atol=1e-3
    ), (
        f"last row of the first face pair is wrong: got {result[boundary - 1].tolist()}, "
        f"expected {expected[boundary - 1].tolist()}"
    )
    assert torch.allclose(
        result[boundary] - result[boundary - 1],
        logical_src_A[boundary],
        rtol=0,
        atol=1e-3,
    ), (
        "cumulative total was not carried across the face-pair boundary "
        f"(tile row {boundary - 1} -> {boundary}): row {boundary} = "
        f"{result[boundary].tolist()}, row {boundary - 1} = {result[boundary - 1].tolist()}"
    )

    # Every logical tile row, which also rejects any chaining that skips or reorders rows.
    assert torch.allclose(
        result, expected, rtol=0, atol=1e-3
    ), f"cumsum mismatch:\ngot\n{result}\nexpected\n{expected}"


# ---------------------------------------------------------------------------
# Float32 -> UInt16 negative-clamp and rounding detector.
#
# The Typecast sweep above feeds this pair through _prepare_typecast_input, which sets
# lo = 0.0 whenever an endpoint is unsigned and rounds every value to a whole number. So
# the sweep only ever supplies non-negative integers.

# The sweep cannot simply be widened, because TypecastGolden models float -> int with
# torch.trunc (it says so: whole-number stimuli make trunc == round). Fractional stimuli
# would make the golden disagree with correct hardware. This test therefore carries its
# own exact oracle instead, like test_cumsum_tilized_dest_quasar above.
#

# (input fp32, expected uint16) pairs, grouped by the kernel property each one pins.
_TYPECAST_NEGATIVE_CASES = (
    (-0.5, 0),
    (-2.5, 0),
    (-65535.0, 0),
)

_TYPECAST_FRACTIONAL_CASES = (
    (0.25, 0),
    (1.75, 2),
    (100.6, 101),
)

# The discriminator: these separate round-nearest-even from round-half-away-from-zero
# (which would give 1, 2, 3, 4, 5, 101, 102) and from truncation (0, 1, 2, 3, 4, 100, 101).
_TYPECAST_TIE_CASES = (
    (0.5, 0),
    (3.5, 4),
    (100.5, 100),
)

_TYPECAST_EXACT_CASES = (
    (1.0, 1),
    (65535.0, 65535),
)

# Inputs beyond the UInt16 range saturate to 65535 rather than wrapping modulo 65536, matching
# TypecastGolden and ttnn.typecast, both of which clamp UInt16 results.
_TYPECAST_SATURATION_CASES = (
    (65535.5, 65535),
    (65537.0, 65535),
    (1000000000.0, 65535),
)

_TYPECAST_EDGE_GROUPS = (
    ("negative clamp", _TYPECAST_NEGATIVE_CASES),
    ("fractional rounding", _TYPECAST_FRACTIONAL_CASES),
    ("round-nearest-even ties", _TYPECAST_TIE_CASES),
    ("exact values", _TYPECAST_EXACT_CASES),
    ("upper saturation", _TYPECAST_SATURATION_CASES),
)

_TYPECAST_EDGE_CASES = tuple(
    case for _, group in _TYPECAST_EDGE_GROUPS for case in group
)


def _typecast_edge_case_tile() -> tuple:
    """Tile the edge cases over a full 32x32 tile, plus the matching expected tensor.

    The case list is repeated rather than front-loaded and padded: 23 cases into 1024
    elements is coprime with the 16-wide face row, so each case lands on a different SFPU
    lane, column parity and face on successive repeats. A clamp that is wrong on only some
    lanes (the failure mode a mis-set CC enable produces) survives a front-loaded stimulus.
    """
    element_count = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM
    repeats = math.ceil(element_count / len(_TYPECAST_EDGE_CASES))

    inputs = [value for value, _ in _TYPECAST_EDGE_CASES] * repeats
    expected = [result for _, result in _TYPECAST_EDGE_CASES] * repeats

    return (
        torch.tensor(inputs[:element_count], dtype=torch.float32),
        torch.tensor(expected[:element_count], dtype=torch.int64),
    )


@pytest.mark.quasar
@parametrize(dest_sync=[DestSync.Half, DestSync.Full])
def test_typecast_fp32_to_uint16_edge_cases_quasar(dest_sync):
    """
    Deterministic proof that Float32 -> UInt16 clamps negatives to 0, rounds nearest-even and
    saturates above 65535, none of which the randomised Typecast sweep can observe.
    """
    dest_sync = dest_sync[0]

    formats = InputOutputFormat(DataFormat.Float32, DataFormat.UInt16)
    variants = generate_quasar_sfpu_format_variants(MathOperation.Typecast, [formats])
    assert len(variants) == 1, (
        "expected exactly one Float32 -> UInt16 Quasar variant, got "
        f"{len(variants)}: {variants}"
    )
    variant = variants[0]

    input_dimensions = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    src_A, expected_flat = _typecast_edge_case_tile()
    # src_B is unused by a unary op, but StimuliConfig requires an operand-B buffer.
    src_B = torch.zeros_like(src_A)

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs={
            "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
            "formats": formats,
            "templates": [
                MATH_OP(mathop=MathOperation.Typecast),
                APPROX_MODE(ApproximationMode.No),
                # Typecast names every Dest access format explicitly, so the implied
                # math format must be off — same as the sweep.
                IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(
                    UnpackerEngine.UnpDest
                    if variant.unpack_to_dest
                    else UnpackerEngine.UnpA
                ),
                DEST_SYNC(dest_sync),
                TYPECAST_FORMATS(
                    input_format=variant.sfpu_src,
                    output_format=variant.sfpu_dst,
                ),
            ],
            "runtimes": [
                TILE_COUNT(1),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=1,
                tile_count_B=1,
                tile_count_res=1,
                num_faces=MAX_NUM_FACES,
            ),
            "unpack_to_dest": variant.unpack_to_dest,
            "dest_acc": variant.dest_acc,
        },
    )

    variant.apply_formats(configuration.formats_config)

    res_from_L1 = configuration.run().result

    # Typecast is element-wise (not in LAYOUT_SENSITIVE_OPS), so the packed result is read
    # back in the same row-major order the stimulus was written in — no untilize needed.
    result = torch.tensor(res_from_L1, dtype=torch.int64)
    assert (
        result.numel() == expected_flat.numel()
    ), f"result has {result.numel()} elements, expected {expected_flat.numel()}"

    period = len(_TYPECAST_EDGE_CASES)
    case_index = 0
    for group_name, group in _TYPECAST_EDGE_GROUPS:
        for value, want in group:
            # Every repeat of this case across the tile, so a lane-dependent failure shows.
            got = result[case_index::period]
            mismatched = (got != want).nonzero().flatten()
            assert mismatched.numel() == 0, (
                f"{group_name}: input {value} should convert to {want}, got "
                f"{got[mismatched[:8]].tolist()} at {mismatched.numel()} of "
                f"{got.numel()} tile positions"
            )
            case_index += 1

    assert torch.equal(result, expected_flat), (
        "Float32 -> UInt16 mismatch:\n"
        f"got      {result[:32].tolist()}\n"
        f"expected {expected_flat[:32].tolist()}"
    )


# ---------------------------------------------------------------------------
# Rand: a property-based check.
#
# rand overwrites Dest with draws from the per-lane hardware PRNG, so no element-wise golden
# exists. The oracle checks the properties the op promises instead: every value lies in
# [from, from + scale], the sample is uniform (mean, spread, histogram), every lane and every
# row gets its own draw, and scale == 0 yields a constant tile. The input tile is filled with
# a value outside every tested interval, so an element the kernel failed to write fails the
# range check. test_rand_seed_quasar checks the seed itself: same seed, same tile; another
# seed, another tile; and the all-ones lock-up seed is repaired.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class RandCase:
    name: str
    from_value: float
    scale: float
    formats: tuple  # DataFormats whose Dest can represent [from, from + scale]

    def __repr__(self) -> str:
        return self.name


_RAND_FLOAT_FORMATS = (DataFormat.Float16, DataFormat.Float32, DataFormat.Float16_b)

# A scale whose fp32 exponent is <= 31 cannot absorb the 2^-31 normalization, so the kernel
# replays its body with a per-row SFPMULI instead of the folded one.
_RAND_PER_ROW_NORMALIZE_SCALE = 2.0**-96

RAND_CASES = (
    RandCase("unit", 1.0, 2.0, _RAND_FLOAT_FORMATS),
    RandCase("signed", -4.0, 8.0, _RAND_FLOAT_FORMATS),
    # Float16 cannot represent 2^-96, so the per-row-normalize case runs on the 8-bit-exponent formats.
    RandCase(
        "per_row_normalize",
        0.0,
        _RAND_PER_ROW_NORMALIZE_SCALE,
        (DataFormat.Float32, DataFormat.Float16_b),
    ),
    RandCase("zero_scale", 1.5, 0.0, _RAND_FLOAT_FORMATS),
)

# rand does not depend on Dest sync, and 4 tiles fit one Dest section in every mode, so every
# case runs one shape and one sync mode.
_RAND_DEST_SYNC = DestSync.Half
_RAND_DIMS = [64, 64]

# Written to Dest before rand runs; outside every RandCase interval.
_RAND_UNWRITTEN_SENTINEL = -100.0

_RAND_DEFAULT_SEED = 0x12345678

# Uniformity thresholds.
_RAND_MEAN_SIGMAS = 6.0  # mean within this many standard errors of the midpoint
_RAND_STD_REL_TOL = (
    0.15  # sample standard deviation within this fraction of the uniform one
)
_RAND_HISTOGRAM_BINS = 8
_RAND_HISTOGRAM_MIN_FRACTION = 0.5  # every bin holds [0.5, 1.5] x its expected count
_RAND_HISTOGRAM_MAX_FRACTION = 1.5
_RAND_FP32_MIN_DISTINCT_FRACTION = 0.95  # Float32 output: nearly every draw distinct
_RAND_NARROW_MIN_DISTINCT = 64  # 16-bit output: at least this many distinct values
_RAND_ROW_MIN_DISTINCT_DIVISOR = 2  # a face row holds >= FACE_C_DIM / 2 distinct values
_RAND_COL_MIN_DISTINCT_DIVISOR = (
    4  # a face column holds >= face rows / 4 distinct values
)
_RAND_MAX_SERIAL_CORRELATION = 0.2


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def generate_rand_combinations():
    """Every RandCase over its formats, at one Dest sync mode and shape."""
    combinations = []
    for case in RAND_CASES:
        for variant in generate_quasar_sfpu_format_variants(
            MathOperation.Rand, input_output_formats(list(case.formats))
        ):
            combinations.append((case, variant, _RAND_DEST_SYNC, runtime(_RAND_DIMS)))
    return combinations


def _rand_output_ulp(magnitude: float, output_format: DataFormat) -> float:
    mantissa_bits = {
        DataFormat.Float16_b: 7,
        DataFormat.Float16: 10,
        DataFormat.Float32: 23,
    }[output_format]
    return (
        math.ldexp(1.0, math.frexp(magnitude)[1] - 1 - mantissa_bits)
        if magnitude
        else 0.0
    )


def _serial_correlation(sequence: torch.Tensor) -> float:
    a = sequence[:-1] - sequence[:-1].mean()
    b = sequence[1:] - sequence[1:].mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))


def _check_rand_properties(values: torch.Tensor, case: RandCase, output_format):
    """Assert the uniform-distribution properties of a flat, tile-ordered rand result."""
    n = values.numel()
    lo = case.from_value
    hi = case.from_value + case.scale

    assert torch.isfinite(values).all(), "rand produced a non-finite value"

    out_of_range = ((values < lo) | (values > hi)).nonzero().flatten()
    assert out_of_range.numel() == 0, (
        f"{out_of_range.numel()} of {n} values fall outside [{lo}, {hi}]; first: "
        f"{values[out_of_range[:8]].tolist()} at {out_of_range[:8].tolist()}"
    )

    if case.scale == 0.0:
        assert torch.all(values == lo), (
            f"scale == 0 must give a constant {lo} tile, got "
            f"{torch.unique(values)[:8].tolist()}"
        )
        return

    # Mean within _RAND_MEAN_SIGMAS standard errors of the midpoint, plus one output ulp for the
    # truncating 16-bit store.
    sigma = case.scale / math.sqrt(12.0)
    mean = float(values.mean())
    mean_tol = _RAND_MEAN_SIGMAS * sigma / math.sqrt(n) + _rand_output_ulp(
        max(abs(lo), abs(hi)), output_format
    )
    assert (
        abs(mean - (lo + hi) / 2) <= mean_tol
    ), f"mean {mean} is not within {mean_tol} of the midpoint {(lo + hi) / 2}"

    std = float(values.std())
    assert (
        abs(std - sigma) <= _RAND_STD_REL_TOL * sigma
    ), f"standard deviation {std} is not within {_RAND_STD_REL_TOL:.0%} of the uniform {sigma}"

    counts = torch.histc(values, bins=_RAND_HISTOGRAM_BINS, min=lo, max=hi)
    expected = n / _RAND_HISTOGRAM_BINS
    assert torch.all(
        (counts >= _RAND_HISTOGRAM_MIN_FRACTION * expected)
        & (counts <= _RAND_HISTOGRAM_MAX_FRACTION * expected)
    ), f"histogram over [{lo}, {hi}] is not uniform: {counts.tolist()} (expected ~{expected})"

    distinct = torch.unique(values).numel()
    min_distinct = (
        int(_RAND_FP32_MIN_DISTINCT_FRACTION * n)
        if output_format == DataFormat.Float32
        else _RAND_NARROW_MIN_DISTINCT
    )
    assert (
        distinct >= min_distinct
    ), f"only {distinct} distinct values in {n} draws (need >= {min_distinct})"

    # One PRNG step covers a row pair: the 32 SFPU lanes span 2 face rows x FACE_C_DIM columns,
    # and the kernel steps Dest by 2 rows. So a face row's 16 values are 16 different lanes, and
    # a face column's values come from 2 lanes over successive PRNG steps.
    face_rows = values.reshape(-1, FACE_C_DIM)
    row_distinct = torch.tensor([torch.unique(r).numel() for r in face_rows])
    row_min = FACE_C_DIM // _RAND_ROW_MIN_DISTINCT_DIVISOR
    assert torch.all(row_distinct >= row_min), (
        f"lanes are not independent: a face row holds only {int(row_distinct.min())} "
        f"distinct values of {FACE_C_DIM}"
    )
    col_distinct = torch.tensor([torch.unique(c).numel() for c in face_rows.T])
    col_min = face_rows.shape[0] // _RAND_COL_MIN_DISTINCT_DIVISOR
    assert torch.all(col_distinct >= col_min), (
        f"the PRNG does not advance per row pair: a face column holds only "
        f"{int(col_distinct.min())} distinct values over {face_rows.shape[0]} rows"
    )
    distinct_rows = torch.unique(face_rows, dim=0).shape[0]
    assert (
        distinct_rows == face_rows.shape[0]
    ), f"{face_rows.shape[0] - distinct_rows} face rows repeat an earlier row"

    along_row_corr = _serial_correlation(face_rows.flatten())
    along_col_corr = _serial_correlation(face_rows.T.flatten())
    assert (
        abs(along_row_corr) < _RAND_MAX_SERIAL_CORRELATION
        and abs(along_col_corr) < _RAND_MAX_SERIAL_CORRELATION
    ), (
        f"adjacent draws are correlated: along a row {along_row_corr:.3f}, "
        f"along a column {along_col_corr:.3f}"
    )


def _rand_config(
    case: RandCase,
    format_variant: QuasarSfpuVariant,
    dest_sync: DestSync,
    input_dimensions,
    seed: int = _RAND_DEFAULT_SEED,
):
    """Build (without running) the eltwise-unary rand configuration for one case and seed."""
    formats = format_variant.formats
    input_torch_format = format_dict[formats.input_format]
    element_count = input_dimensions[0] * input_dimensions[1]
    tile_count = element_count // (DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM)
    src_A = torch.full(
        (element_count,), _RAND_UNWRITTEN_SENTINEL, dtype=input_torch_format
    )
    # src_B is unused by a unary op, but StimuliConfig requires an operand-B buffer.
    src_B = torch.zeros_like(src_A)

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs={
            "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
            "formats": formats,
            "templates": [
                MATH_OP(mathop=MathOperation.Rand),
                APPROX_MODE(ApproximationMode.No),
                IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(
                    UnpackerEngine.UnpDest
                    if format_variant.unpack_to_dest
                    else UnpackerEngine.UnpA
                ),
                DEST_SYNC(dest_sync),
                TYPECAST_FORMATS(),
                RAND_RANGE(
                    rand_from_bits=_fp32_bits(case.from_value),
                    rand_scale_bits=_fp32_bits(case.scale),
                ),
                RAND_SEED(rand_seed=seed),
            ],
            "runtimes": [
                TILE_COUNT(tile_count),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_count,
                tile_count_B=tile_count,
                tile_count_res=tile_count,
                num_faces=MAX_NUM_FACES,
            ),
            "unpack_to_dest": format_variant.unpack_to_dest,
            "dest_acc": format_variant.dest_acc,
        },
    )
    format_variant.apply_formats(configuration.formats_config)
    return configuration


def _run_rand(configuration, format_variant: QuasarSfpuVariant, element_count: int):
    res_from_L1 = configuration.run().result
    values = torch.tensor(
        res_from_L1, dtype=format_dict[format_variant.formats.output_format]
    ).to(torch.float64)
    assert (
        values.numel() == element_count
    ), f"result has {values.numel()} elements, expected {element_count}"
    return values


@pytest.mark.quasar
@parametrize(rand_case_formats_sync_dims=generate_rand_combinations())
def test_rand_quasar(rand_case_formats_sync_dims):
    """Property-based check of the Quasar rand SFPU op over both replayed bodies and scale == 0."""
    case, format_variant, dest_sync, input_dimensions = rand_case_formats_sync_dims[0]
    configuration = _rand_config(case, format_variant, dest_sync, input_dimensions)
    values = _run_rand(
        configuration, format_variant, input_dimensions[0] * input_dimensions[1]
    )
    _check_rand_properties(values, case, format_variant.formats.output_format)


_RAND_OTHER_SEED = 0x9E3779B9
_RAND_LOCKUP_SEED = 0xFFFFFFFF  # the XNOR LFSR lock-up state init_rand repairs
_RAND_LOCKUP_REPAIR_SEED = 0xFFFFFFFE  # what init_rand replaces it with
_RAND_MIN_DIFFERENT_FRACTION = 0.9  # two seeds must disagree on nearly every element


def _rand_seed_variant() -> QuasarSfpuVariant:
    """Float32 in, Float32 Dest, Float32 out: every PRNG bit the kernel keeps reaches L1."""
    return next(
        v
        for v in generate_quasar_sfpu_format_variants(
            MathOperation.Rand, input_output_formats([DataFormat.Float32])
        )
        if v.dest_acc == DestAccumulation.Yes
    )


@pytest.mark.quasar
def test_rand_seed_quasar():
    """The seed takes effect: same seed repeats bit-for-bit, another seed differs, lock-up is repaired."""
    case = RAND_CASES[0]
    format_variant = _rand_seed_variant()
    element_count = _RAND_DIMS[0] * _RAND_DIMS[1]
    seeds = (
        _RAND_DEFAULT_SEED,
        _RAND_OTHER_SEED,
        _RAND_LOCKUP_SEED,
        _RAND_LOCKUP_REPAIR_SEED,
    )
    configs = {
        seed: _rand_config(case, format_variant, _RAND_DEST_SYNC, _RAND_DIMS, seed)
        for seed in seeds
    }
    # compile-producer skips on run(), so prepare every ELF first.
    for configuration in configs.values():
        configuration.prepare()

    # Run the default seed, then another seed, then the default seed again: an unseeded PRNG
    # would carry on from where the other seed's run left it rather than repeat run one.
    first = _run_rand(configs[_RAND_DEFAULT_SEED], format_variant, element_count)
    other = _run_rand(configs[_RAND_OTHER_SEED], format_variant, element_count)
    again = _run_rand(configs[_RAND_DEFAULT_SEED], format_variant, element_count)
    lockup = _run_rand(configs[_RAND_LOCKUP_SEED], format_variant, element_count)
    repair = _run_rand(configs[_RAND_LOCKUP_REPAIR_SEED], format_variant, element_count)

    for values in (first, other, lockup):
        _check_rand_properties(values, case, format_variant.formats.output_format)

    mismatched = (first != again).nonzero().flatten()
    assert mismatched.numel() == 0, (
        f"the same seed gave a different tile in {mismatched.numel()} of {element_count} "
        f"elements; first at {mismatched[:8].tolist()}"
    )
    different = int((first != other).sum())
    assert different >= _RAND_MIN_DIFFERENT_FRACTION * element_count, (
        f"seeds {_RAND_DEFAULT_SEED:#x} and {_RAND_OTHER_SEED:#x} agree on "
        f"{element_count - different} of {element_count} elements"
    )
    assert torch.equal(lockup, repair), (
        f"seed {_RAND_LOCKUP_SEED:#x} must be repaired to {_RAND_LOCKUP_REPAIR_SEED:#x}, "
        f"but their tiles differ in {int((lockup != repair).sum())} elements"
    )
