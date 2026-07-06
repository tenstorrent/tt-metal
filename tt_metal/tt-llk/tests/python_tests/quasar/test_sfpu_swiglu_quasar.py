# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-24_swiglu_quasar_9a4f086d

"""
Functional test for the swiglu SFPU kernel on Quasar.

Swiglu is a binary SFPU op used in GPT-OSS-style MLPs. For two input tiles
``gate`` and ``up``:

    gate_c = min(gate, +7.0)
    up_c   = clip(up, -7.0, +7.0)
    sig    = sigmoid(1.702 * gate_c)
    out    = (up_c + 1.0) * gate_c * sig

Both inputs are laid out in ``buffer_A`` as 2 concatenated tiles (gate at
tile 0, up at tile 1). The kernel unpacks both to Dest, runs the swiglu
inner loop over all 4 faces, and writes the result to Dest tile 2 which is
packed back out to ``buffer_Res``.

The kernel uses hardware ``SFPNONLINEAR`` EXP/RECIP estimators plus a BF16
quantized alpha constant (1.702 → 1.703125), so we accept the PCC-based
tolerance used by other sigmoid-family SFPU kernels rather than an exact
match.
"""

from enum import Enum
from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig, InputOutputFormat
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
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

# GPT-OSS swiglu hyperparameters (matching ckernel_sfpu_swiglu.h
# SwiGLUConfigGPTOSS).
_SWIGLU_ALPHA = 1.702
_SWIGLU_CLAMP_LIMIT = 7.0


def _swiglu_golden(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Numpy/torch reference implementation of GPT-OSS swiglu. Matches the math
    in ckernel_sfpu_swiglu.h:

        gate_c = min(gate, +L)
        up_c   = clip(up, -L, +L)
        out    = (up_c + 1) * gate_c * sigmoid(alpha * gate_c)

    where L = 7.0 and alpha = 1.702.

    Computation is done in float32 then cast to the output format.
    """
    torch_out_dtype = format_dict[output_format]
    gate_f = gate.to(torch.float32)
    up_f = up.to(torch.float32)

    gate_c = torch.minimum(gate_f, torch.tensor(_SWIGLU_CLAMP_LIMIT))
    up_c = torch.clamp(up_f, -_SWIGLU_CLAMP_LIMIT, _SWIGLU_CLAMP_LIMIT)

    # Numerically stable sigmoid via torch.sigmoid (equivalent to 1/(1+e^-x)).
    sig = torch.sigmoid(_SWIGLU_ALPHA * gate_c)

    result = (up_c + 1.0) * gate_c * sig
    return result.to(torch_out_dtype)


# Deterministic boundary / corner-case (gate, up) pairs that exercise every
# branch of the swiglu kernel — clamp boundaries, sigmoid saturation, signed
# zero, big magnitudes, and the (up+1)*gate*sigmoid amplification chain.
# Each tuple is one (gate, up) pair; len(_SWIGLU_CORNER_CASES) lanes at the
# start of the input get these exact values, the rest follow the distribution.
_SWIGLU_CORNER_CASES = (
    # gate, up
    # --- exact clamp boundaries ---
    (+7.0, +7.0),  # exact +clamp on both
    (-7.0, -7.0),  # exact -clamp on up; gate at edge (no min clamp)
    (+7.0, -7.0),  # diagonal: gate=+L, up=-L → (-L+1)*L*~1 = -42
    (-7.0, +7.0),  # diagonal: gate=-L (no clamp), up=+L → (L+1)*-L*~0 ≈ 0
    # --- just inside / outside clamp ---
    (+6.99, +6.99),  # just inside +clamp_limit (no clamp fires)
    (+7.01, +7.01),  # just outside +clamp_limit (clamp fires by epsilon)
    (-6.99, -6.99),  # just inside -clamp_limit on up
    (-7.01, -7.01),  # just outside -clamp_limit on up
    (+6.999, +7.001),  # asymmetric near-boundary
    (-7.001, -6.999),  # asymmetric near-boundary
    # --- way past clamp ---
    (+8.0, +8.0),  # both above +clamp_limit
    (-8.0, -8.0),  # both below -clamp_limit (up clamps to -7, gate stays -8)
    (+50.0, -50.0),  # huge magnitudes: clamp + saturating sigmoid → -42
    (-50.0, +50.0),  # huge negative gate: sigmoid(alpha*-50)≈0 → result≈0
    (+1000.0, +1000.0),  # extreme magnitudes; FP16 saturates to +inf if not clamped
    (-1000.0, -1000.0),  # extreme negative
    # --- zeros and (up+1) = 0 path ---
    (0.0, 0.0),  # linear region origin; sigmoid(0)=0.5 → result = 0
    (0.0, -1.0),  # up+1 = 0 → result = 0 regardless of gate*sig
    (+5.0, -1.0),  # up+1 = 0 with non-zero gate
    (-5.0, -1.0),  # up+1 = 0 with negative gate
    (+7.0, -1.0),  # up+1 = 0 with clamped gate
    # --- tiny magnitudes (linear region) ---
    (+0.001, +0.001),  # tiny positive
    (-0.001, -0.001),  # tiny negative
    (+0.0001, -0.0001),  # very tiny opposite signs
    # --- mixed sign in linear region ---
    (+5.0, -3.0),  # mixed sign, both inside clamp
    (-5.0, +3.0),  # mixed sign, both inside clamp
    (+3.5, -5.5),  # asymmetric mid-range
    # --- one large, one small (asymmetric clamp triggers) ---
    (+50.0, +0.5),  # gate clamps, up linear
    (+0.5, +50.0),  # gate linear, up clamps
    (-50.0, +0.5),  # gate negative big, up linear
    (+0.5, -50.0),  # gate linear, up clamps to -7
    # --- alpha*gate near sigmoid steep region ---
    (+1.0, +0.5),  # alpha*gate=1.7, sigmoid steep
    (+2.0, -0.5),  # alpha*gate=3.4, sigmoid ~0.97
)


class _StimulusDistribution(Enum):
    """Stimulus distribution shapes for the swiglu test sweep.

    Each distribution explores a different region of the swiglu input space:
    LOG_UNIFORM is the default broad sweep; UNIFORM is uniformly distributed
    over [-9, +9] (more density inside the clamp range); GAUSSIAN concentrates
    near 0 (mostly linear-region behavior); LARGE_ONLY forces every lane to
    cross the ±clamp_limit boundary (worst-case clamp pressure); TINY_ONLY
    keeps everything in [-0.5, +0.5] (sigmoid linear-around-zero behavior).
    """

    LOG_UNIFORM = "log_uniform"
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    LARGE_ONLY = "large_only"
    TINY_ONLY = "tiny_only"


def _normalize_to_unit(x: torch.Tensor) -> torch.Tensor:
    """Map a tensor to [0, 1] linearly. Uniform tensor returns all zeros."""
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        return (x - x_min) / (x_max - x_min)
    return torch.zeros_like(x)


def _signs_from(src_B: torch.Tensor) -> torch.Tensor:
    """Per-lane ±1 signs derived deterministically from src_B."""
    src_B_u = _normalize_to_unit(src_B.to(torch.float32))
    return torch.where(src_B_u < 0.5, -1.0, 1.0)


def _prepare_swiglu_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    distribution: _StimulusDistribution = _StimulusDistribution.LOG_UNIFORM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (gate, up) tensors that exercise the swiglu kernel. The first
    ``len(_SWIGLU_CORNER_CASES)`` lanes are fixed deterministic boundary
    values; the remainder is filled per the chosen ``distribution``.

    For Float16 the ranges are bounded so that intermediate / final values
    from swiglu stay within fp16 range: |(up+1) * gate * 1| ≤ (7+1) * 7 = 56.
    """
    torch_dtype = format_dict[input_format]

    src_A_float = src_A.to(torch.float32)
    src_A_u = _normalize_to_unit(src_A_float)
    signs = _signs_from(src_B)

    if distribution == _StimulusDistribution.LOG_UNIFORM:
        # Magnitudes log-uniform in [0.1, 9.0], signed by src_B.
        log_low = torch.log(torch.tensor(0.1, dtype=torch.float32))
        log_high = torch.log(torch.tensor(9.0, dtype=torch.float32))
        magnitudes = torch.exp(log_low + src_A_u * (log_high - log_low))
        gate_up = signs * magnitudes
    elif distribution == _StimulusDistribution.UNIFORM:
        # Uniform in [-9, +9] — more density inside the ±7 clamp range.
        gate_up = (src_A_u * 2.0 - 1.0) * 9.0
    elif distribution == _StimulusDistribution.GAUSSIAN:
        # Gaussian σ=3 from src_A_float treated as random — most lanes inside
        # ±7, with ~5% in the clamp region. Uses Box-Muller via src_A_u as the
        # uniform input (already in [0, 1]).
        eps = torch.tensor(1e-7, dtype=torch.float32)
        u1 = torch.clamp(src_A_u, min=eps, max=1.0 - eps.item())
        u2 = _normalize_to_unit(src_B.to(torch.float32))
        u2 = torch.clamp(u2, min=eps, max=1.0 - eps.item())
        z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * torch.pi * u2)
        gate_up = z * 3.0  # σ = 3
    elif distribution == _StimulusDistribution.LARGE_ONLY:
        # |x| ∈ [7.5, 50] — every lane crosses ±clamp_limit, max stress on clamps.
        gate_up = signs * (7.5 + src_A_u * 42.5)
    elif distribution == _StimulusDistribution.TINY_ONLY:
        # |x| ∈ [0, 0.5] — sigmoid linear-around-zero, no clamps fire.
        gate_up = signs * (src_A_u * 0.5)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    gate_up = gate_up.to(torch.float32)

    # Split into gate (first half) and up (second half). The C++ test expects
    # buffer_A to contain [gate_tile, up_tile] in that order.
    n = gate_up.numel() // 2
    gate = gate_up[:n].clone()
    up = gate_up[n:].clone()

    # Inject deterministic corner cases at the head of each half, but only if
    # there's room (smaller tile counts may have fewer than the corner count).
    num_corner = min(len(_SWIGLU_CORNER_CASES), gate.numel(), up.numel())
    for i in range(num_corner):
        g_val, u_val = _SWIGLU_CORNER_CASES[i]
        gate[i] = g_val
        up[i] = u_val

    return gate.to(torch_dtype), up.to(torch_dtype)


def _generate_sfpu_swiglu_combinations(formats_list: List[FormatConfig]):
    """
    Build the parametrize product for the swiglu test. Wraps the shared
    ``generate_sfpu_format_dest_acc_combinations`` from ``helpers.param_config``
    with an extra ``implied_math_format`` axis.
    """
    combinations = []

    for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(formats_list):
        for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
            # MX formats aren't part of this test but keep the pattern
            # consistent with other float SFPU tests.
            if (
                fmt.input_format.is_mx_format()
                and implied_math_format == ImpliedMathFormat.No
            ):
                continue

            combinations.append((fmt, dest_acc, implied_math_format))

    return combinations


# Float-only per the planner spec (swiglu needs float arithmetic: clamp,
# sigmoid via SFPNONLINEAR EXP/RECIP, alpha/clamp_limit float constants).
# Tf32 is supported by the kernel but the test framework's format_dict in
# helpers/llk_params.py has no Tf32 entry yet, so we leave it for follow-up.
SFPU_SWIGLU_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Float32,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math=_generate_sfpu_swiglu_combinations(
        SFPU_SWIGLU_FORMATS
    ),
    distribution=runtime(list(_StimulusDistribution)),
)
def test_sfpu_swiglu_quasar(formats_dest_acc_implied_math, distribution):
    """
    Swiglu SFPU kernel end-to-end test.

    Input layout: ``buffer_A`` = 2 tiles [gate, up]. Output: 1 tile (swiglu
    result). The gate/up tensor is (64, 32) split in half: first 1024 → gate,
    next 1024 → up. The first ``len(_SWIGLU_CORNER_CASES)`` lanes are
    deterministic boundary cases; the rest is sampled from ``distribution``.
    """
    (formats, dest_acc, implied_math_format) = formats_dest_acc_implied_math

    torch.manual_seed(42)

    num_faces = 4

    # 64x32 = 2 tiles; we interpret the first tile as gate, the second as up.
    input_dimensions = [64, 32]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    gate, up = _prepare_swiglu_inputs(src_A, src_B, formats.input_format, distribution)

    # Rebuild the combined input tensor with our crafted (gate, up) values.
    combined_input = torch.cat([gate, up]).to(format_dict[formats.input_format])

    # Compute the golden on CPU: element-wise swiglu of (gate, up).
    golden_tile = _swiglu_golden(gate, up, formats.output_format)
    # Golden is a single tile's worth of elements.
    golden_tensor = golden_tile.flatten().to(format_dict[formats.output_format])

    # SFPU tests: unpack_to_dest only when format bit-width matches Dest mode.
    unpack_to_dest = formats.input_format.is_32_bit() == (
        dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_swiglu_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuSwiGLU),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
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
            # Only 1 output tile (swiglu writes Dest[2] → buffer_Res[0]).
            tile_count_res=1,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor), (
        f"Result length {len(res_from_L1)} does not match golden length "
        f"{len(golden_tensor)}"
    )

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # passed_test enforces both the per-format isclose default (atol=0.05,
    # rtol=0.05 for Float16/Float16_b/Float32) and PCC ≥ 0.99. Empirically all
    # base variants pass at PCC ≥ 0.99999, so no custom thresholds are needed.
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), "Assert against golden failed for swiglu"


# ----------------------------------------------------------------------------
# NaN / Inf propagation test (Tier 1)
# ----------------------------------------------------------------------------
#
# Goals:
#   - Verify the kernel does not crash on ±Inf or NaN inputs.
#   - Verify ±Inf in `gate` is handled by the relu-based clamps (gate=+Inf
#     should be clamped to +L; gate=-Inf has no min clamp so should produce
#     ~0 via sigmoid saturation).
#   - Verify ±Inf in `up` clamps to ±L.
#   - Document NaN behavior: hardware SFPNONLINEAR/SFPMAD on NaN is not
#     guaranteed by the Quasar ISA pages; this test records what the
#     simulator does so future kernel changes don't silently regress it.
#
# We use Float32/Float32 with dest_acc=Yes — the most precise path. If NaN
# behavior diverges from the torch golden, the test will fail and we'll
# document the divergence. If it converges, we lock in the behavior.
_SWIGLU_NAN_INF_CASES = (
    # gate, up
    (float("inf"), 0.5),  # +Inf gate → clamps to +7
    (float("-inf"), 0.5),  # -Inf gate → no min clamp → sigmoid(-Inf*alpha)=0 → result≈0
    (0.5, float("inf")),  # +Inf up → clamps to +7
    (0.5, float("-inf")),  # -Inf up → clamps to -7
    (float("inf"), float("inf")),  # both +Inf → result = 8*7*1 = 56
    (float("-inf"), float("-inf")),  # both -Inf → result ≈ 0 (sigmoid saturates low)
    (float("inf"), float("-inf")),  # +Inf gate, -Inf up → -42
    (float("-inf"), float("inf")),  # -Inf gate, +Inf up → ≈ 0
    (float("nan"), 0.5),  # NaN gate → expect NaN
    (0.5, float("nan")),  # NaN up → expect NaN
    (float("nan"), float("nan")),  # both NaN → NaN
)


@pytest.mark.quasar
def test_sfpu_swiglu_nan_inf_quasar():
    """
    Verify swiglu kernel behavior on ±Inf and NaN inputs (Float32 path).

    The first ``len(_SWIGLU_NAN_INF_CASES)`` lanes hold deterministic special
    values; the rest are zeros (so any divergence is isolated to the
    special-value lanes). We use the same Float32→Float32 dest_acc=Yes path
    as the main test and the standard ``passed_test`` tolerance.
    """
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    implied_math_format = ImpliedMathFormat.No

    torch.manual_seed(42)

    num_faces = 4
    input_dimensions = [64, 32]

    # We don't actually use src_A / src_B here, but generate_stimuli sets up
    # the framework state (tile_cnt_A, etc.) consistently with the main test.
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Build all-zero gate/up, then inject NaN/Inf cases at the head.
    n = (input_dimensions[0] * input_dimensions[1]) // 2  # 1024 lanes per half
    gate = torch.zeros(n, dtype=torch.float32)
    up = torch.zeros(n, dtype=torch.float32)
    for i, (g_val, u_val) in enumerate(_SWIGLU_NAN_INF_CASES):
        gate[i] = g_val
        up[i] = u_val

    # Cast to input format.
    gate_f = gate.to(format_dict[formats.input_format])
    up_f = up.to(format_dict[formats.input_format])
    combined_input = torch.cat([gate_f, up_f]).to(format_dict[formats.input_format])

    # Golden: torch swiglu propagates NaN naturally.
    golden_tile = _swiglu_golden(gate_f, up_f, formats.output_format)
    golden_tensor = golden_tile.flatten().to(format_dict[formats.output_format])

    unpack_to_dest = formats.input_format.is_32_bit() == (
        dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_swiglu_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.SfpuSwiGLU),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
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

    assert len(res_from_L1) == len(golden_tensor), (
        f"Result length {len(res_from_L1)} does not match golden length "
        f"{len(golden_tensor)}"
    )

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # passed_test handles paired-NaN as valid, paired-Inf via isclose.
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "swiglu kernel diverged from golden on NaN/Inf inputs"
