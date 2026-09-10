# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
clamped-SiLU activation family SFPU test.

Entries:
    GATE         y = min(x, limit) * sigmoid(alpha * min(x, limit))
    UP           y = clamp(x, -limit, limit) + 1
    CLAMP_ONLY   y = clamp(x, -limit, limit)
    SITU_GATE    y = beta * tanh(x/beta) * sigmoid(x)
    SCALED_TANH  y = beta * tanh(x/beta)

Scalars reach the kernel as fp32 bit patterns.

GATE clamps only the top (min(x, limit)). UP and CLAMP_ONLY clamp both ends.

tanh is calculated as tanh(u) = 2*sigmoid(2u) - 1. That identity is used in the
golden.

Note that _sfpu_sigmoid_ gets its reciprocal from sfpu_reciprocal_iter, which
requires vConstFloatPrgm0 to be set to 2.0f. The header neither says so nor
provides an init.
"""

import torch
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIM, round_to_dest_width
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CLAMPED_SILU_PARAMS, TILE_COUNT
from helpers.utils import passed_test

# Same format in and out, as these are activations applied in place on Pack.
FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)

GATE = "GATE"
UP = "UP"
CLAMP_ONLY = "CLAMP_ONLY"
SITU_GATE = "SITU_GATE"
SCALED_TANH = "SCALED_TANH"

CLAMP_OPS = [GATE, UP, CLAMP_ONLY]
SITU_OPS = [SITU_GATE, SCALED_TANH]
ALL_OPS = CLAMP_OPS + SITU_OPS


# 7.0 is GPT-OSS' clamp limit, and 1.0 is just an arbitrary small value.
LIMITS = [7.0, 1.0]

# For GATE. 1.702 is the usual GELU-approximating value, and 1.0 makes GATE a
# plain clamped SiLU, so a dropped alpha multiply still shows up on the 1.702 side.
ALPHAS = [1.702, 1.0]

# For SiTU. The kernel takes 1/beta from the caller.
BETAS = [1.0, 2.0]

# (low, high) ranges.
# Zeros are avoided for SiTU because of the way tanh is computed.
CLAMP_RANGES = [(-10.0, 10.0), (-3.0, 3.0), (-0.5, 0.5)]
SITU_RANGES = [(-8.0, 8.0), (-2.0, 2.0)]


def _op_ranges(op):
    return SITU_RANGES if op in SITU_OPS else CLAMP_RANGES


def _op_scalars(op):
    """(scalar0, scalar1) pairs for this entry.

    scalar0 is the limit for the clamping ops and beta for the SiTU ops; scalar1 is
    alpha for GATE and 1/beta for the SiTU ops, and unread for UP and CLAMP_ONLY.
    """
    if op == GATE:
        return [(limit, alpha) for limit in LIMITS for alpha in ALPHAS]
    if op in (UP, CLAMP_ONLY):
        return [(limit, 1.0) for limit in LIMITS]
    return [(beta, 1.0 / beta) for beta in BETAS]


def _tanh_via_sigmoid(u):
    return 2.0 * torch.sigmoid(2.0 * u) - 1.0


def _clamped_silu_golden(x, op, scalar0, scalar1, dest_acc):
    """Golden for one entry, in the kernel's algebra."""
    x = x.to(torch.float32)

    if op == GATE:
        limit, alpha = scalar0, scalar1
        xc = torch.minimum(x, torch.tensor(limit))
        y = xc * torch.sigmoid(alpha * xc)
    elif op == UP:
        y = torch.clamp(x, -scalar0, scalar0) + 1.0
    elif op == CLAMP_ONLY:
        y = torch.clamp(x, -scalar0, scalar0)
    elif op == SITU_GATE:
        beta, beta_recip = scalar0, scalar1
        y = beta * _tanh_via_sigmoid(x * beta_recip) * torch.sigmoid(x)
    elif op == SCALED_TANH:
        beta, beta_recip = scalar0, scalar1
        y = beta * _tanh_via_sigmoid(x * beta_recip)
    else:
        raise ValueError(f"unknown op {op}")

    return round_to_dest_width(y, dest_acc)


def _valid_dest_acc(formats):
    if formats.input.is_32_bit():
        return [DestAccumulation.Yes]
    return get_valid_dest_accumulation_modes(formats)


def _run(formats, dest_acc, op, scalars, input_range):
    scalar0, scalar1 = scalars

    low, high = input_range
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[TILE_DIM, TILE_DIM],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[TILE_DIM, TILE_DIM],
        spec_A=StimuliSpec.uniform(low=low, high=high, seed=0),
    )

    golden = _clamped_silu_golden(src_A, op, scalar0, scalar1, dest_acc)

    configuration = TestConfig(
        "sources/sfpu_clamped_silu_test.cpp",
        formats,
        templates=[
            CLAMPED_SILU_PARAMS(clamped_silu_op=op, scalar0=scalar0, scalar1=scalar1),
        ],
        runtimes=[
            TILE_COUNT(1),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )

    res_from_L1 = configuration.run().result
    res = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    return res.to(torch.float32), golden, src_A.to(torch.float32)


@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=lambda formats: _valid_dest_acc(formats),
    op=ALL_OPS,
    scalars=lambda op: _op_scalars(op),
    input_range=lambda op: _op_ranges(op),
)
def test_sfpu_clamped_silu(formats, dest_acc, op, scalars, input_range):
    scalar0, scalar1 = scalars
    res, golden, x = _run(formats, dest_acc, op, scalars, input_range)

    assert passed_test(
        golden.to(format_dict[formats.output_format]),
        res.to(format_dict[formats.output_format]),
        formats.output_format,
        print_errors=True,
    ), (
        f"{op} does not match golden (scalar0={scalar0}, scalar1={scalar1}, "
        f"x in {input_range})"
    )

    # Anything past the limit must land on the clamped value.
    if op in (UP, CLAMP_ONLY):
        offset = 1.0 if op == UP else 0.0  # UP adds 1 when clamping
        for side, mask in (
            (scalar0, x > scalar0),
            (-scalar0, x < -scalar0),
        ):
            if not bool(mask.any()):
                continue
            expected = side + offset
            got = res[mask]
            assert bool(
                ((got - expected).abs() <= 1e-2 * max(abs(expected), 1.0)).all()
            ), (
                f"{op}: inputs past {'+' if side > 0 else '-'}limit must clamp to "
                f"{expected}, got range [{got.min().item()}, {got.max().item()}]"
            )

    # Below -limit the one-sided min() leaves x alone. passed_test cannot see this:
    # the correlation barely moves when a two-sided clamp pins those lanes to
    # -limit * sigmoid(-alpha * limit), so compare them elementwise instead.
    if op == GATE:
        limit, alpha = scalar0, scalar1
        below = x < -limit
        if bool(below.any()):
            expected = x[below] * torch.sigmoid(alpha * x[below])
            assert bool(
                torch.isclose(res[below], expected, rtol=5e-2, atol=5e-3).all()
            ), (
                f"GATE clamps only the top (min(x, limit)); inputs below -limit "
                f"must stay on x * sigmoid(alpha * x), got "
                f"{res[below][:4].tolist()} against {expected[:4].tolist()}"
            )
