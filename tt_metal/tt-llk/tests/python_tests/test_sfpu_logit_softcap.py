# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
logit-softcap SFPU test. Covers llk_sfpu/ckernel_sfpu_logit_softcap.h.

cap reaches the kernel as an fp32 bit pattern. The second parameter is not read.
This runs on Pack.

_sfpu_tanh_fp32_accurate_ reads vConstFloatPrgm0 and vConstFloatPrgm1, but the
kernel doesn't mention that and gives no init. They are programmed by tanh_init,
and the branch that must run for that is is_fp32_dest_acc_en = true. The false
branch loads coefficients belonging to a different tanh implementation.

tanh saturates to +-1 by |x| around 5, so the test covers the linear region, the
knee, and deep saturation.
"""

import struct

import torch
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIM, truncate_to_bfloat16
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SFPU_UNARY_SCALAR, TILE_COUNT
from helpers.utils import passed_test

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])

# (low, high) input ranges. tanh is within 1e-4 of +-1 by |x| = 5.
INPUT_RANGES = [
    (-1.0, 1.0),  # near-linear region
    (-4.0, 4.0),  # through the knee
    (-20.0, 20.0),  # saturated tails and the near-zero middle together
    (8.0, 20.0),  # saturated, positive
    (-20.0, -8.0),  # saturated, negative
    (0.0, 0.0),  # tanh(0) = 0 exactly, so y must be exactly 0
]

# |tanh(x)| > 1 - 2e-5 beyond this, so y is cap to well within any tolerance here.
SATURATED_ABS = 6.0

# Gemma's final logit softcap is 30.0. 1.0 is the degenerate pass-through case,
# and 0.5 exercises a cap below 1.
CAPS = [30.0, 1.0, 0.5]


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _valid_dest_acc(formats):
    if formats.input.is_32_bit():
        return [DestAccumulation.Yes]
    return get_valid_dest_accumulation_modes(formats)


def _logit_softcap_golden(input_tensor, cap: float, dest_acc):
    y = cap * torch.tanh(input_tensor.to(torch.float32))
    if dest_acc == DestAccumulation.Yes:
        return y
    return truncate_to_bfloat16(y)


def _assert_matches(golden, res, output_format, flat: bool, context: str) -> None:
    """Compare golden to result, elementwise for flat cases and by PCC otherwise."""
    golden_f = golden.to(torch.float32)
    res_f = res.to(torch.float32)

    if flat:
        assert bool(torch.isclose(res_f, golden_f, rtol=1e-2, atol=1e-3).all()), (
            f"{context}: flat golden ~{golden_f.flatten()[0].item()}, got range "
            f"[{res_f.min().item()}, {res_f.max().item()}]"
        )
        return

    assert passed_test(golden, res, output_format, print_errors=True), context


@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=lambda formats: _valid_dest_acc(formats),
    input_range=INPUT_RANGES,
    cap=CAPS,
)
def test_sfpu_logit_softcap(formats, dest_acc, input_range, cap):
    low, high = input_range
    spec_A = (
        StimuliSpec.constant(value=low, seed=0)
        if low == high
        else StimuliSpec.uniform(low=low, high=high, seed=0)
    )
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[TILE_DIM, TILE_DIM],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[TILE_DIM, TILE_DIM],
        spec_A=spec_A,
    )

    golden_tensor = _logit_softcap_golden(src_A, cap, dest_acc)

    configuration = TestConfig(
        "sources/sfpu_logit_softcap_test.cpp",
        formats,
        templates=[
            SFPU_UNARY_SCALAR(value_bits=_fp32_bits(cap)),
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
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Every lane lands on the same value, either because the whole range sits
    # on one saturated side (y = sgn(x) * cap) or because x is 0 (y = 0).
    flat = (
        low >= SATURATED_ABS or high <= -SATURATED_ABS or (low == 0.0 and high == 0.0)
    )

    _assert_matches(
        golden_tensor,
        res_tensor,
        formats.output_format,
        flat,
        f"cap * tanh(x) does not match golden (cap={cap}, x in {input_range})",
    )

    # tanh(0) = 0, so this can be checked exactly.
    if low == 0.0 and high == 0.0:
        assert bool(
            (res_tensor.to(torch.float32) == 0.0).all()
        ), f"tanh(0) = 0 so the output must be exactly 0 for cap={cap}"

    # Saturated values must sit at +-cap.
    saturated = src_A.to(torch.float32).abs() >= SATURATED_ABS
    if bool(saturated.any()):
        expected = cap * torch.sign(src_A.to(torch.float32)[saturated])
        got = res_tensor.to(torch.float32)[saturated]
        # Allow a little room at cap=30/50 for bf16 (im)precision.
        assert bool(((got - expected).abs() <= 0.02 * cap).all()), (
            f"saturated lanes (|x| >= {SATURATED_ABS}) must clamp to sgn(x) * cap "
            f"= +-{cap}, got range [{got.min().item()}, {got.max().item()}]"
        )
