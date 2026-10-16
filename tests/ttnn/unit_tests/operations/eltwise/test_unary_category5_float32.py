# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_ulp
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    assert_ftz_band,
    assert_ftz_then_ulp,
    generate_float32_bits,
    to_float32_tt_tensor,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device


"""
Category 5: ops with a dim parameter

Each op splits its input in half along dim, A, B = split(input, 2, dim), and
returns A * gate(B).

| # | Op          | Gate                          |
|---|-------------|-------------------------------|
| 1 | ttnn.glu    | sigmoid(B)                    |
| 2 | ttnn.reglu  | relu(B)                       |
| 3 | ttnn.geglu  | gelu(B), per GeluVariant      |
| 4 | ttnn.swiglu | silu(B) = B * sigmoid(B)      |

Every test fixes A = 1.0, so the output is the gate itself and the sweep runs
over every normal bfloat16 value of B. A swapped A/B would still show up, as
A * gate(1.0) instead of gate(B).

Accuracy criteria (FTZ = flush-to-zero, SNB = smallest normal bfloat16)

| Op     | Input range (B)     | Check                      |
|--------|---------------------|----------------------------|
| glu    | golden < SNB        | exactly 0                  |
|        | elsewhere           | ULP ≤ 2                    |
| reglu  | all                 | exact                      |
| swiglu | |B| < 2*SNB         | exactly 0                  |
|        | -88.5 <= B <= -87.5 | exactly 0                  |
|        | elsewhere           | ULP ≤ 3                    |
| geglu  | all                 | bit-identical to ttnn.gelu |
|        |                     | (sign of zero ignored)     |
"""


def _gate_input(B):
    """[1, 1, H, 2*W] input with value half A = 1.0 and gate half B."""
    return torch.cat([torch.ones_like(B), B], dim=-1).unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# glu, reglu — sigmoid and relu gates
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [-1, 3])
def test_glu(device, dim):
    """glu = A * sigmoid(B), with A = 1.

    | Input range (B) | Check     | Reason                                      |
    |-----------------|-----------|---------------------------------------------|
    | golden < SNB    | exactly 0 | sigmoid underflows near B = -87.5; FTZ      |
    | elsewhere       | ULP ≤ 2   | fp32 sigmoid exp/reciprocal error           |
    """
    B = generate_float32_bits()
    input_tensor = _gate_input(B)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.glu)
    golden = golden_function(input_tensor, dim=dim, device=device)

    result = ttnn.to_torch(ttnn.glu(tt_in, dim=dim))

    assert_ftz_then_ulp(golden, result, ulp_threshold=2)


@pytest.mark.parametrize("dim", [-1, 3])
def test_reglu(device, dim):
    """reglu = A * relu(B), with A = 1.

    | Input range (B) | Check | Reason                                 |
    |-----------------|-------|----------------------------------------|
    | all             | exact | relu is a select; multiplying by 1 is  |
    |                 |       | exact                                  |
    """
    B = generate_float32_bits()
    input_tensor = _gate_input(B)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.reglu)
    golden = golden_function(input_tensor, dim=dim, device=device)

    result = ttnn.to_torch(ttnn.reglu(tt_in, dim=dim))

    assert_equal(golden, result)


# ─────────────────────────────────────────────────────────────────────────────
# swiglu — silu gate, same bounds as silu
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [-1, 3])
def test_swiglu(device, dim):
    """swiglu = A * silu(B), with A = 1.

    | Input range (B)     | Check     | Reason                                          |
    |---------------------|-----------|-------------------------------------------------|
    | |B| < 2*SNB         | exactly 0 | output ≈ B/2 is subnormal; FTZ                  |
    | -88.5 <= B <= -87.5 | exactly 0 | device sigmoid FTZs at B <= -87.5; torch's      |
    |                     |           | golden only reaches 0 at B <= -89               |
    | elsewhere           | ULP ≤ 3   | fp32 sigmoid exp/reciprocal error (3 ULP at     |
    |                     |           | B = -11.5)                                      |
    """
    B = generate_float32_bits()
    input_tensor = _gate_input(B)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.swiglu)
    golden = golden_function(input_tensor, dim=dim, device=device)

    result = ttnn.to_torch(ttnn.swiglu(tt_in, dim=dim))

    gate = B.unsqueeze(0).unsqueeze(0)
    near_zero_ftz = gate.abs() < 2 * SMALLEST_NORMAL_BF16
    sigmoid_ftz = (gate >= -88.5) & (gate <= -87.5)
    assert_ftz_band(result, near_zero_ftz, "|B| < 2*SNB")
    assert_ftz_band(result, sigmoid_ftz, "-88.5 <= B <= -87.5")

    remaining = ~near_zero_ftz & ~sigmoid_ftz
    assert_with_ulp(expected_result=golden[remaining], actual_result=result[remaining], ulp_threshold=3)


# ─────────────────────────────────────────────────────────────────────────────
# geglu — bit-identical to ttnn.gelu, per GELU variant
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [-1, 3])
@pytest.mark.parametrize(
    "variant",
    [None, ttnn.GeluVariant.Accurate, ttnn.GeluVariant.FastLut, ttnn.GeluVariant.Tanh],
    ids=["default", "accurate", "fast_lut", "tanh"],
)
def test_geglu(device, dim, variant):
    """geglu = A * gelu(B), with A = 1, for every GELU variant.

    | Input range (B) | Check                          | Reason                               |
    |-----------------|--------------------------------|--------------------------------------|
    | all             | bit-identical to ttnn.gelu     | same gate kernel, so ttnn.gelu's own |
    |                 | with the same variant, with    | accuracy tests carry over            |
    |                 | -0.0 read as +0.0              |                                      |

    geglu is ttnn.gelu(B) followed by ttnn.multiply(A, gelu(B)). On Blackhole that
    multiply returns +0.0 for 1.0 * -0.0, and Accurate gelu returns -0.0 for
    -2^-125 < B <= -2^-126, so the sign of zero is not compared.

    With variant=None, geglu must resolve to Accurate, on the device and in the golden.
    """
    B = generate_float32_bits()
    input_tensor = _gate_input(B)

    tt_in = to_float32_tt_tensor(input_tensor, device)
    gate_in = to_float32_tt_tensor(B.unsqueeze(0).unsqueeze(0), device)
    geglu_kwargs = {} if variant is None else {"variant": variant}

    expected_variant = ttnn.GeluVariant.Accurate if variant is None else variant
    expected = ttnn.to_torch(ttnn.gelu(gate_in, variant=expected_variant))
    result = ttnn.to_torch(ttnn.geglu(tt_in, dim=dim, **geglu_kwargs))

    result_bits = torch.where(result == 0, torch.zeros_like(result), result).view(torch.int32)
    expected_bits = torch.where(expected == 0, torch.zeros_like(expected), expected).view(torch.int32)
    assert torch.equal(result_bits, expected_bits), (
        f"geglu gate diverged from ttnn.gelu(variant={expected_variant}) for "
        f"{int((result_bits != expected_bits).sum().item())} of {result.numel()} elements"
    )

    golden_function = ttnn.get_golden_function(ttnn.geglu)
    golden = golden_function(input_tensor, dim=dim, device=device, **geglu_kwargs)
    golden_explicit = golden_function(input_tensor, dim=dim, device=device, variant=expected_variant)
    assert torch.equal(golden, golden_explicit), (
        f"geglu golden with variant={variant!r} differs from explicit variant={expected_variant} "
        f"for {int((golden != golden_explicit).sum().item())} of {golden.numel()} elements"
    )
