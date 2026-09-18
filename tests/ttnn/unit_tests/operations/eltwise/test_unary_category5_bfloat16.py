# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    generate_bfloat16_bits_in_range,
    to_tt_tensor,
)

pytestmark = pytest.mark.use_module_device

"""
Category 5: Ops with dim parameter
All four ops split the input tensor in half along dim=-1:

    A, B = split(input, 2, dim=-1)
    output = A * gate_fn(B)

1. ttnn.glu    - Gated Linear Unit:      gate_fn(B) = sigmoid(B)
2. ttnn.reglu  - ReLU-gated Linear Unit: gate_fn(B) = relu(B)
3. ttnn.geglu  - GELU-gated Linear Unit: gate_fn(B) = gelu(B)   [Accurate variant]
4. ttnn.swiglu - SiLU-gated Linear Unit: gate_fn(B) = silu(B) = B * sigmoid(B)

Accuracy criteria
─────────────────
  glu    : ULP ≤ 2  (sigmoid via SFPU, multiplication rounds at most 1 ULP)
  reglu  : ULP ≤ 1  (relu is exact piecewise linear, multiplication ≤ 1 ULP)
  geglu  : bitwise equality against ttnn.gelu on an A = 1.0 gate, per variant
                          (pins the gate implementation, so ttnn.gelu's own
                          accuracy tests apply)
  swiglu : PCC ≥ 0.999  (silu = B*sigmoid(B); SFPU FTZ for B < -88 gives
                          device → 0 while CPU returns a tiny non-zero value;
                          ULP cannot distinguish legitimate hardware FTZ)
"""


def _build_glu_input(A, B):
    """Concatenate value-half A and gate-half B into a 4D input for glu-family ops.

    A and B must be the same shape [H, W] with H and W multiples of 32.
    Returns [1, 1, H, 2*W].
    """
    assert A.shape == B.shape, f"A and B must have identical shapes; got {A.shape} vs {B.shape}"
    return torch.cat([A, B], dim=-1).unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# glu, reglu — ULP-based
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [-1, 3])
@pytest.mark.parametrize(
    "ttnn_op, ulp",
    [
        (ttnn.glu, 2),
        (ttnn.reglu, 1),
    ],
)
def test_glu_reglu_ops(device, ttnn_op, ulp, dim):
    """Exhaustive normal bfloat16 coverage for glu and reglu.

    All 32 512 positive and 32 512 negative normal bfloat16 values are swept
    through the gate-half (B).  A is fixed to 1.0 to avoid FTZ amplification
    for glu and overflow for reglu with large inputs.  A half-swap is still
    detectable: it would produce B * sigmoid(1.0) instead of sigmoid(B).
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    A = torch.ones_like(B)  # A=1.0: output = gate_fn(B) directly
    input_tensor = _build_glu_input(A, B)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, dim=dim, device=device)

    tt_result = ttnn_op(tt_in, dim=dim)
    result = ttnn.to_torch(tt_result)

    # Derive FTZ mask from golden: permit device to output 0 only where the
    # reference itself is subnormal.  Do not flush result independently.
    tiny = torch.finfo(torch.bfloat16).tiny
    ftz = (golden.abs() > 0) & (golden.abs() < tiny)
    golden = golden.clone()
    golden[ftz] = 0.0
    result = result.clone()
    result[ftz] = 0.0

    assert_with_ulp(golden, result, ulp)


# ─────────────────────────────────────────────────────────────────────────────
# swiglu — PCC (SFPU FTZ for large negative gate inputs)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [-1, 3])
def test_swiglu_op(device, dim):
    """Exhaustive normal bfloat16 coverage for swiglu.

    PCC ≥ 0.999 is used because SFPU flushes sigmoid subnormals to zero for
    large negative gate inputs, causing small legitimate differences from the
    CPU reference.  A uses values in [-1, 1] to prevent A * silu(B) from
    overflowing to inf for large-magnitude B.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256) — all normal bf16
    A_vals = generate_bfloat16_bits_in_range(-1.0, 1.0).flatten()
    A = A_vals.repeat(B.numel() // A_vals.numel() + 1)[: B.numel()].view(B.shape)
    input_tensor = _build_glu_input(A, B)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.swiglu)
    golden = golden_function(input_tensor, dim=dim, device=device)

    tt_result = ttnn.swiglu(tt_in, dim=dim)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# geglu — bitwise gate comparison against ttnn.gelu, per GELU variant
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim", [-1, 3])
@pytest.mark.parametrize(
    "variant",
    [None, ttnn.GeluVariant.Accurate, ttnn.GeluVariant.FastLut, ttnn.GeluVariant.Tanh],
    ids=["default", "accurate", "fast_lut", "tanh"],
)
def test_geglu_op(device, dim, variant):
    """Exhaustive normal bfloat16 coverage for geglu, for every GELU variant.

    geglu's gate must be bitwise-identical to ttnn.gelu with the same variant,
    which pins the gate implementation exactly and makes ttnn.gelu's own
    accuracy tests carry over.  Passing no variant must resolve to Accurate.

    A = 1.0 isolates the gate: the multiply becomes exact, so the output is the
    gate value itself and can be compared bit-for-bit.  A half-swap is still
    detectable, since it would produce B * gelu(1.0) instead of gelu(B).
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256) — all normal bf16 values
    input_tensor = _build_glu_input(torch.ones_like(B), B)

    tt_in = to_tt_tensor(input_tensor, device)
    gate_in = to_tt_tensor(B.unsqueeze(0).unsqueeze(0), device)
    geglu_kwargs = {} if variant is None else {"variant": variant}

    expected_variant = ttnn.GeluVariant.Accurate if variant is None else variant
    expected = ttnn.to_torch(ttnn.gelu(gate_in, variant=expected_variant))
    result = ttnn.to_torch(ttnn.geglu(tt_in, dim=dim, **geglu_kwargs))

    assert torch.equal(result, expected), (
        f"geglu gate diverged from ttnn.gelu(variant={expected_variant}) for "
        f"{int((result != expected).sum().item())} of {result.numel()} elements"
    )
    # Verify the golden function resolves variant=None to GeluVariant.Accurate by
    # comparing its output against the golden called with the explicit expected_variant.
    # Both calls go through the identical code path (same split tensors, same gelu
    # kernel), so they must agree regardless of subnormal/FTZ behaviour on the runner
    golden = ttnn.get_golden_function(ttnn.geglu)(input_tensor, dim=dim, device=device, **geglu_kwargs)
    golden_explicit = ttnn.get_golden_function(ttnn.geglu)(
        input_tensor, dim=dim, device=device, variant=expected_variant
    )

    assert torch.equal(golden, golden_explicit), (
        f"geglu golden with variant={variant!r} differs from explicit variant={expected_variant} "
        f"for {int((golden != golden_explicit).sum().item())} of {golden.numel()} elements"
    )
