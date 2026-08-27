# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    flush_to_zero,
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
3. ttnn.geglu  - GELU-gated Linear Unit: gate_fn(B) = gelu(B)   [tanh approx]
4. ttnn.swiglu - SiLU-gated Linear Unit: gate_fn(B) = silu(B) = B * sigmoid(B)

Accuracy criteria
─────────────────
  glu    : ULP ≤ 2  (sigmoid via SFPU, multiplication rounds at most 1 ULP)
  reglu  : ULP ≤ 1  (relu is exact piecewise linear, multiplication ≤ 1 ULP)
  geglu  : PCC ≥ 0.999  (GELU uses a tanh polynomial approximation)
  swiglu : PCC ≥ 0.999  (silu = B*sigmoid(B); SFPU FTZ for B < -88 gives
                          device → 0 while CPU returns a tiny non-zero, same
                          pattern as atanh in category 1)

Implementation note
───────────────────
The C++ split_tensor_for_glu always indexes inshape[3], so the input must be
4D: [1, 1, H, 2*W] with H and W both multiples of 32 (one TTNN tile = 32×32).
generate_bfloat16_bits() returns a (256, 256) tensor of all 65 536 bfloat16
values.  glu and reglu fix A=1.0 and sweep all values through B (see test
docstrings for why).  swiglu and geglu use generate_bfloat16_bits() for both
halves, giving all 65 536 values in each half.
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


@pytest.mark.parametrize(
    "ttnn_op, ulp",
    [
        (ttnn.glu, 2),
        (ttnn.reglu, 1),
    ],
)
def test_glu_reglu_ops(device, ttnn_op, ulp):
    """Exhaustive bfloat16 coverage for glu and reglu.

    All 65 536 bfloat16 values are tested as the gate-half (B).  A is fixed
    to 1.0 for two practical reasons:
      • glu: sigmoid(B < -88) is a float32 subnormal on the SFPU (FTZ → 0).
        With A=1.0 the output is sigmoid(B) ≈ 6e-39 < bfloat16_tiny, so
        flush_to_zero zeros both sides correctly.  Arbitrary A would amplify
        this into the normal bfloat16 range and produce a spurious ULP failure.
      • reglu: output = A * relu(B). With arbitrary A, large positive A × large
        positive B overflows to Inf.  With A=1.0, output = relu(B) ≤ max_bfloat16.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    A = torch.ones_like(B)  # A=1.0: output = gate_fn(B) directly
    input_tensor = _build_glu_input(A, B)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_tensor, dim=-1, device=device)

    tt_result = ttnn_op(tt_in, dim=-1)
    result = ttnn.to_torch(tt_result)

    result = flush_to_zero(result)
    golden = flush_to_zero(golden)

    assert_with_ulp(golden, result, ulp)


# ─────────────────────────────────────────────────────────────────────────────
# swiglu — PCC (SFPU FTZ for large negative gate inputs)
# ─────────────────────────────────────────────────────────────────────────────


def test_swiglu_op(device):
    """Exhaustive bfloat16 coverage for swiglu.

    silu(B) = B * sigmoid(B).  For large negative B (roughly B < -88),
    sigmoid(B) becomes a float32 subnormal on the SFPU and gets flushed to 0,
    giving device output = 0 while the CPU reference returns a tiny non-zero
    value.  This is the same hardware FTZ behaviour that causes atanh to use
    PCC in category 1.  PCC ≥ 0.999 tolerates these legitimate differences.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    A = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    input_tensor = _build_glu_input(A, B)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.swiglu)
    golden = golden_function(input_tensor, dim=-1, device=device)

    tt_result = ttnn.swiglu(tt_in, dim=-1)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# geglu — PCC (GELU tanh polynomial approximation)
# ─────────────────────────────────────────────────────────────────────────────


def test_geglu_op(device):
    """Exhaustive bfloat16 coverage for geglu.

    GELU uses a tanh polynomial approximation which introduces ~0.1-0.5%
    relative error in the transition region [-3, 3].  PCC ≥ 0.999 is used
    instead of ULP to tolerate this expected approximation error.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    A = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    input_tensor = _build_glu_input(A, B)

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.geglu)
    golden = golden_function(input_tensor, dim=-1, device=device)

    tt_result = ttnn.geglu(tt_in, dim=-1)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, pcc=0.999)
