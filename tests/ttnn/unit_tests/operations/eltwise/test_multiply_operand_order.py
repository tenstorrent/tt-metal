# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Operand order in multiply.

The order an operand is passed in decides whether it is read into srcA or srcB, and the FPU does
not reduce the two source slots' bfloat16 mantissas alike, so x * y and y * x are not the same
computation there. Off the FPU the order cannot matter and commutativity is exact; on it the
contract is a bound, and these tests pin both halves. Block-float operands are computed on the FPU
whatever fast_and_approximate_mode says, so they are covered by the bound rather than by the exact
case.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import ulp_distance

pytestmark = pytest.mark.use_module_device

# Worst case measured over every case below, on P300 and on N300: 2 ULP. One of margin, as
# _SCALAR_FIRST_ULP_THRESHOLD does in test_binary_scalar.py.
ORDER_ULP_THRESHOLD = 3

SHAPE = (1, 1, 256, 256)

# Clear of the smallest normal in both directions, so a single rounding is the only error in play
# and the FPU is not also losing bits to gradual underflow.
WIDE_EXPONENTS = (-40, 40)
# One shared exponent covers a whole block of a block-float tensor, so a wide spread inside a block
# would quantise most of it to zero and leave nothing to compare.
BLOCK_FLOAT_EXPONENTS = (-2, 2)


def _operand(seed, exponent_range, torch_dtype):
    """Uniform mantissas, so every stored bit is exercised, over a controlled exponent range."""
    gen = torch.Generator().manual_seed(seed)
    mantissa = torch.rand(SHAPE, generator=gen) + 1.0
    low, high = exponent_range
    scale = torch.exp2(torch.randint(low, high + 1, SHAPE, generator=gen).float())
    sign = torch.where(torch.rand(SHAPE, generator=gen) < 0.5, -1.0, 1.0)
    return (mantissa * scale * sign).to(torch_dtype)


def _both_orders(device, op, lhs_dtype, rhs_dtype, exponent_range, **kwargs):
    is_float32 = lhs_dtype == ttnn.float32 and rhs_dtype == ttnn.float32
    torch_dtype = torch.float32 if is_float32 else torch.bfloat16
    out_dtype = ttnn.float32 if is_float32 else ttnn.bfloat16
    a = ttnn.from_torch(
        _operand(0x11, exponent_range, torch_dtype), dtype=lhs_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = ttnn.from_torch(
        _operand(0x22, exponent_range, torch_dtype), dtype=rhs_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    forward = ttnn.to_torch(op(a, b, dtype=out_dtype, **kwargs), dtype=torch_dtype)
    reverse = ttnn.to_torch(op(b, a, dtype=out_dtype, **kwargs), dtype=torch_dtype)
    return forward, reverse


def _worst(forward, reverse):
    distance = ulp_distance(forward, reverse).flatten()
    i = int(distance.argmax())
    return (
        f"element {i} is {int(distance[i])} ULP apart: {forward.flatten()[i].item()!r} in one "
        f"operand order, {reverse.flatten()[i].item()!r} in the other"
    )


@pytest.mark.parametrize(
    "dtype, fast_and_approximate_mode",
    [(ttnn.bfloat16, False), (ttnn.float32, False), (ttnn.float32, True)],
    ids=["bfloat16-accurate", "float32-accurate", "float32-fast"],
)
def test_multiply_is_bit_commutative_off_the_fpu(device, dtype, fast_and_approximate_mode):
    """The SFPU kernel reads both operands the same way, and float32 never reaches the FPU in
    either mode, so both orders must return the same bits."""
    forward, reverse = _both_orders(
        device, ttnn.multiply, dtype, dtype, WIDE_EXPONENTS, fast_and_approximate_mode=fast_and_approximate_mode
    )
    assert int(ulp_distance(forward, reverse).max()) == 0, _worst(forward, reverse)


def test_fpu_multiply_orders_agree_within_a_few_ulp(device):
    """bfloat16 on the FPU is the one case that is not bit-commutative. It still owes a bound: a
    swapped srcA costs a couple of ULP, and anything larger is a different defect."""
    forward, reverse = _both_orders(
        device, ttnn.multiply, ttnn.bfloat16, ttnn.bfloat16, WIDE_EXPONENTS, fast_and_approximate_mode=True
    )
    assert int(ulp_distance(forward, reverse).max()) <= ORDER_ULP_THRESHOLD, _worst(forward, reverse)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfloat8_b", "bfloat4_b"])
def test_block_float_multiply_orders_agree_within_a_few_ulp(device, dtype):
    """No flag is passed, because a block-float operand selects the FPU whatever it says. The
    counterpart is bfloat16, which is both the shape models use and the one that keeps enough
    mantissa for the asymmetry to show: two block-float operands carry too few stored bits to
    reach it."""
    forward, reverse = _both_orders(device, ttnn.multiply, dtype, ttnn.bfloat16, BLOCK_FLOAT_EXPONENTS)
    assert int(ulp_distance(forward, reverse).max()) <= ORDER_ULP_THRESHOLD, _worst(forward, reverse)


@pytest.mark.parametrize("fast_and_approximate_mode", [False, True])
def test_add_is_bit_commutative_on_both_paths(device, fast_and_approximate_mode):
    """The asymmetry is in the multiplier, not in the source slots themselves: the adder reads the
    same two slots and is exactly commutative on either path."""
    forward, reverse = _both_orders(
        device,
        ttnn.add,
        ttnn.bfloat16,
        ttnn.bfloat16,
        WIDE_EXPONENTS,
        fast_and_approximate_mode=fast_and_approximate_mode,
    )
    assert int(ulp_distance(forward, reverse).max()) == 0, _worst(forward, reverse)
