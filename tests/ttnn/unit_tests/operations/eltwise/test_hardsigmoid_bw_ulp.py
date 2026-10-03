# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive BF16 accuracy of ttnn.hardsigmoid_bw.

Every BF16 bit pattern is paired with each gradient in GRADS. The reference is torch autograd
of torch.nn.functional.hardsigmoid(x) in float64, rounded once to BF16 with subnormal results flushed to zero.
The SFPU may read a subnormal operand as zero, so torch is evaluated both at the operands and at
the operands with subnormals flushed, and an output may match either. The BF16 compute and pack
path stores NaN as +inf and -0 as +0, so classes are compared as stored. Each output must have
the reference's class and a pure ULP error, |reference - output| / ulp(rounded reference),
below 1.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import generate_all_bfloat16_bitpatterns

GRADS = ["1", "-1", "0.5", "3", "random0", "random1"]
SMALLEST_NORMAL = 2.0**-126


def _flush(t):
    return torch.where(t.abs() < SMALLEST_NORMAL, torch.zeros_like(t), t)


def _grad_like(x, grad):
    if grad.startswith("random"):
        generator = torch.Generator().manual_seed(int(grad.removeprefix("random")))
        return torch.randn(x.shape, generator=generator).to(torch.bfloat16)
    return torch.full(x.shape, float(grad), dtype=torch.bfloat16)


def _reference(grad, x, flush):
    with torch.enable_grad():
        x64, g = x.to(torch.float64), grad.to(torch.float64)
        if flush:
            x64, g = _flush(x64), _flush(g)
        x64.requires_grad_(True)
        torch.nn.functional.hardsigmoid(x64).backward(g)
        return x64.grad.detach()


def _round_to_bfloat16(t):
    """Round float64 to BF16 once (round-to-odd into float32, then nearest-even), then flush."""
    f32 = t.to(torch.float32)
    back = f32.to(torch.float64)
    inexact = torch.isfinite(t) & (back != t)
    bits = f32.view(torch.int32) - (inexact & (back.abs() > t.abs())).to(torch.int32)
    bits = bits | inexact.to(torch.int32)
    return _flush(bits.view(torch.float32).to(torch.bfloat16))


def _stored_classes(t):
    """0 +inf (or NaN), 1 -inf, 2 zero of either sign, 3 finite nonzero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, 3, dtype=torch.int8)
    classes[t == 0] = 2
    classes[(t == float("inf")) | torch.isnan(t)] = 0
    classes[t == float("-inf")] = 1
    return classes


def _pure_ulp(reference, actual):
    """Pure ULP error, infinite where the stored class differs."""
    rounded = _round_to_bfloat16(reference).to(torch.float64)
    magnitude = rounded.abs()
    exponent = torch.floor(torch.log2(torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))))
    spacing = torch.where(magnitude > 0, 2.0 ** (exponent.clamp(min=-126) - 7), torch.full_like(magnitude, 2.0**-133))
    # The numerator is flushed only where the correctly rounded result is zero (post-round flush).
    golden = torch.where(magnitude == 0, torch.zeros_like(reference), reference)
    ulp = ((golden - actual.to(torch.float64)).abs().to(torch.float32) / spacing.to(torch.float32)).to(torch.float64)
    same_class = _stored_classes(rounded) == _stored_classes(actual)
    ulp = torch.where(torch.isfinite(rounded), ulp, torch.zeros_like(ulp))
    return torch.where(same_class, ulp, torch.full_like(ulp, float("inf")))


@pytest.mark.parametrize("grad", GRADS)
def test_hardsigmoid_bw_exhaustive_bfloat16(grad, device):
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16)
    g = _grad_like(x, grad)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.hardsigmoid_bw(tt_g, tt_x)[0]).to(torch.bfloat16)

    ulp = torch.minimum(_pure_ulp(_reference(g, x, False), actual), _pure_ulp(_reference(g, x, True), actual))
    worst = ulp.argmax()
    assert ulp.max().item() < 1.0, (
        f"{(ulp >= 1.0).sum().item()} outputs at or beyond 1 ulp or of the wrong class; worst at "
        f"x={x.flatten()[worst].item()}, grad={g.flatten()[worst].item()}: "
        f"expected {_reference(g, x, False).flatten()[worst].item()}, got {actual.flatten()[worst].item()}"
    )
