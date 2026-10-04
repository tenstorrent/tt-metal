# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive BF16 accuracy of ttnn.logit.

Every BF16 bit pattern is an input. The reference is torch.logit(x) in float64, on its real domain x ≥ 0 and x ≤ 1 and NaN elsewhere,
rounded once to BF16 with subnormal results flushed to zero. The SFPU may read a subnormal input
as zero, so torch is evaluated both at the input and at the input with subnormals flushed, and an
output may match either. The BF16 pack stores NaN as +inf and -0 as +0, so classes are compared as
stored. Each output must have the reference's class and a pure ULP error, |reference - output| /
ulp(rounded reference), below 1.
The inputs in DECLARED are stored as the class given there, the first row that holds; where
that differs from torch, it is the class the TT-NN op this kernel replaces stores.
"""

import torch
import ttnn

from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from tests.ttnn.utils_for_testing import generate_all_bfloat16_bitpatterns

SMALLEST_NORMAL = 2.0**-126
CLASS_CODES = {"inf": 0, "-inf": 1, "zero": 2, "finite": 3}

# Per board: (inputs, as a torch expression of the input x or of daz, the input as the SFPU reads it
# with subnormals as zero, and the class their outputs are stored as).
DECLARED = {
    "blackhole": [
        ("torch.isfinite(daz) & (daz >= 8.507059173023462e+37)", "-inf"),  # finite x >= 8.507059e+37
        ("torch.isfinite(daz) & (daz <= -8.507059173023462e+37)", "-inf"),  # finite x <= -8.507059e+37
        ("torch.isfinite(daz) & (daz < 0.0)", "inf"),  # finite x < 0
        ("torch.isfinite(daz) & (daz > 1.0)", "inf"),  # finite x > 1
        ("torch.isfinite(daz) & (daz <= 0.0)", "-inf"),  # finite x <= 0
        ("torch.isfinite(daz) & (daz >= 1.0)", "inf"),  # finite x >= 1
    ],
    "wormhole_b0": [
        ("torch.isfinite(daz) & (daz >= 8.573520572812707e+37)", "-inf"),  # finite x >= 8.573521e+37
        ("torch.isfinite(daz) & (daz <= -8.573520572812707e+37)", "-inf"),  # finite x <= -8.573521e+37
        ("torch.isfinite(daz) & (daz < 0.0)", "inf"),  # finite x < 0
        ("torch.isfinite(daz) & (daz > 1.0)", "inf"),  # finite x > 1
        ("torch.isfinite(daz) & (daz <= 0.0)", "-inf"),  # finite x <= 0
        ("torch.isfinite(daz) & (daz >= 1.0)", "inf"),  # finite x >= 1
    ],
}


def _flush(t):
    return torch.where(t.abs() < SMALLEST_NORMAL, torch.zeros_like(t), t)


def _reference(x):
    domain = (x >= 0.0) & (x <= 1.0)
    result = torch.full_like(x, float("nan"))
    result[domain] = torch.logit(x[domain])
    return result


def _round_to_bfloat16(t):
    """Round float64 to BF16 once (round-to-odd into float32, then nearest-even), then flush."""
    f32 = t.to(torch.float32)
    back = f32.to(torch.float64)
    inexact = torch.isfinite(t) & (back != t)
    bits = f32.view(torch.int32) - (inexact & (back.abs() > t.abs())).to(torch.int32)
    bits = bits | inexact.to(torch.int32)
    return _flush(bits.view(torch.float32).to(torch.bfloat16))


def _stored_classes(t):
    """The CLASS_CODES of each value as stored: NaN as +inf, either zero as zero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, CLASS_CODES["finite"], dtype=torch.int8)
    classes[t == 0] = CLASS_CODES["zero"]
    classes[(t == float("inf")) | torch.isnan(t)] = CLASS_CODES["inf"]
    classes[t == float("-inf")] = CLASS_CODES["-inf"]
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


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
def test_logit_exhaustive_bfloat16(device):
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.logit(tt_x)).to(torch.bfloat16)

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    x64 = x.to(torch.float64)
    expected = torch.full(x.shape, -1, dtype=torch.int8)
    for inputs, stored in DECLARED[board]:
        lanes = eval(inputs, {"torch": torch, "x": x64, "daz": _flush(x64), "SMALLEST_NORMAL": SMALLEST_NORMAL})
        expected = torch.where(lanes & (expected < 0), CLASS_CODES[stored], expected)
    declared = expected >= 0
    wrong = declared & (_stored_classes(actual) != expected)
    assert not wrong.any(), (
        f"{wrong.sum().item()} declared inputs of the wrong class; first x={x[wrong][0].item()}, "
        f"got {actual[wrong][0].item()}"
    )

    ulp = torch.minimum(_pure_ulp(_reference(x64), actual), _pure_ulp(_reference(_flush(x64)), actual))
    ulp = torch.where(declared, torch.zeros_like(ulp), ulp)
    worst = ulp.argmax()
    assert ulp.max().item() < 1.0, (
        f"{(ulp >= 1.0).sum().item()} outputs at or beyond 1 ulp or of the wrong class; worst at "
        f"x={x.flatten()[worst].item()}: expected {_reference(x64).flatten()[worst].item()}, "
        f"got {actual.flatten()[worst].item()}"
    )
