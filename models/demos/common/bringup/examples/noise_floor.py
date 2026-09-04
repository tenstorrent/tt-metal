# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The noise-floor primitive: how to set a PCC threshold that means something.

Lifted verbatim from a completed bring-up, where it was the single highest-value
artefact produced. The full method is section 2 of BRINGUP_RECIPE.md.

Do NOT gate against another implementation's published PCC. Its reference may be
computed at a different precision -- an HF model loaded at torch_dtype bfloat16
yields a reference that shares the device's own rounding and therefore reports a
flattered number -- so the two figures are not comparable. Do NOT pick the input
distribution that passes either: the floor is distribution-stable, so that is
threshold-shopping.

Instead: quantise every tensor the device *stores* to the device dtype, do all the
remaining arithmetic in fp32 on the host, and PCC that against the fp32 reference.
That is the best any correct kernel can do. Record measured, floor, and the ratio
between their errors, and gate on the ratio -- 20x off the floor is a finding even
when the absolute PCC looks pretty.

Standing caveat: a storage-dtype floor does not model a fused kernel's interior. In
the run this came from, SDPA alone sat 71x off its floor and accounted for the whole
attention-block gap, while every stage written by hand sat at 1.0-1.5x. Attribute a
large ratio to a named stage before treating it as a bug.

Keep exactly ONE definition of these two functions per package. The run this came
from briefly had two copies and had to reconcile them; a test now asserts single
definition by object identity.
"""


def quantize_like_device(t, dtype):
    """Round `t` to exactly the values the device will hold, via ttnn, and return fp32.

    Host-only (no `device=` argument), so this is a pure quantiser and never a compute path.
    Reproduces `bfloat8_b`'s shared-exponent tile blocking exactly — which no hand-rolled torch
    emulation does. Requires a 4D, tile-shaped tensor for TILE_LAYOUT.

    This is the primitive the noise-floor method is built on: quantise every tensor the
    device *stores*, do all the arithmetic in fp32, and PCC that against the fp32 reference. The
    result is implementation-independent and distribution-stable, unlike a PCC copied from another
    implementation whose reference shares the device's own rounding (`BRINGUP_RECIPE.md` section 2.1).
    """
    import ttnn

    assert t.dim() == 4, f"quantize_like_device expects a 4D tensor, got {tuple(t.shape)}"
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def err_ratio(measured: float, floor: float) -> float:
    """`(1 - measured) / (1 - floor)` — the measured error in units of the noise floor's.

    `1.0` means the module is exactly at the floor. `20x+` off the floor is a finding even when the
    absolute PCC looks pretty (`BRINGUP_RECIPE.md` section 2.2) — and section 2.3 is the standing caveat: a
    storage-dtype floor does not model a fused kernel's interior, so a large ratio must be
    attributed to a named stage before it is treated as a bug.
    """
    return float("inf") if floor >= 1.0 else (1.0 - float(measured)) / (1.0 - float(floor))
