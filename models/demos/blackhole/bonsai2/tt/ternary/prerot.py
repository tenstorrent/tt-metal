# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hand-off of a pre-rotated decode activation: the layer's fused norm + rotation (rotate.NormRotation) registers the
rotated copies x' under the normalised tensor it returns; the ternary module that receives that tensor takes x'
instead of rotating it again."""

_PENDING = {}


def put(xn, xr):
    _PENDING[id(xn)] = (xn, xr)


def take(x):
    """x' for x, or None (removed: each x' is used once)."""
    hit = _PENDING.pop(id(x), None)
    return hit[1] if hit is not None and hit[0] is x else None
