# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hand-off of a pre-rotated decode activation: the layer's fused norm + rotation (rotate.NormRotation) registers the
rotated copies x' under the normalised tensor it returns; the ternary module that receives that tensor takes x'
instead of rotating it again.

Residual hand-off (decode): the layer registers the residual of its next add (put_residual); the output projection
that produces the add's other operand takes it (take_residual) and adds it in its output stage
(decode_linear3.py residual=); the layer then asks residual_added() and skips its own add."""

_PENDING = {}


def put(xn, xr):
    _PENDING[id(xn)] = (xn, xr)


def take(x):
    """x' for x, or None (removed: each x' is used once)."""
    hit = _PENDING.pop(id(x), None)
    return hit[1] if hit is not None and hit[0] is x else None


_RESID = []


def put_residual(r):
    import ttnn

    _RESID.clear()
    if r.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:  # the reader's accessor: interleaved
        _RESID.extend([r, False])


def take_residual():
    """The pending residual (once), or None."""
    if _RESID and not _RESID[1]:
        _RESID[1] = True
        return _RESID[0]
    return None


def residual_added():
    """True if an output projection took the pending residual; clears it either way."""
    added = bool(_RESID) and _RESID[1]
    _RESID.clear()
    return added


_XNT = {}


def put_xnt(xn, xnt):
    """The normalised activation's blocks tiled (rotate.NormRotation), for the GDN step's a / b dot products."""
    _XNT.clear()
    _XNT[id(xn)] = (xn, xnt)


def take_xnt(x):
    """xnt for x, or None (removed: used once)."""
    hit = _XNT.pop(id(x), None)
    return hit[1] if hit is not None and hit[0] is x else None


_NORM = {}


def put_norm(x, nrot):
    """A decode norm + rotation not run yet: the ternary module that receives x runs it inside its first linear's op
    (decode_linear3 rotated() with the NormRotation)."""
    _NORM.clear()
    _NORM[id(x)] = (x, nrot)


def take_norm(x):
    hit = _NORM.pop(id(x), None)
    return hit[1] if hit is not None and hit[0] is x else None
