# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared cached zero-pad buffers for the conv stubs.

`ttnn.zeros(...)` zero-fills a fresh buffer via a HOST WRITE, which is illegal while a trace is
being captured ("Writes are not supported during trace capture"). The conv/tokenizer stubs use
zero tensors only as CONSTANT causal-padding blocks whose shapes are fixed for a given pipeline,
so we create each one ONCE (on the first, eager, warm-up call — before any trace capture) and
reuse the same resident tensor afterwards. The value is always zero, so the numerics are identical
to calling `ttnn.zeros` every time; the only change is that `forward` becomes host-op-free after
warm-up, which lets the padded convolution be captured into a trace.
"""

from __future__ import annotations

import ttnn


def cached_zeros(cache, shape, dtype, layout, device):
    """Return a resident all-zeros tensor of `shape`, creating it once and caching by shape."""
    key = (tuple(int(s) for s in shape), dtype, layout)
    t = cache.get(key)
    if t is None:
        t = ttnn.zeros(shape, dtype=dtype, layout=layout, device=device)
        cache[key] = t
    return t
