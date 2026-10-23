# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Small helpers around `ttnn.linear` for VADv2.

`ttnn.linear`'s default matmul program-config heuristic chooses the
core grid from the M dim it sees. For inputs of shape
`(*, batch, M, K)` where `batch > 1`, M is read as the second-to-last
dim alone — often 1 or 32 (a single tile) — so the heuristic picks
N-only parallelisation on a handful of cores.

`linear_flatten_batch` reshapes the input so all leading dims are
folded into a single M axis, letting the heuristic dispatch across
many more cores. It restores the original layout downstream of the
linear.
"""

import ttnn


def linear_flatten_batch(x, weight, bias=None, **kwargs):
    """ttnn.linear with leading batch dims flattened into M.

    Equivalent to `ttnn.linear(x, weight, bias=bias)` numerically; the
    only behavioural difference is the M dim the matmul heuristic sees,
    which controls how many cores the kernel uses.

    No-ops the reshape when the input is already 2-D or when flattening
    wouldn't increase the M row count.
    """
    orig = [int(d) for d in x.shape]
    if len(orig) < 3:
        return ttnn.linear(x, weight, bias=bias, **kwargs)
    flat_m = 1
    for d in orig[:-1]:
        flat_m *= d
    if flat_m <= orig[-2]:
        return ttnn.linear(x, weight, bias=bias, **kwargs)
    x_flat = ttnn.reshape(x, (1, 1, flat_m, orig[-1]))
    y_flat = ttnn.linear(x_flat, weight, bias=bias, **kwargs)
    out_shape = tuple(orig[:-1]) + (int(y_flat.shape[-1]),)
    return ttnn.reshape(y_flat, out_shape)
