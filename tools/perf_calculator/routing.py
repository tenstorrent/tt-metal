# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Directed torus routing shared by transport estimators."""


def links(a, b, noc, grid):
    x, y = a
    dx, dy = b
    out = []
    axes = [0, 1] if noc == 0 else [1, 0]
    for axis in axes:
        while (x if axis == 0 else y) != (dx if axis == 0 else dy):
            nx, ny = x, y
            if axis == 0:
                nx = (x + (1 if noc == 0 else -1)) % grid[0]
            else:
                ny = (y + (1 if noc == 0 else -1)) % grid[1]
            out.append(("link", noc, x, y, nx, ny))
            x, y = nx, ny
    return out
