# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Constants of BEVFormer's detection decoder shared by the reference and the TTNN port."""

# Channels of BEVFormer's 10-value box code (cx, cy, w, l, cz, h, sin, cos, vx, vy) that the
# reg branches use to refine the (x, y, z) reference point.
REG_XY = slice(0, 2)
REG_Z = slice(4, 5)
