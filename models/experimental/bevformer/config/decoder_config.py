# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Constants of BEVFormer's detection decoder shared by the reference and the TTNN port."""

# BEVFormer's box code, (cx, cy, w, l, cz, h, sin, cos, vx, vy) with the sizes as logs. In
# the reg branches' raw output, the box codes, the centre channels are logit offsets the
# decoder adds to its reference points; the head's box predictions carry the refined
# centres in metres instead.
CODE_SIZE = 10
CODE_XY = slice(0, 2)
CODE_WL = slice(2, 4)
CODE_Z = slice(4, 5)
CODE_H = slice(5, 6)
CODE_SIN = slice(6, 7)
CODE_COS = slice(7, 8)
CODE_VELOCITY = slice(8, 10)
