# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Constants of BEVFormer's detection decoder shared by the reference and the TTNN port."""

# BEVFormer's box code, the reg branches' output: (cx, cy, w, l, cz, h, sin, cos, vx, vy),
# centres normalized to [0, 1] (before the head scales them) and sizes as logs.
CODE_SIZE = 10
# The centre channels, which the decoder adds to the reference points' logits.
REG_XY = slice(0, 2)
REG_Z = slice(4, 5)
CODE_WL = slice(2, 4)
CODE_H = slice(5, 6)
CODE_SIN = slice(6, 7)
CODE_COS = slice(7, 8)
CODE_VELOCITY = slice(8, 10)
