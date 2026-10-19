# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Constants of BEVFormer's detection decoder shared by the reference and the TTNN port."""

# BEVFormer's box code layout: (cx, cy, w, l, cz, h, sin, cos, vx, vy), the sizes as logs.
# Box codes, the reg branches' raw output: cx, cy and cz are logit offsets, which the decoder
# adds to its reference points' logits.
# Box predictions, the head's output: cx, cy and cz are the refined centers in metres.
# The other channels are the same in both.
CODE_SIZE = 10
CODE_XY = slice(0, 2)
CODE_WL = slice(2, 4)
CODE_Z = slice(4, 5)
CODE_H = slice(5, 6)
CODE_SIN = slice(6, 7)
CODE_COS = slice(7, 8)
CODE_VELOCITY = slice(8, 10)

_CENTERS = [*range(CODE_SIZE)][CODE_XY] + [*range(CODE_SIZE)][CODE_Z]
# The TTNN port's channel order for box codes: the centers first, (cx, cy, cz, w, l, h, sin, cos,
# vx, vy), so its decoder takes the center offsets in one slice. Its reg branches emit it, from
# the last Linear's rows in this order.
CENTERS_FIRST = _CENTERS + [channel for channel in range(CODE_SIZE) if channel not in _CENTERS]
