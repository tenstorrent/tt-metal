# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
AutoAnchor utils
"""

from loguru import logger
from models.experimental.yolov5.reference.utils.general import colorstr

PREFIX = colorstr("AutoAnchor: ")


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        logger.info(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
