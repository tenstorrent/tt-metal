# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Configuration file for SSD512 - exports voc and coco configs."""

from .voc import voc, VOC_CLASSES
from .coco import coco

__all__ = ["voc", "coco", "VOC_CLASSES"]
