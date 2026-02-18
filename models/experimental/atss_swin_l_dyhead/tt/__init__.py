# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN ATSS detection model (Swin-L + FPN + DyHead + ATSS Head).

Components:
  tt_swin_backbone.py  — Swin-L backbone (wraps models/experimental/swin_l/)
  tt_fpn.py            — Feature Pyramid Network
  tt_atss_head.py      — ATSS detection head
  tt_atss_model.py     — Full model assembly (hybrid TTNN + PyTorch DyHead)
  weight_loading.py    — FPN / DyHead / ATSS Head weight loading
"""

from models.experimental.atss_swin_l_dyhead.tt.tt_swin_backbone import build_atss_backbone
from models.experimental.atss_swin_l_dyhead.tt.tt_fpn import TtFPN
from models.experimental.atss_swin_l_dyhead.tt.tt_atss_head import TtATSSHead
from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel
from models.experimental.atss_swin_l_dyhead.tt.weight_loading import (
    load_fpn_weights,
    load_atss_head_weights,
    load_dyhead_weights,
)
