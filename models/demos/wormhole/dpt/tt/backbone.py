"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

# Delegate to experimental implementation for now
from models.experimental.dpt_large.vit_backbone import DPTViTBackboneTTNN as _Backbone


def build_backbone(config=None, hf_model=None, pretrained=True, device="cpu", tt_layer_cfg=None):
    return _Backbone(config=config, hf_model=hf_model, pretrained=pretrained, device=device, tt_layer_cfg=tt_layer_cfg)
