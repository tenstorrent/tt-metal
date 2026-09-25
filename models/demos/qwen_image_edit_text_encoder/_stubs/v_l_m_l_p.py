# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the vision `Qwen2_5_VLMLP` (`model.visual.blocks[i].mlp`).
gate/up column-parallel (intermediate zero-padded to a multiple of TP*32), down row-parallel +
all_reduce. See encoder_stack.TtVisionMLP.
"""

from __future__ import annotations

from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtVisionMLP
from models.demos.qwen_image_edit_text_encoder._stubs.layer import TtTextMLP


def build(device, torch_module=None):
    has_bias = torch_module.gate_proj.bias is not None or torch_module.up_proj.bias is not None
    return TtVisionMLP(device, torch_module) if has_bias else TtTextMLP(device, torch_module)


def v_l_m_l_p(device, torch_module=None):
    return build(device, torch_module)
