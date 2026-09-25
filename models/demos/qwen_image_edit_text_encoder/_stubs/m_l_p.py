# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of a Qwen2.5-VL SwiGLU MLP: down(silu(gate(x)) * up(x)).

Covers the text `Qwen2MLP` (no biases, 3584 -> 18944) and the vision `Qwen2_5_VLMLP` (biases,
1280 -> 3420, intermediate zero-padded to a multiple of TP*32). gate/up are column-parallel, down is
row-parallel followed by an all_reduce over the TP axis; any DP axis replicates.
"""

from __future__ import annotations

from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtVisionMLP
from models.demos.qwen_image_edit_text_encoder._stubs.layer import TtTextMLP


def _has_bias(torch_module):
    return any(getattr(p, "bias", None) is not None for p in (torch_module.gate_proj, torch_module.up_proj))


def build(device, torch_module=None):
    return TtVisionMLP(device, torch_module) if _has_bias(torch_module) else TtTextMLP(device, torch_module)


def m_l_p(device, torch_module=None):
    return build(device, torch_module)
