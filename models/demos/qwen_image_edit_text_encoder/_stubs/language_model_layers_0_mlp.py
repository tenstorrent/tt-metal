# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the Qwen2.5-VL text MLP (`language_model.layers.0.mlp`).

SwiGLU down(silu(gate(x)) * up(x)), 3584 -> 18944 -> 3584, no biases. gate/up are column-parallel
(intermediate split across the TP axis), down is row-parallel followed by an all_reduce over the TP
axis; any DP axis replicates. Same scheme as m_l_p.py.
"""

from __future__ import annotations

from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtVisionMLP
from models.demos.qwen_image_edit_text_encoder._stubs.layer import TtTextMLP


def _has_bias(torch_module):
    return any(getattr(p, "bias", None) is not None for p in (torch_module.gate_proj, torch_module.up_proj))


def build(device, torch_module=None, pair=None):
    if _has_bias(torch_module):
        assert pair is None, "row-staged pairs are only used for the text MLP"
        return TtVisionMLP(device, torch_module)
    return TtTextMLP(device, torch_module, pair=pair)


def language_model_layers_0_mlp(device, torch_module=None):
    return build(device, torch_module)
