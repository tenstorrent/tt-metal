# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `Qwen2_5_VLRMSNorm`: weight * x * rsqrt(mean(x^2) + eps), via ttnn.rms_norm.
Elementwise over the hidden dim -- replicated on every chip of a mesh, never sharded.
"""

from __future__ import annotations

from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtRMSNorm


def build(device, torch_module=None):
    return TtRMSNorm(device, torch_module)


def v_l_r_m_s_norm(device, torch_module=None):
    return build(device, torch_module)
