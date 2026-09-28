# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash-RL text decoder on the 2x2 mesh: module builders shared by the component hooks and (later) the
all-device model, so both paths construct identical modules. Grows one validated component at a time."""

from __future__ import annotations

# Norm steps -> checkpoint weight name (under model.layers.<i>.).
NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}


def build_norm(mesh, loader, layer: int, step: str, eps: float = 1e-6):
    """TtRMSNorm (replicated on all 4 chips, HiFi4 + fp32 acc, plain w) for one layer's norm step."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import TtRMSNorm

    return TtRMSNorm(mesh, loader.get(f"model.layers.{layer}.{NORM_WEIGHTS[step]}"), eps=eps)
