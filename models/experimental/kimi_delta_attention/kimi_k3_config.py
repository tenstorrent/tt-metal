# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 text/KDA configuration from the pinned Hugging Face checkpoint."""

from __future__ import annotations

from typing import Any

from models.experimental.kimi_delta_attention.config import KDAConfig, KDAProgramConfig


class KimiK3Config:
    """Kimi-K3 text-tower and KDA constants."""

    HF_REPO_ID = "moonshotai/Kimi-K3"
    HF_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    FIRST_KDA_LAYER = 1

    HIDDEN_SIZE = 7168
    NUM_HIDDEN_LAYERS = 93
    NUM_ATTENTION_HEADS = 96
    KDA_NUM_HEADS = 96
    KDA_HEAD_DIM = 128
    KDA_CONV_KERNEL_SIZE = 4
    KDA_SUMMARY_GROUP_CHUNKS = 20
    KDA_OUTPUT_PROJECTION_OUT_BLOCK_W = 4
    KDA_USE_FULL_RANK_GATE = True
    KDA_GATE_LOWER_BOUND = -5.0
    RMS_NORM_EPS = 1e-5


def kimi_k3_model_config() -> dict[str, Any]:
    """Return the HF JSON-shaped fields consumed by :class:`KDAConfig`."""
    return {
        "hidden_size": KimiK3Config.HIDDEN_SIZE,
        "num_hidden_layers": KimiK3Config.NUM_HIDDEN_LAYERS,
        "num_attention_heads": KimiK3Config.NUM_ATTENTION_HEADS,
        "rms_norm_eps": KimiK3Config.RMS_NORM_EPS,
        "linear_attn_config": {
            "num_heads": KimiK3Config.KDA_NUM_HEADS,
            "head_dim": KimiK3Config.KDA_HEAD_DIM,
            "short_conv_kernel_size": KimiK3Config.KDA_CONV_KERNEL_SIZE,
            "use_full_rank_gate": KimiK3Config.KDA_USE_FULL_RANK_GATE,
            "gate_lower_bound": KimiK3Config.KDA_GATE_LOWER_BOUND,
        },
    }


def kimi_k3_kda_config() -> KDAConfig:
    """Build the TT KDA configuration from the pinned Kimi-K3 constants."""
    return KDAConfig.from_model_config(kimi_k3_model_config())


def kimi_k3_program_config() -> KDAProgramConfig:
    """Return measured K3 TP8 device-program tuning."""
    return KDAProgramConfig(
        # 160 local chunks / 20 = 8 groups/head; TP8 uses 12 * 8 = 96 owners.
        summary_group_chunks=KimiK3Config.KDA_SUMMARY_GROUP_CHUNKS,
        output_projection_out_block_w=KimiK3Config.KDA_OUTPUT_PROJECTION_OUT_BLOCK_W,
    )
