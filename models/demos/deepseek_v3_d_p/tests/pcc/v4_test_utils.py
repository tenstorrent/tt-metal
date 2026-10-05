# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the DeepSeek-V4 attention PCC and perf tests (HCA, CSA)."""

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

V4_SEED = 42


def v4_reference_config(model_config, num_hidden_layers=4):
    """Reference config from one variant's dimension constants.

    ``q_lora_rank`` and ``o_groups`` are passed explicitly because DeepseekV4Config's defaults happen to
    be Flash's values -- a Pro config that left them out would build Pro widths with Flash's latent and
    grouping, and the reference would agree with it, so PCC would pass on the wrong model."""
    m = model_config
    cfg = DeepseekV4Config(
        hidden_size=m.EMB_SIZE,
        head_dim=m.HEAD_DIM,
        num_attention_heads=m.NUM_ATTENTION_HEADS,
        q_lora_rank=m.Q_LORA_RANK,
        o_groups=m.O_GROUPS,
        num_hidden_layers=num_hidden_layers,
        compress_rates=dict(m.COMPRESS_RATES),
        compress_rope_theta=m.COMPRESS_ROPE_THETA,
        rms_norm_eps=m.RMS_NORM_EPS,
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    return cfg
