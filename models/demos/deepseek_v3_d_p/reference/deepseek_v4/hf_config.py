# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A ``DeepseekV4Config`` built from a V4 model-config class, for runs without a checkpoint config.json."""

from __future__ import annotations

from typing import Optional

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


def v4_hf_config(model_cfg: type, num_hidden_layers: Optional[int] = None, max_seq_len: Optional[int] = None):
    """``model_cfg``'s dimensions and layer schedule as a ``DeepseekV4Config``, truncated to ``num_hidden_layers``.

    q_lora_rank, o_groups and rope_parameters are explicit because DeepseekV4Config's defaults are
    Flash's: a Pro config that left them out would build Pro widths with Flash's latent, grouping and
    an unscaled compressed rope.
    """
    m = model_cfg
    n = m.NUM_LAYERS if num_hidden_layers is None else num_hidden_layers
    assert 0 < n <= m.NUM_LAYERS, f"{m.__name__} has {m.NUM_LAYERS} layers, asked for {n}"
    cfg = DeepseekV4Config(
        hidden_size=m.EMB_SIZE,
        head_dim=m.HEAD_DIM,
        num_attention_heads=m.NUM_ATTENTION_HEADS,
        q_lora_rank=m.Q_LORA_RANK,
        o_groups=m.O_GROUPS,
        num_hidden_layers=n,
        compress_ratios=list(m.COMPRESS_RATIOS[:n]),
        compress_rates=dict(m.COMPRESS_RATES),
        compress_rope_theta=m.COMPRESS_ROPE_THETA,
        rope_parameters={
            "type": "yarn",
            "factor": m.ROPE_SCALING_FACTOR,
            "original_max_position_embeddings": m.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": m.ROPE_SCALING_BETA_FAST,
            "beta_slow": m.ROPE_SCALING_BETA_SLOW,
        },
        num_hash_layers=m.NUM_HASH_LAYERS,
        rms_norm_eps=m.RMS_NORM_EPS,
        swiglu_limit=m.SWIGLU_LIMIT,
        n_routed_experts=m.NUM_ROUTED_EXPERTS,
        num_experts_per_tok=m.NUM_EXPERTS_PER_TOKEN,
        intermediate_size=m.MOE_INTERMEDIATE_SIZE,
        routed_scaling_factor=m.ROUTE_SCALE,
        vocab_size=m.VOCAB_SIZE,
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    if max_seq_len is not None:
        cfg.max_seq_len = max_seq_len
    return cfg
