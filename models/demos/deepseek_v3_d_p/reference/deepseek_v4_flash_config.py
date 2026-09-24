# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V4 Flash Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for DeepSeek-V4-Flash.
"""


class DeepSeekV4FlashConfig:
    """DeepSeek V4 Flash model dimensions."""

    # Core dimensions
    EMB_SIZE = 4096  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    HEAD_DIM = 512

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 6
    NUM_SHARED_EXPERTS = 1
    # V4 drops V3's expert-group routing: a single group means the gate collapses to a plain top-k.
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    # V4 replaces V3/Kimi's sigmoid router affinity with sqrt(softplus(.)).
    SCORE_FUNC = "sqrtsoftplus"

    # Model architecture
    NUM_LAYERS = 43
    NUM_HASH_LAYERS = 3
    VOCAB_SIZE = 129280
    SLIDING_WINDOW = 128

    # MLA dimensions
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 1
    Q_LORA_RANK = 1024
    O_LORA_RANK = 1024
    O_GROUPS = 8
    QK_ROPE_HEAD_DIM = 64

    # Indexer / sparse attention (NSA-style)
    INDEX_N_HEADS = 64
    INDEX_HEAD_DIM = 128
    INDEX_TOPK = 512
    # Compressed attention config
    COMPRESS_RATES = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
    COMPRESS_ROPE_THETA = 160000.0
    HC_MULT = 4
    HC_SINKHORN_ITERS = 20
    HC_EPS = 1.0e-6

    # Other
    RMS_NORM_EPS = 1e-6
    ROUTE_SCALE = 1.5
    ROPE_THETA = 10000
    SWIGLU_LIMIT = 10.0
    MAX_POSITION_EMBEDDINGS = 1048576


def flash_compress_ratios(num_layers: int = DeepSeekV4FlashConfig.NUM_LAYERS) -> list:
    """Per-layer compression ratios exactly as DeepSeek-V4-Flash's ``config.json`` states them: layers 0 and 1
    sliding-window (0), then even layers CSA (4) and odd layers HCA (128). 43 layers -> 2 SWA, 21 CSA, 20 HCA.
    ``DeepseekV4Config``'s own default schedule (HCA-first interleave) is NOT Flash's; always pass these."""
    return [0 if i < 2 else (4 if i % 2 == 0 else 128) for i in range(int(num_layers))]


def deepseek_v4_flash_hf_config(max_seq: int = 8192, num_hidden_layers: int = DeepSeekV4FlashConfig.NUM_LAYERS):
    """Hand-built ``DeepseekV4Config`` for DeepSeek-V4-Flash (the ``deepseek_v4`` model_type needs the
    reference package's config class; this mirrors ``glm_5_2_hf_config``). Every field comes from
    ``DeepSeekV4FlashConfig`` or the checkpoint's config.json (rope_scaling yarn factor 16 over 65536
    original positions, compress rope theta 160000). ``max_seq_len`` is set for the prefill runner, which
    overwrites it after. Eager attention only: V4's sdpa interface drops the sinks."""
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    m = DeepSeekV4FlashConfig
    cfg = DeepseekV4Config(
        vocab_size=m.VOCAB_SIZE,
        hidden_size=m.EMB_SIZE,
        moe_intermediate_size=m.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=int(num_hidden_layers),
        num_attention_heads=m.NUM_ATTENTION_HEADS,
        num_key_value_heads=m.NUM_KEY_VALUE_HEADS,
        head_dim=m.HEAD_DIM,
        q_lora_rank=m.Q_LORA_RANK,
        o_lora_rank=m.O_LORA_RANK,
        o_groups=m.O_GROUPS,
        num_experts_per_tok=m.NUM_EXPERTS_PER_TOKEN,
        n_routed_experts=m.NUM_ROUTED_EXPERTS,
        n_shared_experts=m.NUM_SHARED_EXPERTS,
        scoring_func=m.SCORE_FUNC,
        routed_scaling_factor=m.ROUTE_SCALE,
        max_position_embeddings=m.MAX_POSITION_EMBEDDINGS,
        rope_theta=float(m.ROPE_THETA),
        compress_rates=dict(m.COMPRESS_RATES),
        compress_rope_theta=float(m.COMPRESS_ROPE_THETA),
        hc_mult=m.HC_MULT,
        hc_sinkhorn_iters=m.HC_SINKHORN_ITERS,
        hc_eps=m.HC_EPS,
        swiglu_limit=m.SWIGLU_LIMIT,
        sliding_window=m.SLIDING_WINDOW,
        index_n_heads=m.INDEX_N_HEADS,
        index_head_dim=m.INDEX_HEAD_DIM,
        index_topk=m.INDEX_TOPK,
        rms_norm_eps=m.RMS_NORM_EPS,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        # legacy kwargs the config folds into layer_types / mlp_layer_types / qk_rope_head_dim
        compress_ratios=flash_compress_ratios(num_hidden_layers),
        num_hash_layers=m.NUM_HASH_LAYERS,
        qk_rope_head_dim=m.QK_ROPE_HEAD_DIM,
    )
    cfg._attn_implementation = "eager"
    cfg.max_seq_len = int(max_seq)
    return cfg
