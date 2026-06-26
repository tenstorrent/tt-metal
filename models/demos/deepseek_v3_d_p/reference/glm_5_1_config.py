# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GLM 5.1 Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for GLM-5.1.
"""

import types


class GLM51Config:
    """GLM 5.1 model dimensions."""

    # Core dimensions
    EMB_SIZE = 6144  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 12288  # Dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    NUM_SHARED_EXPERTS = 1
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1

    # Model architecture
    NUM_LAYERS = 78
    NUM_DENSE_LAYERS = 3  # first_k_dense_replace
    VOCAB_SIZE = 154880
    MAX_POSITION_EMBEDDINGS = 202752

    # MLA dimensions
    NUM_ATTENTION_HEADS = 64
    Q_LORA_RANK = 2048
    KV_LORA_RANK = 512
    QK_NOPE_HEAD_DIM = 192
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 256

    # Indexer / sparse attention
    INDEX_TOPK = 2048
    INDEX_HEAD_DIM = 128
    INDEX_N_HEADS = 32

    # Other
    RMS_NORM_EPS = 1e-5
    ROUTE_SCALE = 2.5
    ROPE_THETA = 1000000


def glm_model_args(max_seq: int = 8192):
    """GLM reference_cpu ModelArgs (the CPU MLACPU/IndexerCPU truth config).

    `max_seq_len == original_seq_len` disables BOTH the YaRN frequency scaling and the
    mscale² softmax correction → plain RoPE with scale = qk_head_dim**-0.5. `index_rope_interleave=True`
    flips IndexerCPU to interleaved RoPE (DeepSeek's indexer is non-interleaved). Imported lazily so the
    lightweight config module doesn't pull in torch / reference_cpu unless a CPU reference is built.
    """
    from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.model import ModelArgs

    return ModelArgs(
        max_batch_size=1,
        max_seq_len=max_seq,
        original_seq_len=max_seq,
        dim=GLM51Config.EMB_SIZE,
        n_heads=GLM51Config.NUM_ATTENTION_HEADS,
        q_lora_rank=GLM51Config.Q_LORA_RANK,
        kv_lora_rank=GLM51Config.KV_LORA_RANK,
        qk_nope_head_dim=GLM51Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=GLM51Config.QK_ROPE_HEAD_DIM,
        v_head_dim=GLM51Config.V_HEAD_DIM,
        rope_theta=GLM51Config.ROPE_THETA,
        rope_factor=1.0,
        mscale=1.0,
        index_n_heads=GLM51Config.INDEX_N_HEADS,
        index_head_dim=GLM51Config.INDEX_HEAD_DIM,
        index_topk=GLM51Config.INDEX_TOPK,
        index_rope_interleave=True,
    )


def glm_hf_config(max_seq: int = 8192):
    """HF-attribute-style config the unified ttMLA reads (GLM dims, no YaRN).

    Built directly rather than via AutoConfig: GLM's model_type (`glm_moe_dsa`) is not
    registered with transformers, so AutoConfig cannot load it. `rope_scaling.factor=1.0`
    disables both the YaRN frequency scaling and the mscale² softmax correction → plain
    RoPE at θ=1e6 with scale = qk_head_dim**-0.5. The four `index_*` attrs configure the
    DSA indexer (GLM's indexer RoPE is interleaved; DeepSeek's is not).
    """
    return types.SimpleNamespace(
        hidden_size=GLM51Config.EMB_SIZE,
        num_attention_heads=GLM51Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=GLM51Config.NUM_ATTENTION_HEADS,
        kv_lora_rank=GLM51Config.KV_LORA_RANK,
        q_lora_rank=GLM51Config.Q_LORA_RANK,
        qk_nope_head_dim=GLM51Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=GLM51Config.QK_ROPE_HEAD_DIM,
        v_head_dim=GLM51Config.V_HEAD_DIM,
        rms_norm_eps=GLM51Config.RMS_NORM_EPS,
        max_seq_len=max_seq,
        rope_theta=float(GLM51Config.ROPE_THETA),
        attention_bias=False,
        rope_scaling={
            "factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": max_seq,
        },
        index_n_heads=GLM51Config.INDEX_N_HEADS,
        index_head_dim=GLM51Config.INDEX_HEAD_DIM,
        index_topk=GLM51Config.INDEX_TOPK,
        index_rope_interleave=True,
    )
