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


def glm_hf_config(max_seq: int = 8192):
    """HF-attribute-style config the unified ttMLA reads (GLM dims, no YaRN).

    Built directly (not from transformers' GlmMoeDsaConfig, which IS available and AutoConfig-
    loadable as of transformers 5.10) because ttMLA/the cache-build path read a curated field set
    this namespace supplies but the stock GlmMoeDsaConfig does not: `rope_scaling` here is the
    DeepSeek-MLA shape (`factor`/`mscale`/`beta_*`) ttMLA indexes, plus top-level `rope_theta`,
    `max_seq_len`, `index_rope_interleave`, `first_k_dense_replace`, and `quantization_config`.
    (The HF *reference* model uses a real GlmMoeDsaConfig instead — see glm_reference_hf_config.)
    `rope_scaling.factor=1.0` disables both the YaRN frequency scaling and the mscale² softmax
    correction → plain RoPE at θ=1e6 with scale = qk_head_dim**-0.5. The four `index_*` attrs
    configure the DSA indexer (GLM's indexer RoPE is interleaved; DeepSeek's is not).
    """
    return types.SimpleNamespace(
        vocab_size=GLM51Config.VOCAB_SIZE,
        hidden_size=GLM51Config.EMB_SIZE,
        intermediate_size=GLM51Config.INTERMEDIATE_SIZE,  # dense-FFN (layers 0-2) hidden dim = 12288
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
        # MoE structure read by the pretrained cache-build path (load_and_compute_layer_by_layer):
        # first_k_dense_replace = number of leading dense-FFN layers before MoE begins (GLM: 3);
        # n_routed_experts = routed expert count (GLM: 256).
        first_k_dense_replace=GLM51Config.NUM_DENSE_LAYERS,
        n_routed_experts=GLM51Config.NUM_ROUTED_EXPERTS,
        # GLM-5.1 ships as FP8 block-quant (e4m3, [128,128] blocks). dequantize_state_dict fetches the
        # block shape UNCONDITIONALLY from here, so the pretrained cache-build path needs it present even
        # though the hand-built config is otherwise weight-free. Harmless for the bf16 checkout
        # (zai-org/GLM-5.1): no *_scale_inv tensors -> weights just pass through .to(bfloat16). Random-
        # weight tests never call dequant, so they're unaffected.
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )
