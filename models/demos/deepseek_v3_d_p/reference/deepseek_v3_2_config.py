# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V3.2-Exp model configuration.

Single source of truth for the V3.2 HF-attribute config the unified ttMLA reads. Built directly
rather than via AutoConfig: V3.2's `model_type` is `deepseek_v32` (architecture
`DeepseekV32ForCausalLM`) and needs trust_remote_code, which we avoid here — and, unlike a borrowed
DeepSeek-V3 / R1 config, this one actually carries the DSA fields so the sparse resolver detects it.

V3.2 shares every MLA dimension and the YaRN RoPE with DeepSeek-V3 / R1, and adds the lightning
indexer (index_n_heads / index_head_dim / index_topk; non-interleaved indexer RoPE). Values are from
the HF config.json: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json
"""

import types


class DeepseekV32Config:
    """DeepSeek-V3.2-Exp model dimensions (from HF config.json)."""

    # Core dimensions
    EMB_SIZE = 7168  # hidden_size
    INTERMEDIATE_SIZE = 18432  # dense FFN hidden dimension
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    NUM_SHARED_EXPERTS = 1

    # Model architecture
    NUM_LAYERS = 61
    NUM_DENSE_LAYERS = 3  # first_k_dense_replace
    VOCAB_SIZE = 129280
    MAX_POSITION_EMBEDDINGS = 163840

    # MLA dimensions (identical to DeepSeek-V3 / R1)
    NUM_ATTENTION_HEADS = 128
    Q_LORA_RANK = 1536
    KV_LORA_RANK = 512
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    V_HEAD_DIM = 128

    # Indexer / sparse attention (DSA)
    INDEX_TOPK = 2048
    INDEX_HEAD_DIM = 128
    INDEX_N_HEADS = 64
    INDEX_ROPE_INTERLEAVE = False  # DeepSeek indexer RoPE is non-interleaved (rotate_half)

    # RoPE / YaRN (factor > 1 → YaRN active, same as DeepSeek-V3 / R1)
    RMS_NORM_EPS = 1e-6
    ROUTE_SCALE = 2.5
    ROPE_THETA = 10000.0
    ROPE_FACTOR = 40
    ORIGINAL_SEQ_LEN = 4096  # rope_scaling.original_max_position_embeddings
    BETA_FAST = 32
    BETA_SLOW = 1
    MSCALE = 1.0
    MSCALE_ALL_DIM = 1.0

    # Other
    INITIALIZER_RANGE = 0.02


def deepseek_v32_hf_config(max_seq: int = 16384):
    """HF-attribute-style config the unified ttMLA reads (DeepSeek-V3.2 dims + YaRN + DSA indexer).

    Mirrors `glm_hf_config()`. YaRN is active (`rope_factor=40 > 1`): the device runs YaRN with
    `original_max_position_embeddings=4096`, identical to DeepSeek-V3 / R1. The four `index_*` attrs
    configure the DSA lightning indexer (non-interleaved RoPE, vs GLM's interleaved). `has_indexer=True`
    marks the layer sparse so the cache/loading resolver (`resolve_has_indexer`) detects it without
    relying on host weights or a built cache. The sparse-MLA CPU reference derives its `ModelArgs` from
    this same config (reference.cpu_deepseek_v32), so the device and the truth share one source of dims.
    """
    return types.SimpleNamespace(
        hidden_size=DeepseekV32Config.EMB_SIZE,
        num_attention_heads=DeepseekV32Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=DeepseekV32Config.NUM_ATTENTION_HEADS,
        kv_lora_rank=DeepseekV32Config.KV_LORA_RANK,
        q_lora_rank=DeepseekV32Config.Q_LORA_RANK,
        qk_nope_head_dim=DeepseekV32Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=DeepseekV32Config.QK_ROPE_HEAD_DIM,
        v_head_dim=DeepseekV32Config.V_HEAD_DIM,
        rms_norm_eps=DeepseekV32Config.RMS_NORM_EPS,
        max_seq_len=max_seq,
        rope_theta=float(DeepseekV32Config.ROPE_THETA),
        attention_bias=False,
        initializer_range=DeepseekV32Config.INITIALIZER_RANGE,
        rope_scaling={
            "factor": DeepseekV32Config.ROPE_FACTOR,
            "mscale": DeepseekV32Config.MSCALE,
            "mscale_all_dim": DeepseekV32Config.MSCALE_ALL_DIM,
            "beta_fast": DeepseekV32Config.BETA_FAST,
            "beta_slow": DeepseekV32Config.BETA_SLOW,
            "original_max_position_embeddings": DeepseekV32Config.ORIGINAL_SEQ_LEN,
        },
        index_n_heads=DeepseekV32Config.INDEX_N_HEADS,
        index_head_dim=DeepseekV32Config.INDEX_HEAD_DIM,
        index_topk=DeepseekV32Config.INDEX_TOPK,
        index_rope_interleave=DeepseekV32Config.INDEX_ROPE_INTERLEAVE,
        has_indexer=True,
    )
