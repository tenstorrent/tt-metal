# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Architecture constants for Mistral-Small-4-119B-2603 (Mistral4).

Mistral-Small-4 language model dimensions (from model card):
  - 36 decoder layers
  - 4096 hidden size
  - 32 attention heads
  - MLA (Multi-head Latent Attention):
      q_lora_rank=1024, kv_lora_rank=256
      qk_rope_head_dim=64, qk_nope_head_dim=64, v_head_dim=128
  - MoE: 128 experts, 4 active + 1 shared expert (always active)
  - Expert / shared-expert intermediate size: 2048
  - Vocabulary: 131072
"""

HF_MODEL_ID = "mistralai/Mistral-Small-4-119B-2603"

EXPECTED_NUM_LAYERS = 36
EXPECTED_VOCAB_SIZE = 131_072

# Key for embed_tokens weight in the HF state dict
TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY = "language_model.model.embed_tokens.weight"


def text_decoder_layer_state_dict_prefix(layer_idx: int) -> str:
    """Return the state-dict key prefix for the i-th decoder layer."""
    return f"language_model.model.layers.{layer_idx}."


# ── Architecture dimensions ────────────────────────────────────────────────

HIDDEN_SIZE = 4096
N_HEADS = 32

# MLA projection sizes
Q_LORA_RANK = 1024  # q_a_proj out / q_a_layernorm dim
KV_LORA_RANK = 256  # kv_a_layernorm dim (compressed KV latent)
QK_ROPE_HEAD_DIM = 64  # rope portion of each Q/K head
QK_NOPE_HEAD_DIM = 64  # non-rope (compressed) portion of each Q/K head
V_HEAD_DIM = 128  # value head dimension

HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 128 total Q/K head dim
KV_A_PROJ_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 320  = kv_a_proj_with_mqa output
KV_B_PROJ_OUT_PER_HEAD = QK_NOPE_HEAD_DIM + V_HEAD_DIM  # 192
KV_B_PROJ_OUT_TOTAL = N_HEADS * KV_B_PROJ_OUT_PER_HEAD  # 6144

# MoE
NUM_EXPERTS = 128
NUM_ACTIVE_EXPERTS = 4
EXPERT_INTERMEDIATE_SIZE = 2048
SHARED_EXPERT_INTERMEDIATE_SIZE = 2048

# Norms
NORM_EPS = 1e-6

# RoPE
ROPE_THETA = 1_000_000.0
