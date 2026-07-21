# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS 120B Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for gpt-oss-120b.
"""


class GptOss120BConfig:
    """GPT-OSS 120B model dimensions."""

    # Core dimensions
    EMB_SIZE = 2880  # embedding dimension
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2880  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 2880  # Dense FFN hidden dimension (same as MoE)
    HEAD_DIM = 64

    # MoE configuration
    NUM_ROUTED_EXPERTS = 128
    NUM_EXPERTS_PER_TOKEN = 4

    # Model architecture
    NUM_LAYERS = 36
    VOCAB_SIZE = 201088
    SLIDING_WINDOW = 128
    # TODO: HF config defines `layer_types` interleaving `sliding_attention` and `full_attention`
    # per layer. Decide whether to encode that pattern here or hardcode it at the call site.

    # Attention dimensions
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 8

    # Other
    RMS_NORM_EPS = 1e-5
    ROPE_THETA = 150000
    SWIGLU_LIMIT = 7.0
