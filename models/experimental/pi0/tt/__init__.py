# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 TTNN Implementation - Tenstorrent hardware-accelerated modules.

All modules in this folder use TTNN for hardware acceleration on
Tenstorrent devices.

This is a pure TTNN implementation (no PyTorch fallbacks) with the
following optimizations:
    - Fused QKV projection
    - Native TTNN RoPE (ttnn.experimental.rotary_embedding)
    - Native head operations (nlp_create_qkv_heads, nlp_concat_heads)
    - Pre-computed attention masks and timesteps
    - Denoising loop stays entirely on device

Modules:
    - ttnn_gemma: GemmaAttentionTTNN, GemmaMLPTTNN, GemmaBlockTTNN
    - ttnn_siglip: SigLIPAttentionTTNN, SigLIPMLPTTNN, SigLIPBlockTTNN, etc.
    - ttnn_prefix: PrefixEmbeddingTTNN
    - ttnn_suffix: SuffixEmbeddingTTNN
    - ttnn_paligemma: PaliGemmaBackboneTTNN
    - ttnn_pi0_model: PI0ModelTTNN
"""

# Gemma components
from .ttnn_gemma import (
    GemmaAttentionTTNN,
    GemmaMLPTTNN,
    GemmaBlockTTNN,
    rms_norm_ttnn,
    precompute_freqs_cis_meta_format,
    # Aliases for backwards compatibility
    GemmaAttention,
    GemmaMLP,
    GemmaBlock,
)

# SigLIP components
from .ttnn_siglip import (
    SigLIPAttentionTTNN,
    SigLIPMLPTTNN,
    SigLIPBlockTTNN,
    SigLIPVisionTowerTTNN,
    PatchEmbeddingTTNN,
    MultiModalProjectorTTNN,
    # Aliases for backwards compatibility
    SigLIPAttention,
    SigLIPMLP,
    SigLIPBlock,
    SigLIPVisionTower,
    PatchEmbedding,
    MultiModalProjector,
)

# Prefix/Suffix embeddings
from .ttnn_prefix import PrefixEmbeddingTTNN, PrefixEmbedding
from .ttnn_suffix import SuffixEmbeddingTTNN, SuffixEmbedding, convert_suffix_weights_to_ttnn

# PaliGemma backbone
from .ttnn_paligemma import PaliGemmaBackboneTTNN, PaliGemmaBackbone

# Full model
from .ttnn_pi0_model import PI0ModelTTNN, PI0Model

# Common utilities
from .ttnn_common import (
    create_sinusoidal_pos_embedding_ttnn,
    safe_cat_ttnn,
    compute_position_ids_ttnn,
    ttnn_to_torch,
    torch_to_ttnn,
)

__all__ = [
    # Gemma
    "GemmaAttentionTTNN",
    "GemmaMLPTTNN",
    "GemmaBlockTTNN",
    "GemmaAttention",
    "GemmaMLP",
    "GemmaBlock",
    "rms_norm_ttnn",
    "precompute_freqs_cis_meta_format",
    # SigLIP
    "SigLIPAttentionTTNN",
    "SigLIPMLPTTNN",
    "SigLIPBlockTTNN",
    "SigLIPVisionTowerTTNN",
    "PatchEmbeddingTTNN",
    "MultiModalProjectorTTNN",
    "SigLIPAttention",
    "SigLIPMLP",
    "SigLIPBlock",
    "SigLIPVisionTower",
    "PatchEmbedding",
    "MultiModalProjector",
    # Prefix/Suffix
    "PrefixEmbeddingTTNN",
    "PrefixEmbedding",
    "SuffixEmbeddingTTNN",
    "SuffixEmbedding",
    "convert_suffix_weights_to_ttnn",
    # Backbone
    "PaliGemmaBackboneTTNN",
    "PaliGemmaBackbone",
    # Full model
    "PI0ModelTTNN",
    "PI0Model",
    # Utilities
    "create_sinusoidal_pos_embedding_ttnn",
    "safe_cat_ttnn",
    "compute_position_ids_ttnn",
    "ttnn_to_torch",
    "torch_to_ttnn",
]
