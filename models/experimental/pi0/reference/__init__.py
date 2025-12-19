# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Reference Implementation - Pure PyTorch modules.

All modules in this folder are pure PyTorch implementations used as
ground truth for PCC (Pearson Correlation Coefficient) testing.

Modules:
    - torch_gemma: GemmaAttention, GemmaMLP, GemmaBlock
    - torch_siglip: SigLIPAttention, SigLIPMLP, SigLIPBlock, SigLIPVisionTower
    - torch_prefix: PrefixEmbedding
    - torch_suffix: SuffixEmbedding
    - torch_paligemma: PaliGemmaBackbone
    - torch_denoise: DenoisingModule, KVCacheManager
    - torch_pi0_model: PI0Model
"""

# Gemma components
from models.experimental.pi0.reference.torch_gemma import (
    GemmaAttention,
    GemmaMLP,
    GemmaBlock,
    rms_norm,
    precompute_freqs_cis,
)

# SigLIP components
from models.experimental.pi0.reference.torch_siglip import (
    SigLIPAttention,
    SigLIPMLP,
    SigLIPBlock,
    SigLIPVisionTower,
    PatchEmbedding,
    MultiModalProjector,
)

# Prefix/Suffix embeddings
from models.experimental.pi0.reference.torch_prefix import PrefixEmbedding
from models.experimental.pi0.reference.torch_suffix import SuffixEmbedding

# PaliGemma backbone
from models.experimental.pi0.reference.torch_paligemma import PaliGemmaBackbone

# Denoising
from models.experimental.pi0.reference.torch_denoise import (
    DenoisingModule,
    KVCacheManager,
)

# Full model
from models.experimental.pi0.reference.torch_pi0_model import PI0Model
