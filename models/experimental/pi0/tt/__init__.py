# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 TTNN Implementation - Tenstorrent hardware-accelerated modules.

All modules in this folder use TTNN for hardware acceleration on
Tenstorrent devices.

Modules:
    - ttnn_gemma: TtGemmaAttention, TtGemmaMLP, TtGemmaBlock
    - ttnn_siglip: TtSigLIPAttention, TtSigLIPMLP, TtSigLIPBlock, TtSigLIPVisionTower
    - ttnn_prefix: TtPrefixEmbedding
    - ttnn_suffix: TtSuffixEmbedding
    - ttnn_paligemma: TtPaliGemmaBackbone
    - ttnn_denoise: TtDenoisingModule, TtKVCacheManager
    - ttnn_pi0_model: TtPI0Model
"""

# Gemma components
from models.experimental.pi0.tt.ttnn_gemma import (
    TtGemmaAttention,
    TtGemmaMLP,
    TtGemmaBlock,
    rms_norm,
)

# SigLIP components
from models.experimental.pi0.tt.ttnn_siglip import (
    TtSigLIPAttention,
    TtSigLIPMLP,
    TtSigLIPBlock,
    TtSigLIPVisionTower,
    TtPatchEmbedding,
    TtMultiModalProjector,
)

# Prefix/Suffix embeddings
from models.experimental.pi0.tt.ttnn_prefix import TtPrefixEmbedding
from models.experimental.pi0.tt.ttnn_suffix import TtSuffixEmbedding

# PaliGemma backbone
from models.experimental.pi0.tt.ttnn_paligemma import TtPaliGemmaBackbone

# Denoising
from models.experimental.pi0.tt.ttnn_denoise import (
    TtDenoisingModule,
    TtKVCacheManager,
)

# Full model
from models.experimental.pi0.tt.ttnn_pi0_model import TtPI0Model
