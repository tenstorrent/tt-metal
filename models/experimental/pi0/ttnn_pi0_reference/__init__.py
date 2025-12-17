# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN PI0 Reference Implementation.

This package provides a TTNN (Tenstorrent Neural Network) implementation
of the PI0 model for robot action prediction.

Architecture Overview:
    PI0 is a dual-expert Vision-Language-Action (VLA) model based on PaliGemma:
    
    1. Prefix Embedding:
        - SigLIP Vision Tower: Processes images into patch embeddings
        - Gemma Embeddings: Processes language tokens
        - Multi-modal Projector: Projects vision features to language dimension
    
    2. Suffix Embedding:
        - State Projection: Projects robot state to expert dimension
        - Action Projection: Projects noisy actions to expert dimension
        - Time Embedding: Sinusoidal encoding of denoising timestep
        - Action-Time MLP: Fuses action and time information
    
    3. PaliGemma Backbone:
        - Gemma 2B VLM: Processes prefix (images + language)
        - Gemma 300M Expert: Processes suffix (state + actions)
        - Shared Attention: Both see full K, V from combined sequence
    
    4. Denoising Module:
        - Flow Matching: Predicts velocity field for ODE integration
        - Euler Integration: Iteratively denoises actions

Module Structure:
    - weight_loader.py: Load weights from HuggingFace or local path
    - ttnn_common.py: Utility functions (sinusoidal embeddings, etc.)
    - ttnn_attention.py: Attention mask utilities
    - ttnn_suffix.py: State/action/time projections
    - ttnn_prefix.py: Image/language embedding
    - ttnn_gemma.py: Gemma transformer blocks
    - ttnn_siglip.py: SigLIP vision tower
    - ttnn_paligemma.py: Combined VLM + Expert backbone
    - ttnn_denoise.py: Denoising loop and KV cache
    - ttnn_pi0.py: Main model orchestrator

Usage:
    ```python
    from ttnn_pi0_reference import PI0Model, PI0ModelConfig
    
    # Load from local weights
    model = PI0Model.from_pretrained("/path/to/pi0_base")
    
    # Or from HuggingFace (requires HF_TOKEN)
    model = PI0Model.from_pretrained("lerobot/pi0_base")
    
    # Sample actions
    actions = model.sample_actions(
        images=[img],
        img_masks=[mask],
        lang_tokens=tokens,
        lang_masks=lang_mask,
        state=robot_state,
    )
    ```

TTNN Usage:
    ```python
    import ttnn
    from ttnn_pi0_reference import PI0ModelTTNN
    
    device = ttnn.open_device(device_id=0)
    model = PI0ModelTTNN.from_pretrained("/path/to/pi0_base", device)
    
    actions = model.sample_actions(images, img_masks, tokens, masks, state)
    
    ttnn.close_device(device)
    ```
"""

from .weight_loader import (
    PI0Config,
    PI0WeightLoader,
    load_pi0_state_dict,
    categorize_weights,
)

from .ttnn_common import (
    create_sinusoidal_pos_embedding,
    create_sinusoidal_pos_embedding_torch,
    create_sinusoidal_pos_embedding_ttnn,
    safe_cat,
    sample_noise,
    sample_time,
    torch_to_ttnn,
    ttnn_to_torch,
)

from .ttnn_attention import (
    AttentionMaskUtils,
    make_att_2d_masks,
    prepare_attention_masks_4d,
    combine_prefix_suffix_masks,
    create_causal_mask,
)

from .ttnn_suffix import (
    SuffixConfig,
    SuffixEmbeddingTorch,
    SuffixEmbeddingTTNN,
    SuffixEmbedding,
)

from .ttnn_prefix import (
    PrefixConfig,
    PrefixEmbeddingTorch,
    PrefixEmbeddingTTNN,
    PrefixEmbedding,
)

from .ttnn_gemma import (
    GemmaConfig,
    GemmaAttentionTorch,
    GemmaAttentionTTNN,
    GemmaMLPTorch,
    GemmaMLPTTNN,
    GemmaBlockTorch,
    GemmaBlockTTNN,
    GemmaAttention,
    GemmaMLP,
    GemmaBlock,
    rms_norm_torch,
    rms_norm_ttnn,
    precompute_freqs_cis_torch,
)

from .ttnn_siglip import (
    SigLIPConfig,
    SigLIPVisionTowerTorch,
    SigLIPVisionTowerTTNN,
    SigLIPVisionTower,
    MultiModalProjectorTorch,
    MultiModalProjectorTTNN,
    MultiModalProjector,
)

from .ttnn_paligemma import (
    PaliGemmaConfig,
    PaliGemmaBackboneTorch,
    PaliGemmaBackboneTTNN,
    PaliGemmaBackbone,
)

from .ttnn_denoise import (
    DenoiseConfig,
    DenoisingModuleTorch,
    DenoisingModuleTTNN,
    DenoisingModule,
    KVCacheManager,
    KVCacheManagerTTNN,
)

from .ttnn_pi0 import (
    PI0ModelConfig,
    PI0ModelTorch,
    PI0ModelTTNN,
    PI0Model,
)

__all__ = [
    # Config classes
    "PI0Config",
    "PI0ModelConfig",
    "GemmaConfig",
    "SigLIPConfig",
    "PaliGemmaConfig",
    "PrefixConfig",
    "SuffixConfig",
    "DenoiseConfig",
    
    # Weight loading
    "PI0WeightLoader",
    "load_pi0_state_dict",
    "categorize_weights",
    
    # Utility functions
    "create_sinusoidal_pos_embedding",
    "create_sinusoidal_pos_embedding_torch",
    "create_sinusoidal_pos_embedding_ttnn",
    "safe_cat",
    "sample_noise",
    "sample_time",
    "torch_to_ttnn",
    "ttnn_to_torch",
    
    # Attention utilities
    "AttentionMaskUtils",
    "make_att_2d_masks",
    "prepare_attention_masks_4d",
    "combine_prefix_suffix_masks",
    "create_causal_mask",
    
    # Gemma components
    "GemmaAttention",
    "GemmaAttentionTorch",
    "GemmaAttentionTTNN",
    "GemmaMLP",
    "GemmaMLPTorch",
    "GemmaMLPTTNN",
    "GemmaBlock",
    "GemmaBlockTorch",
    "GemmaBlockTTNN",
    "rms_norm_torch",
    "rms_norm_ttnn",
    "precompute_freqs_cis_torch",
    
    # SigLIP components
    "SigLIPVisionTower",
    "SigLIPVisionTowerTorch",
    "SigLIPVisionTowerTTNN",
    "MultiModalProjector",
    "MultiModalProjectorTorch",
    "MultiModalProjectorTTNN",
    
    # Embedding components
    "PrefixEmbedding",
    "PrefixEmbeddingTorch",
    "PrefixEmbeddingTTNN",
    "SuffixEmbedding",
    "SuffixEmbeddingTorch",
    "SuffixEmbeddingTTNN",
    
    # Backbone
    "PaliGemmaBackbone",
    "PaliGemmaBackboneTorch",
    "PaliGemmaBackboneTTNN",
    
    # Denoising
    "DenoisingModule",
    "DenoisingModuleTorch",
    "DenoisingModuleTTNN",
    "KVCacheManager",
    "KVCacheManagerTTNN",
    
    # Main model
    "PI0Model",
    "PI0ModelTorch",
    "PI0ModelTTNN",
]

