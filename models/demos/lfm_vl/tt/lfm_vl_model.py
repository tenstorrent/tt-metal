# Fix for Issue #45087: [Bounty $1500] Add /support (LiquidAI/LFM2.5-VL-1.6B) in tt-metal

# models/demos/lfm_vl/tt/lfm_vl_config.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LFMVLConfig:
    """Configuration for LFM2.5-VL-1.6B model"""
    
    # Model dimensions
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    vocab_size: int = 65536
    
    # Vision encoder config
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    
    # Projector config
    projector_hidden_size: int = 2048
    
    # SSM (State Space Model) config for Liquid architecture
    ssm_state_size: int = 16
    ssm_conv_kernel_size: int = 4
    ssm_expand: int = 2
    
    # Training config
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    
    # TT-Metal specific
    tt_cache_path: Optional[str] = None
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """Load config from HuggingFace model"""
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            return cls(
                hidden_size=getattr(hf_config, 'hidden_size', 2048),
                intermediate_size=getattr(hf_config, 'intermediate_size', 5504),
                num_hidden_layers=getattr(hf_config, 'num_hidden_layers', 24),
                num_attention_heads=getattr(hf_config, 'num_attention_heads', 16),
                num_key_value_heads=getattr(hf_config, 'num_key_value_heads', 16),
                vocab_size=getattr(hf_config, 'vocab_size', 65536),
                vision_hidden_size=getattr(hf_config.vision_config, 'hidden_size', 1024) if hasattr(hf_config, 'vision_config') else 1024,
                image_size=getattr(hf_config.vision_config, 'image_size', 384) if hasattr(hf_config, 'vision_config') else 384,
                patch_size=getattr(hf_config.vision_config, 'patch_size', 14) if hasattr(hf_config, 'vision_config') else 14,
            )
        except Exception as e:
            print(f"Warning: Could not load config from {model_name_or_path}: {e}")
            return cls()