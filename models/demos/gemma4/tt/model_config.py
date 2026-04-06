# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 ModelArgs: HF config parsing, weight loading, and model configuration.

Gemma4 26B-A4B architecture:
- hidden_size: 2816
- num_hidden_layers: 30
- num_attention_heads: 16
- sliding layers: num_kv_heads=8, head_dim=256, window=1024
- global layers: num_kv_heads=2, head_dim=512, rope_theta=1M, partial_rotary=0.25
- intermediate_size: 2112 (shared MLP)
- moe_intermediate_size: 704 (per routed expert)
- num_experts: 128, top_k: 8
- vocab_size: 262144
- layer_types: [5 sliding, 1 full] x 5
- activation: GeGLU (gelu_pytorch_tanh)
- attention_k_eq_v: True
- final_logit_softcapping: 30.0
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn


@dataclass
class Gemma4ModelArgs:
    """Gemma4 model arguments parsed from HuggingFace config."""

    # Core dimensions
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # KV heads for sliding layers
    head_dim: int = 256  # head dim for sliding layers
    # Global attention overrides
    num_global_key_value_heads: int = 2  # KV heads for global/full layers
    global_head_dim: int = 512  # head dim for global/full layers
    attention_k_eq_v: bool = True  # K=V tying for global layers
    # Sliding window
    sliding_window: int = 1024
    # RoPE
    rope_theta: float = 10000.0  # sliding layers
    global_rope_theta: float = 1000000.0  # global layers
    partial_rotary_factor: float = 0.25  # global layers partial rotation
    # Shared MLP
    intermediate_size: int = 2112
    hidden_activation: str = "gelu_pytorch_tanh"
    # MoE
    enable_moe_block: bool = True
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    # General
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    # Layer pattern: [5 sliding, 1 full] x 5
    layer_types: tuple = None

    def __post_init__(self):
        if self.layer_types is None:
            pattern = ["sliding_attention"] * 5 + ["full_attention"]
            self.layer_types = tuple(pattern * 5)

    @classmethod
    def from_hf_config(cls, hf_config):
        """Create Gemma4ModelArgs from a HuggingFace AutoConfig.

        Handles both direct Gemma4TextConfig and the nested config.json
        structure where text config is under `text_config`.
        """
        # Handle nested config (Gemma4ForConditionalGeneration wraps text_config)
        tc = getattr(hf_config, "text_config", hf_config)
        layer_types = tuple(tc.layer_types) if hasattr(tc, "layer_types") else None

        # Extract rope params from nested dict if present
        rope_params = getattr(tc, "rope_parameters", {}) or {}
        sliding_rope = rope_params.get("sliding_attention", {})
        full_rope = rope_params.get("full_attention", {})

        return cls(
            hidden_size=tc.hidden_size,
            num_hidden_layers=tc.num_hidden_layers,
            num_attention_heads=tc.num_attention_heads,
            num_key_value_heads=getattr(tc, "num_key_value_heads", 8),
            head_dim=getattr(tc, "head_dim", 256),
            num_global_key_value_heads=getattr(tc, "num_global_key_value_heads", 2),
            global_head_dim=getattr(tc, "global_head_dim", 512),
            attention_k_eq_v=getattr(tc, "attention_k_eq_v", True),
            sliding_window=getattr(tc, "sliding_window", 1024),
            rope_theta=sliding_rope.get("rope_theta", getattr(tc, "rope_theta", 10000.0)),
            global_rope_theta=full_rope.get("rope_theta", 1000000.0),
            partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
            intermediate_size=tc.intermediate_size,
            hidden_activation=getattr(tc, "hidden_activation", "gelu_pytorch_tanh"),
            enable_moe_block=getattr(tc, "enable_moe_block", True),
            moe_intermediate_size=getattr(tc, "moe_intermediate_size", 704),
            num_experts=getattr(tc, "num_experts", 128),
            top_k_experts=getattr(tc, "top_k_experts", 8),
            vocab_size=tc.vocab_size,
            rms_norm_eps=getattr(tc, "rms_norm_eps", 1e-6),
            final_logit_softcapping=getattr(tc, "final_logit_softcapping", 30.0),
            tie_word_embeddings=getattr(tc, "tie_word_embeddings", True),
            attention_bias=getattr(tc, "attention_bias", False),
            layer_types=layer_types,
        )

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False):
        """Load model state dict from HuggingFace checkpoint.

        Args:
            weights_path: Path to model weights directory
            dummy_weights: If True, return empty dict for testing

        Returns:
            State dict mapping parameter names to tensors
        """
        if dummy_weights:
            return {}

        model = AutoModelForCausalLM.from_pretrained(weights_path, torch_dtype="auto")
        state_dict = model.state_dict()

        # Convert to bfloat16 if needed
        if any(v.dtype == torch.float32 for v in state_dict.values()):
            state_dict = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in tqdm(state_dict.items(), desc="Converting to bfloat16")
            }

        return state_dict

    @staticmethod
    def load_hf_config(model_path):
        """Load HuggingFace config."""
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    def weight_cache_path(self, model_path, dtype):
        """Return weight cache path for the model."""
        cache_dir = os.getenv("TT_CACHE_PATH")
        if cache_dir:
            cache_dir = Path(cache_dir)
        else:
            cache_dir = Path(model_path)
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
        cache_path = cache_dir / f"tensor_cache_{dtype_str}"
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
