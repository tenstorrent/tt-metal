# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 ModelArgs: HF config parsing, weight loading, and model configuration.

Supports all Gemma4 variants:
- E2B/E4B: Dense models (no MoE, per-layer input embeddings)
- A4B/26B: MoE models (128 experts, top-8 routing)
- 31B: Similar to A4B with different dimensions

Config is automatically loaded from the model checkpoint's config.json
via HF AutoConfig. Specify model path via HF_MODEL env var.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn


@dataclass
class Gemma4ModelArgs:
    """Gemma4 model arguments parsed from HuggingFace config.

    All fields have safe defaults but should be populated via from_hf_config()
    for any real model.
    """

    # Core dimensions
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    # Global attention overrides (None = same as sliding)
    num_global_key_value_heads: int = 2
    global_head_dim: int = 512
    attention_k_eq_v: bool = True
    # Sliding window
    sliding_window: int = 1024
    # RoPE
    rope_theta: float = 10000.0
    global_rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25
    # Shared MLP
    intermediate_size: int = 2112
    hidden_activation: str = "gelu_pytorch_tanh"
    # MoE (disabled by default for dense models)
    enable_moe_block: bool = True
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    # Per-layer input embeddings (E2B/E4B feature)
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 262144
    # KV sharing
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    # General
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    # Layer pattern
    layer_types: tuple = None

    def __post_init__(self):
        if self.layer_types is None:
            pattern = ["sliding_attention"] * 5 + ["full_attention"]
            self.layer_types = tuple(pattern * (self.num_hidden_layers // 6 + 1))
            self.layer_types = self.layer_types[: self.num_hidden_layers]

    @classmethod
    def from_hf_config(cls, hf_config):
        """Create Gemma4ModelArgs from a HuggingFace AutoConfig.

        Handles all Gemma4 variants (E2B, E4B, A4B, 31B).
        """
        tc = getattr(hf_config, "text_config", hf_config)
        layer_types = tuple(tc.layer_types) if hasattr(tc, "layer_types") and tc.layer_types else None

        rope_params = getattr(tc, "rope_parameters", {}) or {}
        sliding_rope = rope_params.get("sliding_attention", {})
        full_rope = rope_params.get("full_attention", {})

        # num_global_key_value_heads: None means use same as sliding
        num_global_kv = getattr(tc, "num_global_key_value_heads", None)
        if num_global_kv is None:
            num_global_kv = getattr(tc, "num_key_value_heads", 8)

        return cls(
            hidden_size=tc.hidden_size,
            num_hidden_layers=tc.num_hidden_layers,
            num_attention_heads=tc.num_attention_heads,
            num_key_value_heads=getattr(tc, "num_key_value_heads", 8),
            head_dim=getattr(tc, "head_dim", 256),
            num_global_key_value_heads=num_global_kv,
            global_head_dim=getattr(tc, "global_head_dim", 512),
            attention_k_eq_v=getattr(tc, "attention_k_eq_v", False),
            sliding_window=getattr(tc, "sliding_window", 1024),
            rope_theta=sliding_rope.get("rope_theta", getattr(tc, "rope_theta", 10000.0)),
            global_rope_theta=full_rope.get("rope_theta", 1000000.0),
            partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
            intermediate_size=tc.intermediate_size,
            hidden_activation=getattr(tc, "hidden_activation", "gelu_pytorch_tanh"),
            enable_moe_block=getattr(tc, "enable_moe_block", False),
            moe_intermediate_size=getattr(tc, "moe_intermediate_size", None) or 0,
            num_experts=getattr(tc, "num_experts", None) or 0,
            top_k_experts=getattr(tc, "top_k_experts", None) or 0,
            hidden_size_per_layer_input=getattr(tc, "hidden_size_per_layer_input", 0) or 0,
            vocab_size_per_layer_input=getattr(tc, "vocab_size_per_layer_input", 262144),
            num_kv_shared_layers=getattr(tc, "num_kv_shared_layers", 0),
            use_double_wide_mlp=getattr(tc, "use_double_wide_mlp", False),
            vocab_size=tc.vocab_size,
            rms_norm_eps=getattr(tc, "rms_norm_eps", 1e-6),
            final_logit_softcapping=getattr(tc, "final_logit_softcapping", None) or 0.0,
            tie_word_embeddings=getattr(tc, "tie_word_embeddings", True),
            attention_bias=getattr(tc, "attention_bias", False),
            layer_types=layer_types,
        )

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False):
        """Load model state dict from safetensors (fast) or HF checkpoint."""
        if dummy_weights:
            return {}

        from pathlib import Path

        safetensor_files = sorted(Path(weights_path).glob("*.safetensors"))
        if safetensor_files:
            from safetensors.torch import load_file

            logger.info(f"Loading {len(safetensor_files)} safetensor files from {weights_path}")
            state_dict = {}
            for f in tqdm(safetensor_files, desc="Loading safetensors"):
                shard = load_file(str(f))
                state_dict.update(shard)

            for k, v in state_dict.items():
                if v.dtype == torch.float32:
                    state_dict[k] = v.to(torch.bfloat16)

            logger.info(f"Loaded {len(state_dict)} tensors")
            return state_dict

        logger.info(f"No safetensors found, loading via AutoModelForCausalLM from {weights_path}")
        model = AutoModelForCausalLM.from_pretrained(weights_path, torch_dtype="auto")
        state_dict = model.state_dict()
        del model

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

    # ── Generator compatibility properties ─────────────────────────────────
    # The tt_transformers Generator expects these attribute names.

    @property
    def dim(self):
        return self.hidden_size

    @property
    def n_layers(self):
        return self.num_hidden_layers

    @property
    def max_batch_size(self):
        return getattr(self, "_max_batch_size", 1)

    @max_batch_size.setter
    def max_batch_size(self, value):
        self._max_batch_size = value

    @property
    def max_seq_len(self):
        return getattr(self, "_max_seq_len", 131072)

    @max_seq_len.setter
    def max_seq_len(self, value):
        self._max_seq_len = value

    def get_warmup_prefill_supported_seq_lens(self):
        """Sequence lengths to compile during prefill warmup."""
        return [32, 128, 512]

    def weight_cache_path(self, model_path, dtype):
        """Return weight cache path for the model."""
        cache_dir = os.getenv("TT_CACHE_PATH")
        if cache_dir:
            cache_dir = Path(cache_dir)
        elif Path(model_path).is_dir():
            # Local checkpoint: cache next to the weights.
            cache_dir = Path(model_path)
        else:
            # Otherwise model_path is an HF id like "google/gemma-4-E2B-it".
            # Caching under Path(model_path) would create that as a relative dir
            # in cwd, which then makes transformers' AutoConfig.from_pretrained
            # treat the id as a local path (os.path.isdir returns True) and fail
            # to find config.json. Fall back to an HF_HOME-based cache instead.
            hf_home = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            sanitized = str(model_path).replace("/", "--")
            cache_dir = Path(hf_home) / "tt_cache" / sanitized
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
        cache_path = cache_dir / f"tensor_cache_{dtype_str}"
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
