# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Model configuration for Qwen3-Coder-Next (80B-A3B MoE + DeltaNet).

Architecture: Hybrid MoE with Gated DeltaNet linear attention (3/4 layers)
and full GQA attention (1/4 layers). 512 experts, 10 active per token.

Reference: HuggingFace Qwen/Qwen3-Coder-Next config.json
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class Qwen3CoderNextConfig:
    """Configuration dataclass matching HF Qwen3-Coder-Next config.json.

    This model is a hybrid MoE with two attention types:
    - Gated DeltaNet (linear attention) for 3 of every 4 layers
    - Full GQA (softmax attention) for every 4th layer

    The pattern is controlled by full_attention_interval: every layer where
    layer_idx % full_attention_interval == (full_attention_interval - 1)
    uses GQA; all other layers use DeltaNet.
    """

    # Core dimensions
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 151936
    max_position_embeddings: int = 262144

    # MLP / MoE
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_experts: int = 512
    num_experts_per_tok: int = 10
    shared_expert_intermediate_size: int = 512
    hidden_act: str = "silu"

    # MoE routing
    router_aux_loss_coef: float = 0.001
    norm_topk_prob: bool = True

    # Attention
    full_attention_interval: int = 4
    partial_rotary_factor: float = 0.25
    rope_theta: float = 5000000.0

    # DeltaNet linear attention
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_value_head_dim: int = 128  # NOT head_dim (256) — DeltaNet uses smaller value heads
    linear_conv_kernel_dim: int = 4

    # Normalization
    rms_norm_eps: float = 1e-6

    # Model identity
    model_type: str = "qwen3_next"
    architectures: list = field(default_factory=lambda: ["Qwen3NextForCausalLM"])

    # Derived properties
    @property
    def rotary_dim(self) -> int:
        """Number of head dimensions that get rotary embeddings."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def non_rotary_dim(self) -> int:
        """Number of head dimensions that pass through without RoPE."""
        return self.head_dim - self.rotary_dim

    @property
    def num_deltanet_layers(self) -> int:
        """Number of layers using Gated DeltaNet attention."""
        return self.num_hidden_layers - self.num_gqa_layers

    @property
    def num_gqa_layers(self) -> int:
        """Number of layers using full GQA attention."""
        return self.num_hidden_layers // self.full_attention_interval

    @property
    def gqa_ratio(self) -> int:
        """Ratio of query heads to KV heads."""
        return self.num_attention_heads // self.num_key_value_heads

    def is_gqa_layer(self, layer_idx: int) -> bool:
        """Returns True if the given layer uses full GQA attention."""
        return layer_idx % self.full_attention_interval == (self.full_attention_interval - 1)

    @classmethod
    def from_hf_config(cls, hf_config_dict: dict) -> "Qwen3CoderNextConfig":
        """Create config from HuggingFace config.json dictionary.

        Args:
            hf_config_dict: Dictionary from HF config.json or AutoConfig.to_dict()

        Returns:
            Qwen3CoderNextConfig instance with values from HF config.
        """
        # Map HF config keys to our dataclass fields
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for key, value in hf_config_dict.items():
            if key in field_names:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "Qwen3CoderNextConfig":
        """Load config from a HuggingFace model name or local path.

        Args:
            model_name_or_path: HF model name (e.g., 'Qwen/Qwen3-Coder-Next')
                or local path to directory containing config.json.

        Returns:
            Qwen3CoderNextConfig instance.
        """
        path = Path(model_name_or_path)
        if path.is_dir():
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                return cls.from_hf_config(config_dict)

        # Try HF AutoConfig
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            return cls.from_hf_config(hf_config.to_dict())
        except ImportError:
            raise ImportError("transformers package required for loading from HF hub")

    def validate(self):
        """Validate config consistency."""
        assert self.num_hidden_layers % self.full_attention_interval == 0, (
            f"num_hidden_layers ({self.num_hidden_layers}) must be divisible by "
            f"full_attention_interval ({self.full_attention_interval})"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({self.num_key_value_heads})"
        )
        assert self.head_dim * self.partial_rotary_factor == int(self.head_dim * self.partial_rotary_factor), (
            f"head_dim ({self.head_dim}) * partial_rotary_factor ({self.partial_rotary_factor}) " f"must be an integer"
        )
        assert (
            self.num_experts >= self.num_experts_per_tok
        ), f"num_experts ({self.num_experts}) must be >= num_experts_per_tok ({self.num_experts_per_tok})"

    def memory_estimate_bytes(self, num_devices: int = 8, weight_dtype_bytes: int = 1) -> dict:
        """Estimate memory requirements per device.

        Args:
            num_devices: Number of devices to shard across.
            weight_dtype_bytes: Bytes per weight element (1 for bfp8, 2 for bf16).

        Returns:
            Dict with weight_bytes, expert_bytes_per_device, total_per_device.
        """
        # Non-expert params: embeddings + attention + norms + lm_head
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.vocab_size * self.hidden_size

        # Per-layer non-expert params (attention weights + norms)
        # GQA layers: Q, K, V, O projections
        gqa_params_per_layer = (
            self.hidden_size * self.num_attention_heads * self.head_dim  # Q
            + self.hidden_size * self.num_key_value_heads * self.head_dim  # K
            + self.hidden_size * self.num_key_value_heads * self.head_dim  # V
            + self.num_attention_heads * self.head_dim * self.hidden_size  # O
        )
        # DeltaNet layers: similar but with different head structure
        deltanet_params_per_layer = (
            self.hidden_size * self.linear_num_key_heads * self.linear_key_head_dim  # K
            + self.hidden_size * self.linear_num_value_heads * self.head_dim  # V
            + self.hidden_size * self.hidden_size  # O (approximate)
        )

        norm_params_per_layer = 2 * self.hidden_size  # 2 RMSNorms per layer

        non_expert_params = (
            embedding_params
            + lm_head_params
            + self.num_gqa_layers * (gqa_params_per_layer + norm_params_per_layer)
            + self.num_deltanet_layers * (deltanet_params_per_layer + norm_params_per_layer)
        )

        # Expert params: each expert has gate_proj, up_proj, down_proj (SwiGLU)
        expert_params_per_expert = (
            self.hidden_size * self.moe_intermediate_size  # gate_proj
            + self.hidden_size * self.moe_intermediate_size  # up_proj
            + self.moe_intermediate_size * self.hidden_size  # down_proj
        )
        # Shared expert (same structure but different size)
        shared_expert_params = (
            self.hidden_size * self.shared_expert_intermediate_size  # gate_proj
            + self.hidden_size * self.shared_expert_intermediate_size  # up_proj
            + self.shared_expert_intermediate_size * self.hidden_size  # down_proj
        )

        total_expert_params = (
            self.num_experts * expert_params_per_expert + self.num_hidden_layers * shared_expert_params
        ) * self.num_hidden_layers

        # Router params per layer
        router_params = self.num_hidden_layers * self.hidden_size * self.num_experts

        total_params = non_expert_params + total_expert_params + router_params
        total_bytes = total_params * weight_dtype_bytes
        per_device_bytes = total_bytes // num_devices

        return {
            "total_params": total_params,
            "non_expert_params": non_expert_params,
            "expert_params": total_expert_params,
            "router_params": router_params,
            "total_bytes": total_bytes,
            "per_device_bytes": per_device_bytes,
            "per_device_gb": per_device_bytes / (1024**3),
        }

    def __post_init__(self):
        self.validate()
        logger.info(
            f"Qwen3-Coder-Next config: {self.num_hidden_layers} layers "
            f"({self.num_deltanet_layers} DeltaNet + {self.num_gqa_layers} GQA), "
            f"{self.num_experts} experts ({self.num_experts_per_tok} active)"
        )
