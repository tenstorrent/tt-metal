# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def _read_rope_theta(hf_config) -> float:
    """rope_theta may be a top-level attr or nested under rope_parameters/rope_scaling (transformers 5.x)."""
    theta = getattr(hf_config, "rope_theta", None)
    if theta is None:
        params = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
        theta = params.get("rope_theta")
    if theta is None:
        raise ValueError("SmolLM3 config missing rope_theta (checked top-level and rope_parameters/rope_scaling)")
    return theta


class SmolLM3Config:
    """Configuration for the SmolLM3-3B text encoder (used by Bria FIBO)."""

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5000000.0,
        max_position_embeddings: int = 65536,
        hidden_act: str = "silu",
        attention_bias: bool = False,
        no_rope_layer_interval: int = 4,
        no_rope_layers: list[int] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.no_rope_layer_interval = no_rope_layer_interval
        # HF default: no_rope_layers[i] = int((i + 1) % interval != 0); 1 = apply RoPE, 0 = NoPE.
        if no_rope_layers is not None:
            self.no_rope_layers = list(no_rope_layers)
        else:
            self.no_rope_layers = [int((i + 1) % no_rope_layer_interval != 0) for i in range(num_hidden_layers)]

    @classmethod
    def from_hf_config(cls, hf_config) -> "SmolLM3Config":
        """Build from a transformers SmolLM3Config (or the .config of a loaded model)."""
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=getattr(hf_config, "head_dim", None),
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=_read_rope_theta(hf_config),
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_act=hf_config.hidden_act,
            attention_bias=getattr(hf_config, "attention_bias", False),
            no_rope_layers=list(getattr(hf_config, "no_rope_layers", None))
            if getattr(hf_config, "no_rope_layers", None) is not None
            else None,
            no_rope_layer_interval=getattr(hf_config, "no_rope_layer_interval", 4),
        )
