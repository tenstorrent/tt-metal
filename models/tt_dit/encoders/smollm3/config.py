# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


def _read_rope_theta(hf_config) -> float:
    """rope_theta may be a top-level attr or nested under rope_parameters/rope_scaling (transformers 5.x)."""
    theta = getattr(hf_config, "rope_theta", None)
    if theta is None:
        params = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
        theta = params.get("rope_theta")
    if theta is None:
        raise ValueError("SmolLM3 config missing rope_theta (checked top-level and rope_parameters/rope_scaling)")
    return theta


@dataclass
class SmolLM3Config:
    """Configuration for the SmolLM3-3B text encoder (used by Bria FIBO)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 65536
    hidden_act: str = "silu"
    attention_bias: bool = False
    no_rope_layer_interval: int = 4
    no_rope_layers: list[int] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # HF default: no_rope_layers[i] = int((i + 1) % interval != 0); 1 = apply RoPE, 0 = NoPE.
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0) for i in range(self.num_hidden_layers)
            ]
        else:
            self.no_rope_layers = list(self.no_rope_layers)

    @classmethod
    def from_hf_config(cls, hf_config) -> "SmolLM3Config":
        """Build from a transformers SmolLM3Config (or the .config of a loaded model)."""
        hf_no_rope = getattr(hf_config, "no_rope_layers", None)
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
            no_rope_layer_interval=getattr(hf_config, "no_rope_layer_interval", 4),
            no_rope_layers=list(hf_no_rope) if hf_no_rope is not None else None,
        )
