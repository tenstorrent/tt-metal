# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model arguments read from the snapshot ``config.json`` text config. No weights are loaded here."""

from __future__ import annotations

from dataclasses import dataclass

# The app (``DecisionModel.prepare``) rejects prompts longer than this; HF advertises 262144.
APP_MAX_LENGTH = 8192


@dataclass(frozen=True)
class PplxDeciderArgs:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    vocab_size: int
    rms_norm_eps: float
    # gated full attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    # gated delta net
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    # readout
    num_options: int = 255
    max_seq_len: int = APP_MAX_LENGTH

    @classmethod
    def from_hf_config(cls, text_config, *, max_seq_len: int = APP_MAX_LENGTH) -> "PplxDeciderArgs":
        rope = text_config.rope_parameters
        partial = rope.get("partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 1.0))
        args = cls(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            num_hidden_layers=text_config.num_hidden_layers,
            layer_types=tuple(text_config.layer_types),
            vocab_size=text_config.vocab_size,
            rms_norm_eps=text_config.rms_norm_eps,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            rotary_dim=int(text_config.head_dim * partial),
            rope_theta=float(rope["rope_theta"]),
            linear_num_key_heads=text_config.linear_num_key_heads,
            linear_num_value_heads=text_config.linear_num_value_heads,
            linear_key_head_dim=text_config.linear_key_head_dim,
            linear_value_head_dim=text_config.linear_value_head_dim,
            linear_conv_kernel_dim=text_config.linear_conv_kernel_dim,
            max_seq_len=max_seq_len,
        )
        args.validate()
        return args

    def validate(self) -> None:
        # The TT kernels used here are written for this exact text config; fail rather than guess.
        expected = dict(
            hidden_size=5120,
            intermediate_size=17408,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=256,
            rotary_dim=64,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Unsupported {name}={getattr(self, name)}; expected {value}")

    def layer_kind(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    @property
    def conv_width(self) -> int:
        return 2 * self.linear_num_key_heads * self.linear_key_head_dim + self.linear_value_dim

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim
