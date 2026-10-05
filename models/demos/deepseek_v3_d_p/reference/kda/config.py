# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Immutable, device-independent configuration for Kimi Delta Attention."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# KDA's unbounded decay gate follows torch.nn.functional.softplus defaults.
# Hugging Face Kimi configs do not expose these algorithm-level constants.
KDA_SOFTPLUS_BETA = 1.0
KDA_SOFTPLUS_THRESHOLD = 20.0


@dataclass(frozen=True)
class KDAConfig:
    """Dimensions and numerical policy for one KDA layer."""

    hidden_size: int
    num_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    use_full_rank_gate: bool = False
    gate_lower_bound: float | None = None

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "conv_kernel_size": self.conv_kernel_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.conv_kernel_size != 4:
            raise ValueError(f"KDA currently requires conv_kernel_size=4, got {self.conv_kernel_size}")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be finite and positive, got {self.norm_eps}")
        if self.gate_lower_bound is not None and not -5.0 <= self.gate_lower_bound < 0.0:
            raise ValueError(f"gate_lower_bound must be in [-5, 0), got {self.gate_lower_bound}")

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def k_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def v_dim(self) -> int:
        return self.num_heads * self.head_v_dim

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> "KDAConfig":
        """Build from a Hugging Face configuration mapping of a supported KDA model.

        The text configuration (``text_config`` when present) selects the interpretation by its
        ``model_type``: ``kimi_linear`` (Kimi Linear, Kimi K3) or ``glm5_next_text``
        (GLM-5.3-Flash). Every ``linear_attn_config`` key must be modeled by that interpretation
        or be a layer-schedule key; any other key is rejected rather than silently ignored.
        """
        if "text_config" in model_config:
            model_config = model_config["text_config"]
            if not isinstance(model_config, Mapping):
                raise TypeError("text_config must be a mapping")
        model_type = model_config.get("model_type")
        if model_type not in _MODEL_TYPE_LINEAR_ATTN_KEYS:
            raise ValueError(
                f"unsupported KDA model_type {model_type!r}; expected one of {sorted(_MODEL_TYPE_LINEAR_ATTN_KEYS)}"
            )
        try:
            linear = model_config["linear_attn_config"]
            if not isinstance(linear, Mapping):
                raise TypeError("linear_attn_config must be a mapping")
            unmodeled = sorted(set(linear) - _MODEL_TYPE_LINEAR_ATTN_KEYS[model_type] - _LAYER_SCHEDULE_KEYS)
            if unmodeled:
                raise ValueError(f"{model_type} linear_attn_config has keys KDA does not model: {unmodeled}")
            if model_type == "glm5_next_text":
                # Glm5NextTextLinearAttention runs the short convolution with ACT2FN[hidden_act];
                # the TT layer hardcodes SiLU, which is also the config class default.
                hidden_act = model_config.get("hidden_act", "silu")
                if hidden_act != "silu":
                    raise ValueError(f"glm5_next_text KDA convolution requires hidden_act='silu', got {hidden_act!r}")
                # Glm5NextTextConfig: an absent or null bound becomes -5.0 unless safe_gate is false.
                gate_lower_bound = linear.get("gate_lower_bound", -5.0)
                if gate_lower_bound is None and linear.get("safe_gate", True):
                    gate_lower_bound = -5.0
                # The GLM layer always uses the low-rank output gate.
                use_full_rank_gate = False
            else:
                # KimiDeltaAttention hardcodes the SiLU convolution (hidden_act, "situ" for K3, is the
                # MLP activation) and selects the softplus decay when the bound is absent or null.
                gate_lower_bound = linear.get("gate_lower_bound")
                use_full_rank_gate = bool(linear.get("use_full_rank_gate", False))
            head_dim = int(linear["head_dim"])
            return cls(
                hidden_size=int(model_config["hidden_size"]),
                num_heads=int(linear["num_heads"]),
                head_k_dim=head_dim,
                head_v_dim=head_dim,
                conv_kernel_size=int(linear["short_conv_kernel_size"]),
                norm_eps=float(model_config["rms_norm_eps"]),
                use_full_rank_gate=use_full_rank_gate,
                gate_lower_bound=float(gate_lower_bound) if gate_lower_bound is not None else None,
            )
        except KeyError as error:
            raise ValueError(f"missing {model_type} config field: {error.args[0]}") from error


# linear_attn_config keys each supported text model_type defines and KDAConfig models.
_MODEL_TYPE_LINEAR_ATTN_KEYS = {
    "kimi_linear": frozenset(
        {"num_heads", "head_dim", "short_conv_kernel_size", "use_full_rank_gate", "gate_lower_bound"}
    ),
    "glm5_next_text": frozenset({"num_heads", "head_dim", "short_conv_kernel_size", "gate_lower_bound", "safe_gate"}),
}
# Which layers are KDA layers; consumed by the model's layer schedule, not by one KDA layer.
# Kimi lists them 1-indexed, GLM-5.3-Flash 0-indexed.
_LAYER_SCHEDULE_KEYS = frozenset({"kda_layers", "full_attn_layers"})
