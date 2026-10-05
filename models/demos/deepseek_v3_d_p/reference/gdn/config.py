# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Immutable, device-independent configuration of one Qwen Gated DeltaNet (GDN) layer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Activation of the output gate z in the gated RMSNorm, ``w * rmsnorm(o) * act(z)``. Model configs spell silu as
# ``swish`` or leave it unset; ``from_model_config`` resolves those aliases, so only canonical names reach here.
GDN_OUTPUT_GATE_ACTIVATIONS = ("silu", "sigmoid")


@dataclass(frozen=True)
class GDNConfig:
    """Dimensions and numerical policy of one GDN layer.

    q and k carry ``num_key_heads`` heads; v, z, the decay ``g``, ``beta``, the recurrent state and the output norm
    carry ``num_value_heads``. V head ``j`` reads K head ``j // group`` (transformers ``repeat_interleave`` order).
    ``output_gate_activation`` is ``silu`` (transformers ``qwen3_5`` / ``qwen3_5_moe``) or ``sigmoid`` (``qwen4_exp``
    with ``output_gate_type: sigmoid``, the KDA gate).
    """

    hidden_size: int
    num_key_heads: int
    num_value_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    output_gate_activation: str

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_key_heads": self.num_key_heads,
            "num_value_heads": self.num_value_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "conv_kernel_size": self.conv_kernel_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError(
                f"num_value_heads {self.num_value_heads} must be a multiple of num_key_heads {self.num_key_heads}"
            )
        if self.conv_kernel_size != 4:
            raise ValueError(f"GDN currently requires conv_kernel_size=4, got {self.conv_kernel_size}")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be finite and positive, got {self.norm_eps}")
        if self.output_gate_activation not in GDN_OUTPUT_GATE_ACTIVATIONS:
            raise ValueError(
                f"output_gate_activation must be one of {GDN_OUTPUT_GATE_ACTIVATIONS}, got {self.output_gate_activation!r}"
            )

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> "GDNConfig":
        """Build from a Hugging Face ``config.json`` mapping of a supported Qwen GDN model.

        The text configuration (``text_config`` when present, the top level otherwise, as for Qwen3.8-2.4T) selects
        the interpretation by its ``model_type``. Every ``linear_*`` key must be modeled; non-GDN keys (attention
        output gate, layer schedule, MoE, hyper-connections) belong to other layers and are ignored here.
        """
        if "text_config" in model_config:
            model_config = model_config["text_config"]
            if not isinstance(model_config, Mapping):
                raise TypeError("text_config must be a mapping")
        model_type = model_config.get("model_type")
        if model_type not in _GDN_MODEL_TYPES:
            raise ValueError(f"unsupported GDN model_type {model_type!r}; expected one of {sorted(_GDN_MODEL_TYPES)}")
        unmodeled = sorted(key for key in model_config if key.startswith("linear_") and key not in _LINEAR_KEYS)
        if unmodeled:
            raise ValueError(f"{model_type} has linear_* keys GDN does not model: {unmodeled}")
        try:
            # transformers builds the GDN convolution with ACT2FN[hidden_act]; the TT layer hardcodes SiLU.
            hidden_act = model_config["hidden_act"]
            if hidden_act != "silu":
                raise ValueError(f"{model_type} GDN convolution requires hidden_act='silu', got {hidden_act!r}")
            # The recurrent state dtype; the reference and the device keep the state in fp32.
            state_dtype = model_config.get("mamba_ssm_dtype", "float32")
            if state_dtype != "float32":
                raise ValueError(f"{model_type} GDN state requires mamba_ssm_dtype='float32', got {state_dtype!r}")
            return cls(
                hidden_size=int(model_config["hidden_size"]),
                num_key_heads=int(model_config["linear_num_key_heads"]),
                num_value_heads=int(model_config["linear_num_value_heads"]),
                head_k_dim=int(model_config["linear_key_head_dim"]),
                head_v_dim=int(model_config["linear_value_head_dim"]),
                conv_kernel_size=int(model_config["linear_conv_kernel_dim"]),
                norm_eps=float(model_config["rms_norm_eps"]),
                output_gate_activation=_output_gate_activation(model_type, model_config),
            )
        except KeyError as error:
            raise ValueError(f"missing {model_type} config field: {error.args[0]}") from error

    @property
    def group(self) -> int:
        """V heads per K head."""
        return self.num_value_heads // self.num_key_heads

    @property
    def q_dim(self) -> int:
        return self.num_key_heads * self.head_k_dim

    @property
    def k_dim(self) -> int:
        return self.num_key_heads * self.head_k_dim

    @property
    def v_dim(self) -> int:
        return self.num_value_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        """Channels of the fused ``[q | k | v]`` projection and convolution."""
        return self.q_dim + self.k_dim + self.v_dim


_GDN_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5_moe_text", "qwen4_exp_text"})
# linear_* keys GDNConfig models; any other linear_* key is rejected rather than silently ignored.
_LINEAR_KEYS = frozenset(
    {
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    }
)


def _output_gate_activation(model_type: str, model_config: Mapping[str, Any]) -> str:
    """Resolve the GDN output-gate activation the way the model's reference implementations do.

    transformers ``qwen3_5`` / ``qwen3_5_moe`` hard-wire silu and ignore ``output_gate_type``; vLLM reads it, maps
    ``swish`` to silu and defaults to silu. Values on which the two would disagree are rejected. transformers
    ``qwen4_exp`` uses ``output_gate_type or hidden_act`` and accepts only silu and sigmoid.
    """
    requested = model_config.get("output_gate_type")
    if model_type == "qwen4_exp_text":
        activation = requested or model_config["hidden_act"]
        if activation not in GDN_OUTPUT_GATE_ACTIVATIONS:
            raise ValueError(
                f"qwen4_exp_text output_gate_type must resolve to one of {GDN_OUTPUT_GATE_ACTIVATIONS}, got {activation!r}"
            )
        return activation
    if requested not in (None, "silu", "swish"):
        raise ValueError(
            f"{model_type} hard-wires the silu output gate in transformers; output_gate_type {requested!r} would "
            "disagree with it"
        )
    return "silu"
