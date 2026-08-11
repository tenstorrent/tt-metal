# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Canonical host-weight schema for Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig

KDA_COMMON_WEIGHT_NAMES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "dt_bias",
    "b_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
)
KDA_LOW_RANK_GATE_WEIGHT_NAMES = ("g_a_proj.weight", "g_b_proj.weight")
KDA_FULL_RANK_GATE_WEIGHT_NAMES = ("g_proj.weight",)


def required_kda_weight_names(config: KDAConfig) -> tuple[str, ...]:
    """Return the canonical layer-local checkpoint keys for ``config``."""
    gate_names = KDA_FULL_RANK_GATE_WEIGHT_NAMES if config.use_full_rank_gate else KDA_LOW_RANK_GATE_WEIGHT_NAMES
    return KDA_COMMON_WEIGHT_NAMES + gate_names


def expected_kda_weight_shapes(config: KDAConfig) -> dict[str, tuple[int, ...]]:
    """Return the canonical shape of every host weight for ``config``."""
    hidden = config.hidden_size
    key_rank = config.head_k_dim
    value_rank = config.head_v_dim
    heads = config.num_heads
    shapes = {
        "q_proj.weight": (config.q_dim, hidden),
        "k_proj.weight": (config.k_dim, hidden),
        "v_proj.weight": (config.v_dim, hidden),
        "q_conv1d.weight": (config.q_dim, 1, config.conv_kernel_size),
        "k_conv1d.weight": (config.k_dim, 1, config.conv_kernel_size),
        "v_conv1d.weight": (config.v_dim, 1, config.conv_kernel_size),
        "A_log": (1, 1, heads, 1),
        "f_a_proj.weight": (key_rank, hidden),
        "f_b_proj.weight": (heads * key_rank, key_rank),
        "dt_bias": (heads * key_rank,),
        "b_proj.weight": (heads, hidden),
        "o_norm.weight": (value_rank,),
        "o_proj.weight": (hidden, heads * value_rank),
    }
    if config.use_full_rank_gate:
        shapes["g_proj.weight"] = (heads * value_rank, hidden)
    else:
        shapes["g_a_proj.weight"] = (value_rank, hidden)
        shapes["g_b_proj.weight"] = (heads * value_rank, value_rank)
    return shapes


def normalize_kda_a_log(a_log: torch.Tensor, config: KDAConfig) -> torch.Tensor:
    """Normalize checkpoint head padding into canonical ``[1,1,H,1]`` form."""
    if a_log.numel() == config.num_heads:
        return a_log.reshape(1, 1, config.num_heads, 1)
    if config.num_heads == 96 and a_log.numel() == 128:
        return a_log.reshape(-1)[: config.num_heads].reshape(1, 1, config.num_heads, 1)
    raise ValueError(f"A_log has {a_log.numel()} entries; expected {config.num_heads} logical heads")


def validate_kda_weights(weights: Mapping[str, torch.Tensor], config: KDAConfig) -> None:
    """Validate the presence and shape of every canonical host weight."""
    for name, expected_shape in expected_kda_weight_shapes(config).items():
        try:
            weight = weights[name]
        except KeyError as error:
            raise ValueError(f"missing KDA weight: {name}") from error
        if tuple(weight.shape) != expected_shape:
            raise ValueError(f"{name} shape {tuple(weight.shape)} != {expected_shape}")


def normalize_kda_state_dict(state_dict: Mapping[str, torch.Tensor], config: KDAConfig) -> dict[str, torch.Tensor]:
    """Copy, canonicalize, and validate a layer-local host-weight mapping."""
    normalized = dict(state_dict)
    try:
        normalized["A_log"] = normalize_kda_a_log(normalized["A_log"], config)
    except KeyError as error:
        raise ValueError("missing KDA weight: A_log") from error
    validate_kda_weights(normalized, config)
    return normalized
