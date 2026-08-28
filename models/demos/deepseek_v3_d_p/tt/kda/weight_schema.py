# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Canonical host-weight schema for Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import validate_kda_weights


def normalize_kda_a_log(a_log: torch.Tensor, config: KDAConfig) -> torch.Tensor:
    """Normalize checkpoint head padding into canonical ``[1,1,H,1]`` form."""
    if a_log.numel() == config.num_heads:
        return a_log.reshape(1, 1, config.num_heads, 1)
    if config.num_heads == 96 and a_log.numel() == 128:
        return a_log.reshape(-1)[: config.num_heads].reshape(1, 1, config.num_heads, 1)
    raise ValueError(f"A_log has {a_log.numel()} entries; expected {config.num_heads} logical heads")


def normalize_kda_state_dict(state_dict: Mapping[str, torch.Tensor], config: KDAConfig) -> dict[str, torch.Tensor]:
    """Copy, canonicalize, and validate a layer-local host-weight mapping."""
    normalized = dict(state_dict)
    try:
        normalized["A_log"] = normalize_kda_a_log(normalized["A_log"], config)
    except KeyError as error:
        raise ValueError("missing KDA weight: A_log") from error
    validate_kda_weights(normalized, config)
    return normalized
