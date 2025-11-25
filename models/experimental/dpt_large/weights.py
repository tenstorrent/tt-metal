# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import Dict

import torch

from .config import DPTLargeConfig, DEFAULT_CONFIG

try:
    from transformers import DPTConfig, DPTForDepthEstimation
except Exception:  # pragma: no cover
    DPTConfig = None
    DPTForDepthEstimation = None

LOG = logging.getLogger(__name__)


def load_hf_state_dict(config: DPTLargeConfig = DEFAULT_CONFIG, pretrained: bool = True) -> Dict[str, torch.Tensor]:
    if DPTForDepthEstimation is None:
        raise ImportError("transformers is required for weight conversion.")
    if pretrained:
        model = DPTForDepthEstimation.from_pretrained(config.model_name)
    else:
        hf_cfg = DPTConfig(**config.to_hf_kwargs())
        model = DPTForDepthEstimation(hf_cfg)
    return model.state_dict()


def convert_weights_for_tt(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Map HF parameter names to TTNN-friendly names.

    For the current minimal pipeline we simply passthrough the state dict.
    This helper exists so that hardware-specific scripts can hook in and
    rewrite parameter layouts (e.g., sharded QKV) without touching callers.
    """

    converted: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        converted[name] = tensor
    LOG.info("Converted %d parameters for TT pipeline.", len(converted))
    return converted
