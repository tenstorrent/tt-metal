# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Galaxy weights dicts -> ``from_torch`` reference objects for the streamed-denoise port.

Maps the FULL Galaxy ``weights`` dict (the torch state-dict slices keyed
``model.layers.{i}.*`` / ``model.norm.dense.*`` / ``action_in_proj.*`` etc.) into the
plain-object attribute surface the vendored ``from_torch`` bodies pluck:

  * ``expert_reference_blocks(action_expert, ec)`` -> 18 reference AdaRMS blocks.
    Reuses the TARGET ``reference/torch_gemma.AdaRMSGemmaBlock`` (plan §7.5 / evaluator
    note #4) -- its attribute surface (``.pre_attn_mod_weight``, ``.attention.q_proj``,
    ``.mlp.gate_proj``, ...) is EXACTLY what the vendored ``from_torch`` reads. The
    ``action_expert`` dict is keyed ``model.layers.{i}.<local>`` so we slice the per-layer
    sub-dict (strip the prefix) and hand it to ``AdaRMSGemmaBlock(ec, layer_weights, i)``.
  * ``final_mod(action_expert)`` -> ``(weight, bias_or_None)`` from ``model.norm.dense.*``.
  * ``suffix_reference(pi0_projections, suffix_cfg)`` -> the TARGET
    ``reference/torch_suffix.Pi0_5SuffixEmbedding`` (it already exposes ``.action_in_weight``,
    ``.time_mlp_in_weight``, ... unconditionally for pi05 -- the precise G5 seam: the pi05
    SUBCLASS sets the time_mlp_* attrs unconditionally, unlike the base in its ``not pi05`` branch).

ZERO tt_symbiote imports.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock
from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"


def _layer_subdict(action_expert: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    """Slice the per-layer local-keyed sub-dict (strip ``model.layers.{i}.``)."""
    prefix = f"model.layers.{layer_idx}."
    sub = {}
    for k, v in action_expert.items():
        if k.startswith(prefix):
            sub[k[len(prefix) :]] = v
    if not sub:
        raise KeyError(f"no weights found for action-expert layer {layer_idx} (prefix {prefix!r})")
    return sub


def expert_reference_blocks(action_expert: Dict[str, torch.Tensor], ec, depth: int = 18):
    """Build ``depth`` reference AdaRMSGemmaBlock objects from the Galaxy action_expert dict."""
    return [AdaRMSGemmaBlock(ec, _layer_subdict(action_expert, i), i) for i in range(depth)]


def final_mod(action_expert: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Final-modulation Dense weight (required) + bias (OPTIONAL -> None if absent)."""
    w = action_expert["model.norm.dense.weight"]  # [2W, W] -> scale, shift
    b = action_expert.get("model.norm.dense.bias")  # OPTIONAL
    return w, b


def suffix_reference(pi0_projections: Dict[str, torch.Tensor], suffix_cfg):
    """Build the reference pi0.5 suffix object whose attributes the vendored
    TTNNPi05SuffixEmbedding.from_torch reads. Reuses the target Pi0_5SuffixEmbedding
    (which sets action_in/out + time_mlp_in/out attrs unconditionally for pi05)."""
    assert suffix_cfg.pi05, "suffix_reference requires a pi05 SuffixConfig"
    return Pi0_5SuffixEmbedding(suffix_cfg, pi0_projections)
