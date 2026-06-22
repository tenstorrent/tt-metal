# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The functions in this file are VERBATIM algorithm extractions from
# transformers `src/transformers/models/diffusion_gemma/` (Apache-2.0,
# © 2026 the HuggingFace Team): `generation_diffusion_gemma.py`
# (EntropyBoundSampler.accept_canvas / renoise_canvas,
# LinearTemperatureScheduleLogitsProcessor, StableAndConfidentStoppingCriteria)
# and `modeling_diffusion_gemma.py` (DiffusionGemmaRMSNorm,
# DiffusionGemmaSelfConditioning + the decoder soft-embedding step). They are
# vendored, with only the surrounding transformers framework stripped, as the
# **drift oracle** for #47468: ``tests/test_upstream_parity.py`` asserts our
# ``reference/`` primitives reproduce these bit-for-bit, so the reference cannot
# silently diverge from the released model.

from __future__ import annotations

import torch
import torch.nn.functional as F


# --- generation_diffusion_gemma.LinearTemperatureScheduleLogitsProcessor ----
def temperature_upstream(cur_step: int, t_min: float, t_max: float, max_denoising_steps: int) -> float:
    """``temperature = t_min + (t_max - t_min) * (cur_step / max_denoising_steps)``.

    HF iterates ``cur_step`` in reverse (``max_denoising_steps .. 1``); pass the
    reversed step index here.
    """
    return t_min + ((t_max - t_min) * (cur_step / max_denoising_steps))


# --- generation_diffusion_gemma.EntropyBoundSampler.accept_canvas -----------
def accept_canvas_upstream(logits: torch.Tensor, entropy_bound: float) -> torch.Tensor:
    """Return the accepted-token bool mask for ``logits`` ``[B, L, vocab]``."""
    dist = torch.distributions.Categorical(logits=logits)
    token_entropy = dist.entropy()  # (B, L)
    sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
    cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)
    sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= entropy_bound
    accepted_token_mask = torch.scatter(
        input=torch.zeros_like(sorted_selection_mask), dim=-1, index=sorted_indices, src=sorted_selection_mask
    )
    return accepted_token_mask


# --- generation_diffusion_gemma.StableAndConfidentStoppingCriteria ----------
def confident_upstream(logits: torch.Tensor, confidence_threshold: float) -> torch.Tensor:
    """Per-example confidence: ``mean(token_entropy) < confidence_threshold`` -> (B,)."""
    dist = torch.distributions.Categorical(logits=logits)
    token_entropy = dist.entropy()
    return torch.mean(token_entropy, dim=-1) < confidence_threshold


# --- modeling_diffusion_gemma.DiffusionGemmaRMSNorm -------------------------
def rmsnorm_upstream(hidden_states: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    """fp32 RMSNorm; ``weight=None`` is the scaleless (``with_scale=False``) variant."""
    out = hidden_states.float()
    out = out * torch.pow(out.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        out = out * weight.float()
    return out.type_as(hidden_states)


# --- modeling_diffusion_gemma soft-embedding + DiffusionGemmaSelfConditioning
def self_conditioning_upstream(
    inputs_embeds: torch.Tensor,
    prev_logits: torch.Tensor | None,
    embed_weight: torch.Tensor,
    *,
    pre_norm_w: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Decoder soft-embedding step + the self-conditioning gated MLP (gelu-tanh)."""
    if prev_logits is not None:
        soft = torch.matmul(prev_logits.softmax(dim=-1, dtype=torch.float32).to(embed_weight.dtype), embed_weight)
    else:
        soft = torch.zeros_like(inputs_embeds)
    normed = rmsnorm_upstream(soft, pre_norm_w, eps)
    gated = F.gelu(F.linear(normed, gate_w), approximate="tanh") * F.linear(normed, up_w)
    sc = F.linear(gated, down_w)
    return rmsnorm_upstream(inputs_embeds + sc, None, eps)  # post_norm is scaleless
