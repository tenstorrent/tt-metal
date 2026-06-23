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
#
# NOTE: ``transformers`` ships ``diffusion_gemma`` since **5.12**, so the
# AUTHORITATIVE parity guard is now ``tests/test_real_transformers_parity.py``,
# which tests our ``reference/`` against the **actually installed** classes (can't
# drift from a frozen copy; auto-tracks future transformers). This vendored module
# is the **fallback** for envs without ``diffusion_gemma`` (old CI / the 4.53 LTX
# env). Keep it byte-faithful; prefer the real-class test where 5.12+ is present.

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


# --- generation_diffusion_gemma.StableAndConfidentStoppingCriteria ----------
class StableAndConfidentUpstream:
    """Verbatim of HF ``StableAndConfidentStoppingCriteria``: stop when the argmax
    canvas has been stable across ``stability_threshold`` steps AND the mean
    per-example token entropy (of the processed logits) is below
    ``confidence_threshold``. Per-example (returns a [B] bool); stateful rolling
    argmax buffer init to -1."""

    def __init__(self, stability_threshold: int, confidence_threshold: float):
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.argmax_canvas_history = None

    def __call__(self, argmax_canvas: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.stability_threshold == 0:
            stable = torch.ones((logits.shape[0],), device=logits.device, dtype=torch.bool)
        else:
            if self.argmax_canvas_history is None:
                self.argmax_canvas_history = torch.full(
                    (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                    -1,
                    dtype=argmax_canvas.dtype,
                    device=argmax_canvas.device,
                )
            stable = (self.argmax_canvas_history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, shifts=-1, dims=0)
            self.argmax_canvas_history[-1] = argmax_canvas
        token_entropy = torch.distributions.Categorical(logits=logits).entropy()
        confident = torch.mean(token_entropy, dim=-1) < self.confidence_threshold
        return stable & confident


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
        # canonical: ( softmax @ embed_tokens.weight ) * embed_tokens.embed_scale, embed_scale = hidden**0.5
        soft = torch.matmul(prev_logits.softmax(dim=-1, dtype=torch.float32).to(embed_weight.dtype), embed_weight) * (
            embed_weight.shape[-1] ** 0.5
        )
    else:
        soft = torch.zeros_like(inputs_embeds)
    normed = rmsnorm_upstream(soft, pre_norm_w, eps)
    gated = F.gelu(F.linear(normed, gate_w), approximate="tanh") * F.linear(normed, up_w)
    sc = F.linear(gated, down_w)
    return rmsnorm_upstream(inputs_embeds + sc, None, eps)  # post_norm is scaleless
