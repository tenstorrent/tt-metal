# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Self-conditioning gated MLP (pure-torch reference, #47461 loader / #47463 runtime).

The one net-new weight module beyond the Gemma-4 backbone (plan.md §3 N4). This
reference is reconciled **1:1** against transformers
``modeling_diffusion_gemma.DiffusionGemmaSelfConditioning`` + the soft-embedding
step in ``DiffusionGemmaDecoderModel.forward``:

  1. The decoder turns the previous step's (temperature-scaled) logits into a
     soft token embedding ``softmax(logits, fp32) @ embed_tokens.weight`` — a
     probability-weighted average of the (tied) token embedding table
     (:meth:`SelfConditioning.soft_embedding`). A per-example
     ``self_conditioning_mask`` zeroes the signal for examples that must not be
     conditioned (encoder/first step); the module is **decoder-only** — the
     encoder (prefill/commit causal passes) has no self-conditioning at all.
  2. The module itself (:meth:`forward`) is a Gemma gated MLP wrapped in
     pre/post RMSNorm::

        normed   = pre_norm(signal)                       # RMSNorm, with scale
        sc       = down_proj(act(gate_proj(normed)) * up_proj(normed))
        out      = post_norm(inputs_embeds + sc)           # RMSNorm, SCALELESS

Checkpoint weights (verified, `convert_diffusion_gemma_weights.py` /
`model.safetensors.index.json`): ``model.decoder.self_conditioning.{pre_norm,
gate_proj, up_proj, down_proj}.weight``. ``post_norm`` is scaleless so it has
**no** weight in the checkpoint. ``intermediate_size`` is the backbone's dense
``intermediate_size`` (2112), **not** ``moe_intermediate_size``.

NOTE: when the signal is zero (first denoise step / masked example) the output is
``post_norm(inputs_embeds)`` — NOT ``inputs_embeds`` unchanged. The decoder always
post-normalizes its input embeddings through this module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionGemmaRMSNorm(nn.Module):
    """RMSNorm matching transformers ``DiffusionGemmaRMSNorm`` exactly.

    fp32 normalization; scales by ``weight`` **directly** (not the Gemma2/3
    ``1 + weight`` convention). ``with_scale=False`` drops the learnable weight
    (the scaleless ``post_norm``).
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = hidden_states.float()
        out = out * torch.pow(out.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            out = out * self.weight.float()
        return out.type_as(hidden_states)


class SelfConditioning(nn.Module):
    """Gated-MLP self-conditioning module (mirror of ``DiffusionGemmaSelfConditioning``).

    The token embedding table is shared with the backbone (tied), so it is passed
    in at call time (``embedding_weight`` ``[vocab, hidden]``) rather than stored.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        *,
        eps: float = 1e-6,
        activation: str = "gelu_pytorch_tanh",
    ):
        super().__init__()
        inter = intermediate_size if intermediate_size is not None else hidden_size
        self.pre_norm = DiffusionGemmaRMSNorm(hidden_size, eps=eps, with_scale=True)
        self.post_norm = DiffusionGemmaRMSNorm(hidden_size, eps=eps, with_scale=False)  # scaleless
        self.gate_proj = nn.Linear(hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        # config.hidden_activation; gemma4 backbone uses gelu_pytorch_tanh.
        self.activation = activation

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation in ("gelu_pytorch_tanh", "gelu_tanh"):
            return F.gelu(x, approximate="tanh")
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"unknown activation {self.activation!r}")

    @staticmethod
    def soft_embedding(
        prev_logits: torch.Tensor,
        embedding_weight: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Probability-weighted average of token embeddings from prev-step logits.

        ``prev_logits`` ``[B, L, vocab]`` -> ``[B, L, hidden]``. Matches the decoder
        (``modeling_diffusion_gemma.py:1278-1281``):
        ``(softmax(dim=-1, fp32) @ embed_tokens.weight) * embed_scale``, where
        ``embed_scale = hidden_size ** 0.5`` (the same scale the tied
        ``DiffusionGemmaTextScaledWordEmbedding`` applies to token embeddings). NO
        temperature here — HF feeds the already temperature-scaled logits. The scale
        matters: the soft signal then feeds the self-conditioning ``pre_norm`` whose
        ``eps`` floor does NOT normalize the factor away at the tiny soft-RMS of a
        262k-vocab softmax. A one-hot row therefore yields ``embed_scale * emb[row]``.
        ``mask`` ``[B]`` is the per-example ``self_conditioning_mask`` (applied AFTER
        the scale, matching canonical order).
        """
        embed_scale = embedding_weight.shape[-1] ** 0.5  # = hidden_size**0.5
        probs = prev_logits.softmax(dim=-1, dtype=torch.float32).to(embedding_weight.dtype)
        soft = (probs @ embedding_weight) * embed_scale
        if mask is not None:
            soft = soft * mask.to(soft.dtype)[:, None, None]
        return soft

    def forward(self, inputs_embeds: torch.Tensor, self_conditioning_signal: torch.Tensor) -> torch.Tensor:
        """post_norm(inputs_embeds + gated_mlp(pre_norm(signal))) — the module itself."""
        normed = self.pre_norm(self_conditioning_signal)
        sc = self.down_proj(self._act(self.gate_proj(normed)) * self.up_proj(normed))
        return self.post_norm(inputs_embeds + sc)

    @torch.no_grad()
    def load_from_state_dict(self, state: dict) -> "SelfConditioning":
        """Load the 4 checkpoint weights (short keys ``{w}.weight``) with shape checks.

        ``state`` is the self-conditioning sub-dict from
        ``weight_mapping.remap_state_dict`` (keys ``pre_norm.weight``,
        ``gate_proj.weight``, ``up_proj.weight``, ``down_proj.weight``). Raises on a
        missing key or shape mismatch — exactly the "catch missing/renamed weights"
        guard #47461 wants. ``post_norm`` is scaleless and has no checkpoint weight.
        """
        targets = {
            "pre_norm.weight": self.pre_norm.weight,
            "gate_proj.weight": self.gate_proj.weight,
            "up_proj.weight": self.up_proj.weight,
            "down_proj.weight": self.down_proj.weight,
        }
        missing = [k for k in targets if k not in state]
        if missing:
            raise KeyError(f"self-conditioning checkpoint missing keys: {missing}; got {sorted(state)}")
        for name, param in targets.items():
            w = state[name]
            if tuple(w.shape) != tuple(param.shape):
                raise ValueError(
                    f"self_conditioning.{name}: checkpoint {tuple(w.shape)} != module {tuple(param.shape)}"
                )
            param.copy_(w.to(param.dtype))
        return self

    def condition(
        self,
        inputs_embeds: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        embedding_weight: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> torch.Tensor:
        """Convenience: build the soft-embedding signal then apply the module.

        ``enabled=False`` or ``prev_logits is None`` (first denoise step / encoder)
        -> zero signal, so the result is ``post_norm(inputs_embeds)`` (the decoder
        still post-normalizes its input embeddings — it is NOT an identity).
        """
        if prev_logits is None or not enabled:
            signal = torch.zeros_like(inputs_embeds)
        else:
            signal = self.soft_embedding(prev_logits, embedding_weight)
        return self.forward(inputs_embeds, signal)
