# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Self-conditioning gated MLP (pure-torch reference, #47461 loader / #47463 runtime).

The one net-new weight module beyond the Gemma-4 backbone (plan.md §3 N4). Per
denoise step the previous step's logits are turned into a soft token embedding
(softmax -> probability-weighted average of the token embedding table) and fed
through a small **gated MLP**, whose output is added to the canvas input
embeddings. It is active **only during denoise** and **zeroed on encoder
passes** (prefill / commit).

This reference pins the algorithm (soft-embedding + gate + encoder-zeroing) and
its shapes; the loader that maps the real checkpoint weights is #47461 and the
exact activation / projection layout is reconciled against
``modeling_diffusion_gemma.py`` once importable (#47468).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfConditioning(nn.Module):
    """Gated-MLP self-conditioning module.

    The token embedding table is shared with the backbone (tied), so it is
    passed in at call time as ``embedding_weight`` ``[vocab, hidden]`` rather
    than duplicated here.
    """

    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None, activation: str = "gelu"):
        super().__init__()
        inter = intermediate_size or hidden_size
        self.gate_proj = nn.Linear(hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        # TODO(confirm): exact activation (gelu vs silu/SwiGLU) vs HF reference.
        self.activation = activation

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"unknown activation {self.activation!r}")

    def soft_embedding(
        self,
        prev_logits: torch.Tensor,
        embedding_weight: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Probability-weighted average of token embeddings from prev-step logits.

        ``prev_logits``: ``[B, L, vocab]`` -> ``[B, L, hidden]``. One-hot logits
        reproduce exactly that token's embedding row.
        """
        probs = F.softmax(prev_logits / temperature, dim=-1)
        return probs @ embedding_weight

    def forward(
        self,
        prev_logits: torch.Tensor,
        embedding_weight: torch.Tensor,
        *,
        enabled: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return the self-conditioning delta added to canvas embeddings.

        ``enabled=False`` (encoder pass: prefill / commit) -> exact zeros, so the
        encoder forward is unperturbed.
        """
        batch, length, _ = prev_logits.shape
        hidden = embedding_weight.shape[-1]
        if not enabled:
            return prev_logits.new_zeros(batch, length, hidden)
        soft = self.soft_embedding(prev_logits, embedding_weight, temperature=temperature)
        gated = self._act(self.gate_proj(soft)) * self.up_proj(soft)
        return self.down_proj(gated)
