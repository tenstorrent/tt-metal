from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AceConfig


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm-Zero.

    Conditioning vector -> (gamma, beta) applied to normalized x:
        y = LN(x) * (1 + gamma) + beta

    "Zero" init: last linear is initialized to zeros so the module starts as identity w.r.t. LN output.
    """

    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.eps = float(cfg.eps)
        self.d_model = int(cfg.d_model)
        self.cond_dim = int(cfg.cond_dim)

        self.cond_to_gb = nn.Linear(self.cond_dim, 2 * self.d_model, bias=True)
        nn.init.zeros_(self.cond_to_gb.weight)
        nn.init.zeros_(self.cond_to_gb.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, S, D]
        cond: [B, C]
        """
        x_norm = F.layer_norm(x, (self.d_model,), weight=None, bias=None, eps=self.eps)
        gb = self.cond_to_gb(cond)  # [B, 2D]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)  # [B, 1, D]
        beta = beta.unsqueeze(1)  # [B, 1, D]
        return x_norm * (1.0 + gamma) + beta


class MultiHeadSelfAttention(nn.Module):
    """
    Explicit QKV multi-head self-attention (no black-box attention APIs).

    Uses causal masking (typical transformer decode/prefill behavior).
    """

    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.d_model = int(cfg.d_model)
        self.n_heads = int(cfg.n_heads)
        self.d_head = int(cfg.d_head if cfg.d_head is not None else cfg.d_model // cfg.n_heads)

        self.wq = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=True)
        self.wk = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=True)
        self.wv = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=True)
        self.wo = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=True)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, S, D]
        attn_mask (optional): broadcastable to [B, 1, S, S], 1 = keep, 0 = mask
        """
        B, S, _D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, Dh]
        k = self.wk(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, Dh]
        v = self.wv(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, S, Dh]

        scale = 1.0 / math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]

        # Causal mask (lower triangular)
        causal = torch.tril(torch.ones((S, S), device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))

        if attn_mask is not None:
            keep = attn_mask.to(torch.bool)
            scores = scores.masked_fill(~keep, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)  # [B, H, S, Dh]
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.d_head)  # [B, S, H*Dh]
        return self.wo(ctx)


class GEGLUMLP(nn.Module):
    """
    MLP with GEGLU activation:
        u = W_up(x) -> split (a, b)
        y = GELU(a) * b
        out = W_down(y)
    """

    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.d_model = int(cfg.d_model)
        self.d_ff = int(cfg.d_ff)
        self.w_up = nn.Linear(self.d_model, 2 * self.d_ff, bias=True)
        self.w_down = nn.Linear(self.d_ff, self.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(F.gelu(a) * b)


class TransformerBlock(nn.Module):
    """
    x -> AdaLN -> Attention -> Residual
      -> AdaLN -> MLP -> Residual
    """

    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.adaln_attn = AdaLNZero(cfg)
        self.attn = MultiHeadSelfAttention(cfg)
        self.adaln_mlp = AdaLNZero(cfg)
        self.mlp = GEGLUMLP(cfg)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.adaln_attn(x, cond)
        x = x + self.attn(h, attn_mask=attn_mask)
        h2 = self.adaln_mlp(x, cond)
        x = x + self.mlp(h2)
        return x
