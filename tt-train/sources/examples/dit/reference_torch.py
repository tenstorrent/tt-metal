# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Golden PyTorch DiT reference for the ttml bringup.

Deliberately structured to mirror the constraints of the planned ttml
implementation so layer-by-layer parity checks are 1:1:

- patchify/unpatchify happen on the host (dataloader), never in the graph;
  the model consumes and predicts patch tokens of shape [B, 1, T, D]
  (rank-4, matching ttml's tensor convention),
- adaLN modulation uses six separate zero-init linears per block instead of
  one chunked 6*D linear (ttml has no autograd split),
- LayerNorm carries no learnable affine inside blocks (modulation supplies
  scale/shift),
- timestep sinusoidal features are computed on the host and fed as an input,
- loss is MSE in token space (targets are patchified on the host too).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Host-side (dataloader) utilities — shared verbatim with the ttml pipeline.
# ---------------------------------------------------------------------------


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, 1, T, patch*patch*C] with T = (H/p)*(W/p)."""
    b, c, h, w = images.shape
    gh, gw = h // patch, w // patch
    x = images.reshape(b, c, gh, patch, gw, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)  # B, gh, gw, p, p, C
    return x.reshape(b, 1, gh * gw, patch * patch * c)


def unpatchify(tokens: torch.Tensor, patch: int, channels: int, height: int, width: int) -> torch.Tensor:
    """[B, 1, T, patch*patch*C] -> [B, C, H, W]. Inverse of patchify."""
    b = tokens.shape[0]
    gh, gw = height // patch, width // patch
    x = tokens.reshape(b, gh, gw, patch, patch, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, channels, height, width)


def timestep_features(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal features [B] -> [B, 1, 1, dim], fp32 on host."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb.reshape(t.shape[0], 1, 1, dim)


@dataclass
class DiffusionSchedule:
    """Linear-beta DDPM schedule; all tensors precomputed fp32 on host."""

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, dtype=torch.float32)
        alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_alphas_bar = alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1.0 - alphas_bar).sqrt()
        self.betas = betas
        self.alphas_bar = alphas_bar

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_bar[t].reshape(-1, 1, 1, 1)
        s = self.sqrt_one_minus_alphas_bar[t].reshape(-1, 1, 1, 1)
        return a * x0 + s * noise


# ---------------------------------------------------------------------------
# Model — every submodule maps 1:1 onto existing ttml modules/ops.
# ---------------------------------------------------------------------------


class Modulation(nn.Module):
    """One adaLN modulation branch: SiLU(c) -> Linear (zero-init)."""

    def __init__(self, dim: int, zero_init: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.nn.functional.silu(c))


class Attention(nn.Module):
    """Non-causal MHA; mirrors ttml MultiHeadAttention (fused qkv, SDPA)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, one, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, d // self.num_heads)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # non-causal
        out = out.transpose(1, 2).reshape(b, 1, t, d)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        # Six separate zero-init modulation linears (no chunk/split in ttml).
        self.shift_msa = Modulation(dim)
        self.scale_msa = Modulation(dim)
        self.gate_msa = Modulation(dim)
        self.shift_mlp = Modulation(dim)
        self.scale_mlp = Modulation(dim)
        self.gate_mlp = Modulation(dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x) * (1.0 + self.scale_msa(c)) + self.shift_msa(c)
        x = x + self.gate_msa(c) * self.attn(h)
        h = self.norm2(x) * (1.0 + self.scale_mlp(c)) + self.shift_mlp(c)
        x = x + self.gate_mlp(c) * self.mlp(h)
        return x


class DiT(nn.Module):
    """Class-conditional pixel/latent-space DiT over pre-patchified tokens.

    Inputs:
        tokens  [B, 1, T, in_dim]  — patchified noisy images (host-side)
        t_feats [B, 1, 1, dim]     — sinusoidal timestep features (host-side)
        labels  [B]                — class ids; num_classes is the CFG null id
    Output:
        [B, 1, T, in_dim] predicted noise, in token space.
    """

    def __init__(self, in_dim: int, dim: int, depth: int, num_heads: int, num_tokens: int, num_classes: int):
        super().__init__()
        self.patch_proj = nn.Linear(in_dim, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.label_emb = nn.Embedding(num_classes + 1, dim)  # +1 = CFG null class
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_shift = Modulation(dim)
        self.final_scale = Modulation(dim)
        self.final_proj = nn.Linear(dim, in_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, tokens: torch.Tensor, t_feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(tokens) + self.pos_emb
        c = self.t_mlp(t_feats) + self.label_emb(labels).reshape(labels.shape[0], 1, 1, -1)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_norm(x) * (1.0 + self.final_scale(c)) + self.final_shift(c)
        return self.final_proj(x)


def dit_s(in_dim: int, num_tokens: int, num_classes: int) -> DiT:
    return DiT(in_dim=in_dim, dim=384, depth=12, num_heads=6, num_tokens=num_tokens, num_classes=num_classes)


def dit_tiny(in_dim: int, num_tokens: int, num_classes: int) -> DiT:
    """Small enough to overfit a batch on CPU in seconds; same topology."""
    return DiT(in_dim=in_dim, dim=128, depth=4, num_heads=4, num_tokens=num_tokens, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Training step — identical math to the planned ttml loop.
# ---------------------------------------------------------------------------


def training_step(
    model: DiT,
    schedule: DiffusionSchedule,
    images: torch.Tensor,  # [B, C, H, W] in [-1, 1]
    labels: torch.Tensor,  # [B]
    patch: int,
    cfg_drop_prob: float = 0.1,
    null_class: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    b = images.shape[0]
    t = torch.randint(0, schedule.timesteps, (b,), generator=generator)
    noise = torch.randn(images.shape, generator=generator)
    noisy = schedule.add_noise(images, noise, t)

    if null_class is not None and cfg_drop_prob > 0:
        drop = torch.rand(b, generator=generator) < cfg_drop_prob
        labels = torch.where(drop, torch.full_like(labels, null_class), labels)

    tokens = patchify(noisy, patch)
    target = patchify(noise, patch)
    t_feats = timestep_features(t, model.pos_emb.shape[-1])
    pred = model(tokens, t_feats, labels)
    return torch.nn.functional.mse_loss(pred, target)


@torch.no_grad()
def sample_ddim(
    model: DiT,
    schedule: DiffusionSchedule,
    labels: torch.Tensor,
    shape: tuple[int, int, int],  # (C, H, W)
    patch: int,
    steps: int = 50,
    cfg_scale: float = 1.0,
    null_class: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    b = labels.shape[0]
    c, h, w = shape
    x = torch.randn((b, c, h, w), generator=generator)
    ts = torch.linspace(schedule.timesteps - 1, 0, steps).long()
    for i, t in enumerate(ts):
        t_batch = torch.full((b,), t.item(), dtype=torch.long)
        t_feats = timestep_features(t_batch, model.pos_emb.shape[-1])
        tokens = patchify(x, patch)
        eps = model(tokens, t_feats, labels)
        if cfg_scale != 1.0 and null_class is not None:
            null = torch.full_like(labels, null_class)
            eps_u = model(tokens, t_feats, null)
            eps = eps_u + cfg_scale * (eps - eps_u)
        eps = unpatchify(eps, patch, c, h, w)

        ab_t = schedule.alphas_bar[t]
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0 = x0.clamp(-1, 1)
        if i + 1 < len(ts):
            ab_prev = schedule.alphas_bar[ts[i + 1]]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        else:
            x = x0
    return x
