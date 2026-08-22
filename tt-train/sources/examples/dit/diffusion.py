# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side diffusion utilities (numpy) shared by the ttml training loop.

Same math as the torch versions in reference_torch.py; kept torch-free so the
device training loop only needs numpy on the host.
"""

from __future__ import annotations

import numpy as np


def patchify(images: np.ndarray, patch: int) -> np.ndarray:
    """[B, C, H, W] -> [B, 1, T, patch*patch*C]."""
    b, c, h, w = images.shape
    gh, gw = h // patch, w // patch
    x = images.reshape(b, c, gh, patch, gw, patch)
    x = x.transpose(0, 2, 4, 3, 5, 1)
    return x.reshape(b, 1, gh * gw, patch * patch * c)


def unpatchify(tokens: np.ndarray, patch: int, channels: int, height: int, width: int) -> np.ndarray:
    """[B, 1, T, patch*patch*C] -> [B, C, H, W]."""
    b = tokens.shape[0]
    gh, gw = height // patch, width // patch
    x = tokens.reshape(b, gh, gw, patch, patch, channels)
    x = x.transpose(0, 5, 1, 3, 2, 4)
    return x.reshape(b, channels, height, width)


def timestep_features(t: np.ndarray, dim: int, max_period: int = 10000) -> np.ndarray:
    """[B] int -> [B, 1, 1, dim] fp32 sinusoidal features."""
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half)
    args = t.astype(np.float32)[:, None] * freqs[None]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1).astype(np.float32)
    return emb.reshape(t.shape[0], 1, 1, dim)


class DiffusionSchedule:
    """Linear-beta DDPM schedule, fp32."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas_bar = np.cumprod(1.0 - self.betas, axis=0)
        self.sqrt_alphas_bar = np.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1.0 - self.alphas_bar)

    def add_noise(self, x0: np.ndarray, noise: np.ndarray, t: np.ndarray) -> np.ndarray:
        a = self.sqrt_alphas_bar[t].reshape(-1, 1, 1, 1)
        s = self.sqrt_one_minus_alphas_bar[t].reshape(-1, 1, 1, 1)
        return (a * x0 + s * noise).astype(np.float32)


def make_training_batch(
    images: np.ndarray,  # [B, C, H, W] fp32 in [-1, 1]
    labels: np.ndarray,  # [B] int
    schedule: DiffusionSchedule,
    patch: int,
    model_dim: int,
    rng: np.random.Generator,
    cfg_drop_prob: float = 0.1,
    null_class: int | None = None,
):
    """Returns (tokens, t_feats, labels_onehot, target) as numpy arrays ready
    for Tensor.from_numpy: [B,1,T,in], [B,1,1,dim], [B,1,1,null_class+1] fp32
    one-hot, [B,1,T,in]. One-hot (not ids): the model embeds labels via
    one-hot @ W because ttnn embedding_backward can't take a single id."""
    b = images.shape[0]
    t = rng.integers(0, schedule.timesteps, size=b)
    noise = rng.standard_normal(images.shape).astype(np.float32)
    noisy = schedule.add_noise(images, noise, t)

    if null_class is not None and cfg_drop_prob > 0:
        drop = rng.random(b) < cfg_drop_prob
        labels = np.where(drop, null_class, labels)

    tokens = patchify(noisy, patch)
    target = patchify(noise, patch)
    t_feats = timestep_features(t, model_dim)
    labels_onehot = one_hot(labels, (null_class if null_class is not None else int(labels.max())) + 1)
    return tokens, t_feats, labels_onehot, target


def one_hot(labels: np.ndarray, num_columns: int) -> np.ndarray:
    """[B] int -> [B, 1, 1, num_columns] fp32 one-hot."""
    b = labels.shape[0]
    out = np.zeros((b, num_columns), dtype=np.float32)
    out[np.arange(b), labels.astype(np.int64)] = 1.0
    return out.reshape(b, 1, 1, num_columns)
