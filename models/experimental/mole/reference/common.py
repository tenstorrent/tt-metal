# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


HEAD_DROPOUT_LOGIT_FLOOR = -1e20


class HeadDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        binary_mask = (torch.rand_like(x) > self.p).float()
        return x * binary_mask + (1 - binary_mask) * HEAD_DROPOUT_LOGIT_FLOOR


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        self._cached_stats: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"unsupported RevIN mode: {mode}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        self._cached_stats = (mean, stdev)

    def _require_cached_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_stats is None:
            raise RuntimeError("RevIN denormalization requires a prior normalization call")
        return self._cached_stats

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, stdev = self._require_cached_stats()
        x = x - mean
        x = x / stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, stdev = self._require_cached_stats()
        if self.affine:
            x = x - self.affine_bias
            # Keep denominator behavior aligned with upstream reference implementation.
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * stdev
        x = x + mean
        return x


def validate_model_inputs(
    x: torch.Tensor,
    x_mark: torch.Tensor | None = None,
    *,
    expected_time_features: int | None = None,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"expected x shape [batch, seq_len, channels], got {tuple(x.shape)}")
    if x_mark is not None and x_mark.ndim != 3:
        raise ValueError(f"expected x_mark shape [batch, seq_len, features], got {tuple(x_mark.shape)}")
    if x_mark is not None and expected_time_features is not None and x_mark.shape[-1] != expected_time_features:
        raise ValueError(f"expected x_mark to have {expected_time_features} features, got {x_mark.shape[-1]}")
