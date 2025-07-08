# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-FileCopyrightText: Copyright 2024 The HuggingFace Team. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/embeddings.py
class FluxPosEmbed(torch.nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()

        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        for i in range(n_axes):
            cos, sin = self._get_1d_rotary_pos_embed(self.axes_dim[i], pos[:, i])
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)

        return freqs_cos, freqs_sin

    def _get_1d_rotary_pos_embed(
        self,
        dim: int,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert dim % 2 == 0

        range_ = torch.arange(0, dim, step=2, dtype=torch.float32, device=pos.device)
        freqs = 1.0 / (self.theta ** (range_[: dim // 2] / dim))
        freqs = torch.outer(pos, freqs)

        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()

        return freqs_cos, freqs_sin
