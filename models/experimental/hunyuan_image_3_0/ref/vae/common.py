# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Z_CHANNELS = 32
BLOCK_IN_CHANNELS = 1024
MID_CHANNELS = 1024
LATENT_T = 1
LATENT_H = 64
LATENT_W = 64
NUM_GROUPS = 32
GN_EPS = 1e-6


class Conv3d(nn.Conv3d):
    """Symmetric-padded Conv3d; chunks along T when activation memory exceeds ~2 GiB."""

    def forward(self, input: Tensor) -> Tensor:
        _b, c, t, h, w = input.shape
        memory_count = (c * t * h * w) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i, chunk in enumerate(chunks):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunk,
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][:, :, -self.padding[0] :]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][:, :, : self.padding[0]]
                else:
                    padded_chunk = chunk
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = [super().forward(padded_chunk) for padded_chunk in padded_chunks]
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        return super().forward(input)
