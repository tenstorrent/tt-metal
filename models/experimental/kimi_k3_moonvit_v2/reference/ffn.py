# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for the MoonViT2 vision FFN (`mlp_type="mlp2"`).

`fc1(gelu_tanh(fc0(x)))`, 1024 -> 4096 -> 1024, no biases. Each SP chip runs it on its own
`[1, 1, MAX_PATCHES / SP_DEVICES, 1024]` shard of the padded patch sequence. The FFN is row-wise, so
the shard needs no communication and the padding rows need no masking.
"""

import torch
from torch import nn

from models.experimental.kimi_k3_moonvit_v2.reference.weights import VisionFfnWeights
from models.experimental.kimi_k3_moonvit_v2.reference.config import KimiK3VisionConfig


class TorchVisionFfn(nn.Module):
    """Computes in `dtype` whatever the dtype of the weights or the input (the checkpoint and the
    device activations are bf16). `torch.float32` is the default golden."""

    def __init__(self, weights: VisionFfnWeights, cfg: type = KimiK3VisionConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        weights.validate(cfg)
        self.dtype = dtype
        self.fc0 = nn.Linear(cfg.VT_HIDDEN_SIZE, cfg.VT_INTERMEDIATE_SIZE, bias=cfg.FFN_BIAS, dtype=dtype)
        self.fc1 = nn.Linear(cfg.VT_INTERMEDIATE_SIZE, cfg.VT_HIDDEN_SIZE, bias=cfg.FFN_BIAS, dtype=dtype)
        self.activation = nn.GELU(approximate="tanh")  # gelu_pytorch_tanh
        with torch.no_grad():
            self.fc0.weight.copy_(weights.fc0)
            self.fc1.weight.copy_(weights.fc1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[1, 1, S, VT_HIDDEN_SIZE]` in, the same shape out, in `self.dtype`."""
        return self.fc1(self.activation(self.fc0(x.to(self.dtype))))
