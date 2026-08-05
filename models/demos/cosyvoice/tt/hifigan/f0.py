# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ConvRNNF0Predictor: mel -> fundamental frequency contour.

Despite the name there is no RNN. It is five weight_norm'd Conv1d(k=3, pad=1)
layers each followed by ELU, then a Linear(512 -> 1) and an absolute value:

    f0 = |Linear(ELU(Conv)^5(mel))|

The `abs()` at the end is what makes f0 non-negative, and it interacts with
SineGen's voiced/unvoiced threshold of 10 Hz -- everything below that is treated
as unvoiced, so small errors near zero are harmless while sign errors are not.

Tensors are channels-last `[B, T, C]`, per conv.py's convention.
"""
from __future__ import annotations

import torch

import ttnn

from .conv import TtConv1d


class TtF0Predictor:
    def __init__(self, device, condnet_convs, classifier_weight, classifier_bias, dtype=ttnn.bfloat16):
        self.device = device
        self.convs = [TtConv1d.from_module(device, m, dtype=dtype) for m in condnet_convs]
        # Linear stored transposed for ttnn.linear.
        self.weight = ttnn.from_torch(
            classifier_weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias = ttnn.from_torch(
            classifier_bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    @classmethod
    def from_module(cls, device, module: torch.nn.Module, dtype=ttnn.bfloat16):
        """Build from cosyvoice.hifigan.f0_predictor.ConvRNNF0Predictor.

        `condnet` is a Sequential of alternating Conv1d and ELU, so the convs are
        the even-indexed entries.
        """
        convs = [
            m
            for m in module.condnet
            if isinstance(m, torch.nn.Conv1d) or hasattr(m, "weight_v") or hasattr(m, "parametrizations")
        ]
        return cls(device, convs, module.classifier.weight, module.classifier.bias, dtype=dtype)

    def __call__(self, mel, length: int, batch_size: int = 1):
        """mel: ttnn [B, T, 80] -> f0 ttnn [B, T, 1]."""
        x = mel
        for i, conv in enumerate(self.convs):
            out, _ = conv(x, length, batch_size)
            if i:
                ttnn.deallocate(x)
            x = ttnn.elu(out, alpha=1.0)  # alpha is keyword-only in this binding
            ttnn.deallocate(out)
        f0 = ttnn.linear(x, self.weight, bias=self.bias)
        ttnn.deallocate(x)
        out = ttnn.abs(f0)
        ttnn.deallocate(f0)
        return out

    @staticmethod
    def torch_reference(mel: torch.Tensor, condnet: torch.nn.Module, classifier: torch.nn.Module) -> torch.Tensor:
        """ConvRNNF0Predictor.forward verbatim: [B, 80, T] -> [B, T]."""
        x = condnet(mel)
        x = x.transpose(1, 2)
        return torch.abs(classifier(x).squeeze(-1))
