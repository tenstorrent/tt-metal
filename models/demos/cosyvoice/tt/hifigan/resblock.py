# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HiFT ResBlock: the dilated residual stack that does most of the vocoder's work.

Per dilation d in (1, 3, 5):

    x = x + convs2[d]( snake2[d]( convs1[d]( snake1[d](x) ) ) )

with convs1 carrying the dilation and convs2 always dilation 1, both "same"-padded.
HiFT instantiates 8 of these (2 upsample stages x 3 kernel sizes, plus 2 source
blocks), so this one class accounts for 48 conv1d calls and 48 Snake activations
per vocoder invocation -- the bulk of the vocoder's dispatch count.

Tensors are `[N, L, C]` throughout, per the convention in conv.py.
"""
from __future__ import annotations

import torch

import ttnn

from .conv import TtConv1d, extract_conv_weights
from .snake import TtSnake


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Same as cosyvoice.hifigan.generator.get_padding -- "same" padding."""
    return int((kernel_size * dilation - dilation) / 2)


class TtResBlock:
    def __init__(
        self,
        device,
        channels: int,
        kernel_size: int,
        dilations=(1, 3, 5),
        convs1=None,
        convs2=None,
        alphas1=None,
        alphas2=None,
        dtype=ttnn.bfloat16,
    ):
        """Weights may be supplied as torch modules (convs1/convs2) and alpha
        tensors, or omitted for a randomly-initialised block in tests."""
        self.device = device
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.n = len(self.dilations)

        def _conv(mod, k, d):
            if mod is not None:
                return TtConv1d.from_module(device, mod, dtype=dtype)
            w = torch.randn(channels, channels, k) * 0.02
            return TtConv1d(device, w, torch.zeros(channels), padding=get_padding(k, d), dilation=d, dtype=dtype)

        self.convs1 = [_conv(convs1[i] if convs1 else None, kernel_size, d) for i, d in enumerate(self.dilations)]
        self.convs2 = [_conv(convs2[i] if convs2 else None, kernel_size, 1) for i in range(self.n)]
        self.act1 = [
            TtSnake(device, alphas1[i] if alphas1 is not None else torch.ones(channels), dtype=dtype)
            for i in range(self.n)
        ]
        self.act2 = [
            TtSnake(device, alphas2[i] if alphas2 is not None else torch.ones(channels), dtype=dtype)
            for i in range(self.n)
        ]

    @classmethod
    def from_module(cls, device, module: torch.nn.Module, dtype=ttnn.bfloat16):
        """Build from a cosyvoice.hifigan.generator.ResBlock."""
        channels = (
            module.convs1[0].weight.shape[0]
            if hasattr(module.convs1[0], "weight")
            else extract_conv_weights(module.convs1[0])[0].shape[0]
        )
        dilations = [int(c.dilation[0]) for c in module.convs1]
        return cls(
            device,
            channels=channels,
            kernel_size=int(module.convs1[0].kernel_size[0]),
            dilations=dilations,
            convs1=list(module.convs1),
            convs2=list(module.convs2),
            alphas1=[a.alpha.detach() for a in module.activations1],
            alphas2=[a.alpha.detach() for a in module.activations2],
            dtype=dtype,
        )

    def __call__(self, x, length: int, batch_size: int = 1):
        """x: ttnn [N, L, C] -> ttnn [N, L, C]. Length is unchanged ("same" padding).

        OWNERSHIP: this frees only the intermediates it creates, never `x`. HiFT
        runs three ResBlocks over the *same* input per stage and averages them, so
        a block that deallocated its input would hand the next one a freed tensor.
        That is exactly what happened -- it passes in isolation and dies at
        integration with `TT_FATAL: Input Tensor A is not allocated`, which names
        the victim rather than the culprit.
        """
        cur = x
        for i in range(self.n):
            xt = self.act1[i](cur)
            xt, _ = self.convs1[i](xt, length, batch_size)
            xt = self.act2[i](xt)
            xt, _ = self.convs2[i](xt, length, batch_size)
            # The residual add is why the block is cheap to get subtly wrong: if
            # conv output layout ever stops matching x's, this broadcasts instead
            # of adding elementwise. Shapes are asserted in the PCC test.
            nxt = ttnn.add(cur, xt)
            ttnn.deallocate(xt)
            if cur is not x:  # ours to free; the caller's is not
                ttnn.deallocate(cur)
            cur = nxt
        return cur

    @staticmethod
    def torch_reference(x: torch.Tensor, convs1, convs2, alphas1, alphas2) -> torch.Tensor:
        """cosyvoice.hifigan.generator.ResBlock.forward, in [N, C, L]."""
        for i in range(len(convs1)):
            xt = TtSnake.torch_reference(x, alphas1[i])
            xt = convs1[i](xt)
            xt = TtSnake.torch_reference(xt, alphas2[i])
            xt = convs2[i](xt)
            x = xt + x
        return x
