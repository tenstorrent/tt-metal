# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `zero_pad2d`
(meituan-longcat/LongCat-Video's `vae.encoder.down_blocks.2.resample.0`,
a plain `torch.nn.ZeroPad2d((0, 1, 0, 1))` -- the pre-pad the Wan VAE's
downsample stage applies before its strided conv).

Stateless (no weight/bias): per the bring-up gate's TP principles this is
a pure elementwise/spatial op that needs no weight split and no
collective -- every mesh chip runs the identical `ttnn.pad` on its own
(replicated) input.
"""

from __future__ import annotations

import ttnn


class TtZeroPad2d:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        # `nn.ZeroPad2d.padding` is normalized to (left, right, top, bottom),
        # always applied to the tensor's last two dims regardless of
        # whether the input is batched (N, C, H, W) or unbatched (C, H, W).
        self.left, self.right, self.top, self.bottom = torch_module.padding

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        rank = len(x.shape)
        padding = [(0, 0)] * rank
        padding[-2] = (self.top, self.bottom)
        padding[-1] = (self.left, self.right)
        return ttnn.pad(x, padding, 0.0)


def build(device, torch_module) -> TtZeroPad2d:
    return TtZeroPad2d(device, torch_module)
