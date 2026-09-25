# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the QwenImage text-stream MLP (`transformer_blocks.N.txt_mlp`,
diffusers FeedForward, gelu-approximate). Same class as img_mlp, so it uses the FeedForward port:
net[0].proj COLUMN-parallel, GELU(tanh) local, net[2] ROW-parallel + all_reduce, bias once after.
"""

from __future__ import annotations

from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.feed_forward import TtQwenFeedForward


def build(device, torch_module=None):
    return TtQwenFeedForward(device, torch_module)


def mlp(device, torch_module=None):
    return TtQwenFeedForward(device, torch_module)
