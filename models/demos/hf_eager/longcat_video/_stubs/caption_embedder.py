# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn port of `caption_embedder` (meituan-longcat/LongCat-Video's
`dit.y_embedder`, class `CaptionEmbedder` in the vendored
`longcat_video/modules/blocks.py`).

The torch reference is a 2-layer MLP projecting UMT5 caption embeddings
(`in_channels`) to the DiT's `hidden_size`:

    y_proj = Sequential(Linear(in_channels, hidden_size),
                         GELU(approximate="tanh"),
                         Linear(hidden_size, hidden_size))
    forward(caption):           # caption: [B, 1, N_token, in_channels]
        return y_proj(caption)

No token-dropout / classifier-free-guidance branch exists in this checkpoint's
`CaptionEmbedder` (unlike the PixArt-style version with `uncond_prob` /
`token_drop`), so the port is a straight two-matmul MLP.
"""

from __future__ import annotations

import ttnn


class TtCaptionEmbedder:
    def __init__(self, device: ttnn.Device, torch_module) -> None:
        self.device = device
        self.dtype = ttnn.bfloat16

        state = torch_module.state_dict()

        def _weight(key):
            # nn.Linear.weight is [out_features, in_features]; ttnn.linear's
            # second operand expects [in_features, out_features].
            return ttnn.from_torch(
                state[key].transpose(0, 1).contiguous(),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        def _bias(key):
            return ttnn.from_torch(
                state[key].reshape(1, -1),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        self.w0 = _weight("y_proj.0.weight")
        self.b0 = _bias("y_proj.0.bias")
        self.w1 = _weight("y_proj.2.weight")
        self.b1 = _bias("y_proj.2.bias")

        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, caption: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.linear(caption, self.w0, bias=self.b0, compute_kernel_config=self.ckc)
        x = ttnn.gelu(x, variant=ttnn.GeluVariant.Tanh)
        x = ttnn.linear(x, self.w1, bias=self.b1, compute_kernel_config=self.ckc)
        return x


def build(device: ttnn.Device, torch_module) -> TtCaptionEmbedder:
    return TtCaptionEmbedder(device, torch_module)
