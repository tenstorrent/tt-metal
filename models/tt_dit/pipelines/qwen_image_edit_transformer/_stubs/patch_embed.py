# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of the QwenImage patch embedding (`img_in`): the pipeline packs the latent into
2x2 patches of 16 channels (64 features per token) and `img_in` projects them to the model dim:

    out = x @ W^T + b        W: [3072, 64]

On a mesh the weights are replicated (64 input features is a single 2-tile K; not worth splitting).
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _precise


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


class TtQwenPatchEmbed:
    def __init__(self, device, torch_module):
        self.device = device
        w = torch_module.weight.detach().to(torch.float32)
        b = torch_module.bias.detach().to(torch.float32) if torch_module.bias is not None else None
        self.w = _replicated(w.t(), device, ttnn.bfloat16)
        self.b = _replicated(b.reshape(1, -1), device) if b is not None else None
        self.hifi = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, input=None, **kwargs):
        x = input if input is not None else kwargs.get("hidden_states")
        if not isinstance(x, ttnn.Tensor):
            x = _replicated(x.to(torch.float32), self.device, ttnn.bfloat16)
        if _precise.ENABLED:
            return _precise.linear(x, self.w, bias=self.b)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        y = ttnn.linear(x, self.w, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        if self.b is not None:
            y = ttnn.add(y, self.b)
        return y


def build(device, torch_module=None):
    return TtQwenPatchEmbed(device, torch_module)


def patch_embed(device, torch_module=None):
    return TtQwenPatchEmbed(device, torch_module)
