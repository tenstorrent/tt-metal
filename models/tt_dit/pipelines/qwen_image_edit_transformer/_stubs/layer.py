# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of the QwenImage block LayerNorm (`transformer_blocks.N.img_norm1` and its
siblings img_norm2 / txt_norm1 / txt_norm2: nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)):

    out = (x - mean(x)) * rsqrt(var(x) + eps) (* weight + bias when affine)

Tensor parallel: a norm reduces over the full hidden dim, so any weight/bias is REPLICATED and every
chip computes the full (replicated) output; no collective. Computed in float32.
"""

from __future__ import annotations

import torch

import ttnn


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


class TtQwenLayerNorm:
    def __init__(self, device, torch_module):
        self.device = device
        self.eps = float(torch_module.eps)
        w = getattr(torch_module, "weight", None)
        b = getattr(torch_module, "bias", None)
        self.w = _replicated(w.detach().to(torch.float32).reshape(1, -1), device) if w is not None else None
        self.b = _replicated(b.detach().to(torch.float32).reshape(1, -1), device) if b is not None else None
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
            x = _replicated(x.to(torch.float32), self.device)
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        mu = ttnn.mean(x, dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        xc = ttnn.subtract(x, mu)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        y = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, self.eps)))
        if self.w is not None:
            y = ttnn.multiply(y, self.w)
        if self.b is not None:
            y = ttnn.add(y, self.b)
        return y


def build(device, torch_module=None):
    return TtQwenLayerNorm(device, torch_module)


def layer(device, torch_module=None):
    return TtQwenLayerNorm(device, torch_module)
