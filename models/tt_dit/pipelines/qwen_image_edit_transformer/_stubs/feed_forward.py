# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the QwenImage FeedForward (`transformer_blocks.N.img_mlp` /
`txt_mlp`, diffusers FeedForward with activation_fn="gelu-approximate"):

    out = net[2](GELU_tanh(net[0].proj(x)))        3072 -> 12288 -> 3072

Tensor parallel (TP = mesh size): net[0].proj is COLUMN-parallel (hidden features split, bias split
with it), GELU is local, net[2] is ROW-parallel followed by all_reduce; its bias is replicated and
added once after the reduce. Matmul inputs are bf16 with float32 accumulation.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _ccl, _precise


def _is_mesh(device):
    return isinstance(device, ttnn.MeshDevice) and device.get_num_devices() > 1


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


def _sharded(t, device, dim, dtype=ttnn.bfloat16):
    if _is_mesh(device):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim),
        )
    return _replicated(t, device, dtype=dtype)


class TtQwenFeedForward:
    def __init__(self, device, torch_module):
        self.device = device
        self.tp = device.get_num_devices() if _is_mesh(device) else 1
        proj, out = torch_module.net[0].proj, torch_module.net[2]
        approx = getattr(torch_module.net[0], "approximate", "tanh")
        self.variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate

        w1 = proj.weight.detach().to(torch.float32)
        b1 = proj.bias.detach().to(torch.float32)
        self.w1 = _sharded(w1.t(), device, dim=-1)
        self.b1 = _sharded(b1.reshape(1, -1), device, dim=-1, dtype=ttnn.float32)
        w2 = out.weight.detach().to(torch.float32)
        b2 = out.bias.detach().to(torch.float32)
        self.w2 = _sharded(w2.t(), device, dim=-2)
        self.b2 = _replicated(b2.reshape(1, -1), device)

        self.hifi = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, hidden_states, **_unused):
        x = hidden_states
        if not isinstance(x, ttnn.Tensor):
            x = _replicated(x.to(torch.float32), self.device, ttnn.bfloat16)
        if _precise.ENABLED:
            h = ttnn.gelu(_precise.linear(x, self.w1, bias=self.b1), variant=self.variant)
            y = _precise.linear(h, self.w2)
            if self.tp > 1:
                y = _ccl.all_reduce(y, self.device)
            return ttnn.add(y, self.b2)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        h = ttnn.linear(x, self.w1, bias=self.b1, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        h = ttnn.gelu(h, variant=self.variant)
        y = ttnn.linear(ttnn.typecast(h, ttnn.bfloat16), self.w2, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        if self.tp > 1:
            y = _ccl.all_reduce(y, self.device)
        return ttnn.add(y, self.b2)


def build(device, torch_module=None):
    return TtQwenFeedForward(device, torch_module)


def feed_forward(device, torch_module=None):
    return TtQwenFeedForward(device, torch_module)
