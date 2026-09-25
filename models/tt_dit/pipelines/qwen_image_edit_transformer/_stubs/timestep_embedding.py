# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of the timestep MLP (`time_text_embed.timestep_embedder`, diffusers
TimestepEmbedding: linear_1 (256 -> 3072) -> SiLU -> linear_2 (3072 -> 3072)).

Weights replicated (this runs once per denoising step on a [B, 256] input). Matmul inputs are bf16
with float32 accumulation; SiLU in float32.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _precise


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


def _act(name):
    if name is None:
        return None
    if isinstance(name, torch.nn.SiLU):
        return ttnn.silu
    if isinstance(name, torch.nn.GELU):
        return ttnn.gelu
    if isinstance(name, torch.nn.ReLU):
        return ttnn.relu
    if isinstance(name, torch.nn.Mish):
        return ttnn.mish
    raise NotImplementedError(f"activation {type(name).__name__} not ported")


class TtTimestepEmbedding:
    def __init__(self, device, torch_module):
        self.device = device
        m = torch_module
        assert getattr(m, "cond_proj", None) is None, "cond_proj path not ported"

        def lin(l):
            w = _replicated(l.weight.detach().to(torch.float32).t(), device, ttnn.bfloat16)
            b = _replicated(l.bias.detach().to(torch.float32).reshape(1, -1), device) if l.bias is not None else None
            return w, b

        self.l1 = lin(m.linear_1)
        self.l2 = lin(m.linear_2)
        self.act = _act(m.act)
        self.post_act = _act(m.post_act)
        self.hifi = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _lin(self, x, wb):
        w, b = wb
        if _precise.ENABLED:
            return _precise.linear(x, w, bias=b)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        y = ttnn.linear(x, w, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        return ttnn.add(y, b) if b is not None else y

    def __call__(self, sample, condition=None, **_unused):
        assert condition is None, "cond_proj path not ported"
        x = sample
        if not isinstance(x, ttnn.Tensor):
            x = _replicated(x.to(torch.float32), self.device)
        x = self._lin(x, self.l1)
        if self.act is not None:
            x = self.act(x)
        x = self._lin(x, self.l2)
        if self.post_act is not None:
            x = self.post_act(x)
        return x


def build(device, torch_module=None):
    return TtTimestepEmbedding(device, torch_module)


def timestep_embedding(device, torch_module=None):
    return TtTimestepEmbedding(device, torch_module)
