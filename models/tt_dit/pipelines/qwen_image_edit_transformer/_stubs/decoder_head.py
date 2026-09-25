# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the QwenImage decoder head (`proj_out`):

    out = hidden_states @ W^T + b        W: [patch*patch*out_channels, inner_dim] = [64, 3072]

Tensor parallel (TP = mesh size): COLUMN-parallel on the output features, followed by all_gather.
64 outputs / 8 chips is 8 columns per chip, which is not tile-aligned, so the output features are
zero-padded up to a multiple of TP*32 (each chip owns one whole tile of columns). After the gather
the padding columns are sliced off and the bias (replicated) is added once.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _ccl, _precise

_TILE = 32


def _is_mesh(device):
    return isinstance(device, ttnn.MeshDevice) and device.get_num_devices() > 1


def _replicated(t, device, dtype):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


class TtQwenDecoderHead:
    def __init__(self, device, torch_module):
        self.device = device
        w = torch_module.weight.detach().to(torch.float32)  # [N, K]
        b = torch_module.bias.detach().to(torch.float32) if torch_module.bias is not None else None
        self.out_features, self.in_features = w.shape
        self.tp = device.get_num_devices() if _is_mesh(device) else 1

        wt = w.t()  # [K, N]
        if self.tp > 1:
            chunk = _TILE * self.tp
            n_pad = -(-self.out_features // chunk) * chunk
            if n_pad != self.out_features:
                wt = torch.nn.functional.pad(wt, (0, n_pad - self.out_features))
            self.w = ttnn.from_torch(
                wt.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
            )
        else:
            self.w = _replicated(wt, device, ttnn.bfloat16)
        self.b = _replicated(b.reshape(1, -1), device, ttnn.float32) if b is not None else None

        self.hifi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x, **_unused):
        if not isinstance(x, ttnn.Tensor):
            x = _replicated(x.to(torch.float32), self.device, ttnn.bfloat16)
        if _precise.ENABLED:
            y = _precise.linear(x, self.w)
        else:
            if x.dtype != ttnn.bfloat16:
                x = ttnn.typecast(x, ttnn.bfloat16)
            y = ttnn.linear(x, self.w, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        if self.tp > 1:
            y = _ccl.all_gather(y, self.device, dim=-1)
            if y.shape[-1] != self.out_features:
                start = [0] * len(y.shape)
                end = list(y.shape)
                end[-1] = self.out_features
                y = ttnn.slice(y, start, end)
        if self.b is not None:
            y = ttnn.add(y, self.b)
        return y


def build(device, torch_module=None):
    return TtQwenDecoderHead(device, torch_module)


def decoder_head(device, torch_module=None):
    return TtQwenDecoderHead(device, torch_module)
