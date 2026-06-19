# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL multimodal projector (spatial merge folds into it).

Per `MiniMaxM3VLMultiModalProjector.forward`:
    x = linear_2(gelu(linear_1(x)))                    # per-patch: 1280 -> 6144 -> 6144
    x = x.reshape(L // merge**2, -1)                   # group 4 patches -> 24576
    x = merge_linear_2(merge_act(merge_linear_1(x)))   # 24576 -> 6144 -> 6144

There is no pre-norm. The 4 grouped patches are consecutive in the
sequence (the image processor pre-orders patches so each group of
spatial_merge**2 is a 2x2 spatial block — same ordering the 3D RoPE coords
use). All four linears carry bias; the activation is the accurate (erf)
GELU (`ACT2FN["gelu"]`).
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


def _as_linear_weight(mesh_device, torch_w: torch.Tensor, dtype) -> ttnn.Tensor:
    w = torch_w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()  # [in, out]
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _is_mesh_device(mesh_device) else None
    return ttnn.as_tensor(
        w,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _as_linear_bias(mesh_device, torch_b: torch.Tensor, dtype) -> ttnn.Tensor:
    b = torch_b.detach().to(torch.bfloat16).view(1, 1, 1, -1).contiguous()
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _is_mesh_device(mesh_device) else None
    return ttnn.as_tensor(
        b,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


class M3VLProjector(LightweightModule):
    """Per-patch MLP -> spatial-merge fold (4 patches) -> merge MLP."""

    def __init__(
        self,
        mesh_device,
        ref_projector: torch.nn.Module,  # _Projector with linear_1/linear_2/merge_linear_1/merge_linear_2
        spatial_merge_size: int,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.merge_factor = int(spatial_merge_size) ** 2

        for n in ("linear_1", "linear_2", "merge_linear_1", "merge_linear_2"):
            assert hasattr(ref_projector, n), f"expected {n} on {type(ref_projector).__name__}"

        self.linear_1_w = _as_linear_weight(mesh_device, ref_projector.linear_1.weight.data, dtype)
        self.linear_1_b = _as_linear_bias(mesh_device, ref_projector.linear_1.bias.data, dtype)
        self.linear_2_w = _as_linear_weight(mesh_device, ref_projector.linear_2.weight.data, dtype)
        self.linear_2_b = _as_linear_bias(mesh_device, ref_projector.linear_2.bias.data, dtype)
        self.merge_1_w = _as_linear_weight(mesh_device, ref_projector.merge_linear_1.weight.data, dtype)
        self.merge_1_b = _as_linear_bias(mesh_device, ref_projector.merge_linear_1.bias.data, dtype)
        self.merge_2_w = _as_linear_weight(mesh_device, ref_projector.merge_linear_2.weight.data, dtype)
        self.merge_2_b = _as_linear_bias(mesh_device, ref_projector.merge_linear_2.bias.data, dtype)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def from_torch(cls, mesh_device, ref_projector, spatial_merge_size, dtype=ttnn.bfloat16) -> "M3VLProjector":
        return cls(mesh_device, ref_projector, spatial_merge_size, dtype=dtype)

    def _linear(self, x, w, b, act=False):
        out = ttnn.linear(
            x, w, bias=b, compute_kernel_config=self.compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if act:
            out = ttnn.gelu(out, fast_and_approximate_mode=False)
        return out

    def forward(self, x: ttnn.Tensor, memory_config: Optional["ttnn.MemoryConfig"] = None) -> ttnn.Tensor:
        """x: (1, 1, L, hidden) -> (1, 1, L//merge_factor, text_hidden)."""
        # Per-patch MLP: linear_1 -> gelu -> linear_2.
        h = self._linear(x, self.linear_1_w, self.linear_1_b, act=True)
        h = self._linear(h, self.linear_2_w, self.linear_2_b, act=False)

        # Fold spatial_merge**2 consecutive patches into the channel dim.
        L = h.shape[-2]
        merged_dim = h.shape[-1] * self.merge_factor
        # Reshape crosses tile rows; go through row-major to keep it exact.
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = ttnn.reshape(h, (1, 1, L // self.merge_factor, merged_dim))
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)

        # Merge MLP: merge_linear_1 -> gelu -> merge_linear_2.
        h = self._linear(h, self.merge_1_w, self.merge_1_b, act=True)
        out = ttnn.linear(
            h,
            self.merge_2_w,
            bias=self.merge_2_b,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        return out
