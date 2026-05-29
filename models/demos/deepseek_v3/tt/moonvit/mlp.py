# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MoonViT MLP.

Linear(1152 -> 4304) -> GELU(tanh) -> Linear(4304 -> 1152).

The activation is `PytorchGELUTanh()` in the reference, which is the
tanh-approximation flavor of GELU. We match it with
`ttnn.gelu(..., fast_and_approximate_mode=True)`.

The intermediate dim 4304 is NOT tile-aligned (4304 % 32 = 16); ttnn's
`as_tensor` auto-pads to 4320 internally and `ttnn.linear` honors the
padded shape transparently. The extra 16 columns of fc0.weight are
auto-padded with zeros, so the corresponding intermediate activations
are GELU(0)=0 (plus bias which is also zero-padded), and they then
multiply zero-rows of fc1.weight, contributing nothing to the output.

Reference: `MLP2` in modeling_kimi_k25.py — attributes:
    self.fc0 = nn.Linear(hidden_dim, mlp_dim, bias=True)
    self.activation = PytorchGELUTanh()
    self.fc1 = nn.Linear(mlp_dim, hidden_dim, bias=True)
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


def _as_linear_weight(mesh_device, torch_w: torch.Tensor, dtype) -> ttnn.Tensor:
    """Move a torch Linear weight to device in ttnn.linear convention.

    PyTorch stores nn.Linear.weight as [out_features, in_features].
    ttnn.linear expects [in_features, out_features] (operand-b semantics
    for matmul), so we transpose.
    """
    w = torch_w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()
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
    """Bias goes in as [1, 1, 1, out_features], TILE_LAYOUT, replicated."""
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


class MoonVisionMLP(LightweightModule):
    """Two-layer MLP with GELU-tanh activation, matching MLP2 from HF."""

    def __init__(
        self,
        mesh_device,
        hidden_size: int,
        intermediate_size: int,
        fc0_weight: torch.Tensor,
        fc0_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)

        # Shape sanity.
        assert tuple(fc0_weight.shape) == (intermediate_size, hidden_size), (
            f"fc0.weight shape {tuple(fc0_weight.shape)} != "
            f"(intermediate={intermediate_size}, hidden={hidden_size})"
        )
        assert tuple(fc1_weight.shape) == (hidden_size, intermediate_size), (
            f"fc1.weight shape {tuple(fc1_weight.shape)} != "
            f"(hidden={hidden_size}, intermediate={intermediate_size})"
        )
        assert fc0_bias.shape[0] == intermediate_size, f"fc0.bias shape {fc0_bias.shape}"
        assert fc1_bias.shape[0] == hidden_size, f"fc1.bias shape {fc1_bias.shape}"

        self.fc0_weight = _as_linear_weight(mesh_device, fc0_weight, dtype)
        self.fc0_bias = _as_linear_bias(mesh_device, fc0_bias, dtype)
        self.fc1_weight = _as_linear_weight(mesh_device, fc1_weight, dtype)
        self.fc1_bias = _as_linear_bias(mesh_device, fc1_bias, dtype)

        # HiFi4 for the matmul accumulation — same as our LayerNorm,
        # consistent with Qwen-VL/Llama-Vision precedents.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # FP32 dst accum for MLP matmuls — small accuracy boost.
            packer_l1_acc=False,
        )

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.Module,
        dtype=ttnn.bfloat16,
    ) -> "MoonVisionMLP":
        """Construct from a torch MLP2 reference module.

        Pulls out fc0/fc1 weights+biases and the dimensions implied by their shapes.
        """
        assert hasattr(ref, "fc0") and hasattr(ref, "fc1"), (
            f"expected an MLP2-like module with fc0+fc1, got {type(ref).__name__} "
            f"with children {[n for n, _ in ref.named_children()]}"
        )
        intermediate_size, hidden_size = ref.fc0.weight.shape
        return cls(
            mesh_device=mesh_device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc0_weight=ref.fc0.weight.data,
            fc0_bias=ref.fc0.bias.data,
            fc1_weight=ref.fc1.weight.data,
            fc1_bias=ref.fc1.bias.data,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor, memory_config: Optional["ttnn.MemoryConfig"] = None) -> ttnn.Tensor:
        # fc0: linear up
        hidden = ttnn.linear(
            x,
            self.fc0_weight,
            bias=self.fc0_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # GELU with tanh approximation
        hidden = ttnn.gelu(hidden, fast_and_approximate_mode=True)
        # fc1: linear down
        out = ttnn.linear(
            hidden,
            self.fc1_weight,
            bias=self.fc1_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        return out
