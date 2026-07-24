# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Multi-modal projector: vision-tower output -> LLM-hidden.

Mirrors HF `KimiVLMultiModalProjector.forward`:
    image_features = torch.cat(image_features, dim=0)       # (L_new, kh*kw, D_vis)
    hidden_states = self.pre_norm(image_features)           # LN over last dim D_vis=1152
    hidden_states = hidden_states.view(-1, merge_dim)       # (L_new, kh*kw*D_vis)
    hidden_states = self.linear_1(hidden_states)            # 4608 -> 4608
    hidden_states = self.act(hidden_states)                 # plain GELU (not tanh approx)
    hidden_states = self.linear_2(hidden_states)            # 4608 -> text_hidden
    return hidden_states

The pre_norm operates on the 3D (..., merge_kh*merge_kw, D_vis) form
where the last dim is the per-sub-patch vision hidden. Note that the
activation here is the plain GELU (`transformers.activations.GELUActivation`),
NOT the tanh-approximated variant used in the encoder MLP — so we use
`ttnn.gelu` without `fast_and_approximate_mode=True`.

Reference: `KimiVLMultiModalProjector` in modeling_kimi_k25.py.
"""
from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3.tt.moonvit.layernorm import MoonVisionLayerNorm


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


def _as_linear_weight(mesh_device, w: torch.Tensor, dtype) -> ttnn.Tensor:
    """Move a torch nn.Linear weight to device with ttnn convention [in, out]."""
    w_pt = w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _is_mesh_device(mesh_device) else None
    return ttnn.as_tensor(
        w_pt,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _as_linear_bias(mesh_device, b: torch.Tensor, dtype) -> ttnn.Tensor:
    """Move a torch nn.Linear bias to device as (1, 1, 1, out)."""
    b_pt = b.detach().to(torch.bfloat16).view(1, 1, 1, -1).contiguous()
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _is_mesh_device(mesh_device) else None
    return ttnn.as_tensor(
        b_pt,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


class MoonViTProjector(LightweightModule):
    """vision-tower output (merged 3D form) -> LLM hidden."""

    def __init__(
        self,
        mesh_device,
        vision_hidden: int,
        merge_dim: int,
        text_hidden: int,
        pre_norm: MoonVisionLayerNorm,
        linear_1_weight: torch.Tensor,
        linear_1_bias: torch.Tensor,
        linear_2_weight: torch.Tensor,
        linear_2_bias: torch.Tensor,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.dtype = dtype
        self.vision_hidden = int(vision_hidden)
        self.merge_dim = int(merge_dim)
        self.text_hidden = int(text_hidden)

        self.pre_norm = pre_norm
        self.linear_1_weight = _as_linear_weight(mesh_device, linear_1_weight, dtype)
        self.linear_1_bias = _as_linear_bias(mesh_device, linear_1_bias, dtype)
        self.linear_2_weight = _as_linear_weight(mesh_device, linear_2_weight, dtype)
        self.linear_2_bias = _as_linear_bias(mesh_device, linear_2_bias, dtype)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.Module,
        dtype=ttnn.bfloat16,
    ) -> "MoonViTProjector":
        """Construct from HF KimiVLMultiModalProjector."""
        for attr in ("pre_norm", "linear_1", "linear_2"):
            assert hasattr(ref, attr), f"expected projector with .{attr}, got {type(ref).__name__}"
        # Vision hidden is what pre_norm normalizes over.
        vision_hidden = ref.pre_norm.normalized_shape[-1]
        merge_dim = ref.linear_1.in_features
        text_hidden = ref.linear_2.out_features
        # Both linears must be bias-true per HF __init__.
        assert ref.linear_1.bias is not None and ref.linear_2.bias is not None

        pre_norm = MoonVisionLayerNorm.from_torch(mesh_device, ref.pre_norm, dtype=dtype)
        return cls(
            mesh_device=mesh_device,
            vision_hidden=vision_hidden,
            merge_dim=merge_dim,
            text_hidden=text_hidden,
            pre_norm=pre_norm,
            linear_1_weight=ref.linear_1.weight.data,
            linear_1_bias=ref.linear_1.bias.data,
            linear_2_weight=ref.linear_2.weight.data,
            linear_2_bias=ref.linear_2.bias.data,
            dtype=dtype,
        )

    def forward(
        self,
        x_merged: ttnn.Tensor,  # (1, 1, L_new * merge_groups, vision_hidden)
        memory_config: Optional["ttnn.MemoryConfig"] = None,
    ) -> ttnn.Tensor:
        """Project merged vision tokens into LLM hidden space.

        Input is the 3D-merged form flattened on dims so that LayerNorm
        normalizes over `vision_hidden`. Specifically, the caller should
        pass (1, 1, L_new * merge_groups, vision_hidden) — each row in dim
        -2 is one sub-patch's vision-hidden vector, which `pre_norm` will
        normalize independently. After LN, we reshape to
        (1, 1, L_new, merge_dim = merge_groups * vision_hidden) for the
        Linear chain.
        """
        # LayerNorm over the last (vision_hidden) dim, applied per-sub-patch.
        normed = self.pre_norm(x_merged)
        # Reshape (1, 1, L_new * merge_groups, vision_hidden) -> (1, 1, L_new, merge_dim).
        # merge_groups = merge_dim // vision_hidden.
        l_total = normed.shape[-2]
        assert l_total % (self.merge_dim // self.vision_hidden) == 0, (
            f"L_total {l_total} not divisible by merge_groups " f"{self.merge_dim // self.vision_hidden}"
        )
        l_new = l_total // (self.merge_dim // self.vision_hidden)
        normed = ttnn.reshape(normed, [1, 1, l_new, self.merge_dim])

        h = ttnn.linear(
            normed,
            self.linear_1_weight,
            bias=self.linear_1_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(normed)

        # Plain GELU (matches transformers GELUActivation; NOT the tanh approximation).
        h = ttnn.gelu(h)

        out = ttnn.linear(
            h,
            self.linear_2_weight,
            bias=self.linear_2_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        return out
