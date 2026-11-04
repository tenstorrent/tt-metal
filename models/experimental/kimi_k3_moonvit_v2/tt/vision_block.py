# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Encoder block: `MoonViTEncoderLayer` in the reference.

x + attention(norm0(x)), then + mlp(norm1(x)). The norms are RMSNorm without bias, eps
`config.norm_eps`. The reference keeps `wqkv` and `wo` on the block itself; here they live in
`attention`, and `_prepare_torch_state` moves the keys.

The residual stream (`x`, `h`, the output) is fp32; everything else stays bf16. The norms read it
and hand bf16 to the matmuls, and `wo` and `fc1` write fp32 for the adds. A bf16 residual would round
away small updates to large values (it reaches |x| ~ 500), which costs end-to-end accuracy.
"""

from __future__ import annotations

import torch

import ttnn

from .model_config import MoonViTConfig, VisionParallel
from .module import Module, rename_substate
from .rms_norm import RMSNorm
from .rope import RopeTensors
from .vision_attention import TT_Vision_Attention
from .vision_mlp import TT_Vision_MLP


class TT_Vision_Block(Module):
    def __init__(
        self,
        config: MoonViTConfig,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel: VisionParallel,
        compute_kernel_config,
        ccl_slot: int = 0,
    ) -> None:
        """`ccl_slot`: see `TT_Vision_Attention`."""
        super().__init__()
        norm_kwargs = dict(
            eps=config.norm_eps,
            mesh_device=mesh_device,
            compute_kernel_config=compute_kernel_config,
            output_dtype=ttnn.bfloat16,
        )
        self.norm0 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.norm1 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.attention = TT_Vision_Attention(
            config,
            mesh_device=mesh_device,
            parallel=parallel,
            compute_kernel_config=compute_kernel_config,
            ccl_slot=ccl_slot,
        )
        self.mlp = TT_Vision_MLP(config, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "wqkv", "attention.wqkv")
        rename_substate(state, "wo", "attention.wo")

    def forward(self, x: ttnn.Tensor, rope: RopeTensors, logical_n: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, rows, hidden_size] fp32 -> the same shape, fp32; rows are this device's shard of
        the padded, window-ordered sequence. `x` is left allocated: the caller owns it."""
        normed = self.norm0(x)
        attn = self.attention(normed, rope, logical_n)
        ttnn.deallocate(normed)
        h = ttnn.add(x, attn, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        normed = self.norm1(h)
        mlp = self.mlp(normed)
        ttnn.deallocate(normed)
        out = ttnn.add(h, mlp, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        ttnn.deallocate(mlp)
        return out
