# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Vision FeedForward for Mistral-Small / Pixtral-class checkpoints.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


class MistralTTVisionMLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        dtype,
        state_dict_prefix=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.state_dict = state_dict
        self.dim = args.dim

        def get_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)

        def as_tensor(torch_2d, dtype):
            return ttnn.as_tensor(
                torch_2d,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w1_t = get_weight("w1")
        w3_t = get_weight("w3")
        if w1_t.shape != w3_t.shape:
            raise ValueError(f"w1 and w3 must match for fused SwiGLU matmul; got {w1_t.shape} vs {w3_t.shape}")
        # Single matmul over x for gate+up cuts duplicate activation reads and one matmul enqueue vs separate w1/w3 linears.
        self.w1_w3 = as_tensor(torch.cat([w1_t, w3_t], dim=-1), dtype)
        self.w2 = as_tensor(get_weight("w2"), dtype)

        self.compute_kernel_config = args.compute_kernel_config_hifi2

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        fused = ttnn.linear(
            x,
            self.w1_w3,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        b0, b1, seq, tw = fused.shape
        half = tw // 2
        w1_out = ttnn.slice(fused, (0, 0, 0, 0), (b0, b1, seq, half))
        w3_out = ttnn.slice(fused, (0, 0, 0, half), (b0, b1, seq, tw))
        w1_out = ttnn.silu(w1_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        w2_in = ttnn.mul(w1_out, w3_out, dtype=ttnn.bfloat16)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        ttnn.deallocate(fused)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        ttnn.deallocate(w2_in)
        return w2_out


__all__ = ["MistralTTVisionMLP"]
