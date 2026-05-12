# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B SwiGLU MLP for BH GLX 8×4 mesh.

Computes:  down_proj(silu(gate_proj(x)) * up_proj(x))

Weight shapes (from HF safetensors):
  gate_proj.weight:  [intermediate=17408, H=5120]  → ttnn linear: [H, intermediate]
  up_proj.weight:    [intermediate=17408, H=5120]  → ttnn linear: [H, intermediate]
  down_proj.weight:  [H=5120, intermediate=17408]  → ttnn linear: [intermediate, H]

No TP sharding here — weights are replicated across the full mesh. The forward
uses ttnn.all_gather to sum partial outputs across mesh rows when needed, but
for the standalone decoder test path we use simple replicated matmuls.
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwen36MLP(LightweightModule):
    """SwiGLU MLP for Qwen3.6-27B on BH GLX 8×4 mesh.

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        Full 8×4 mesh.
    state_dict : dict
        Weight dict with keys:
          mlp.gate_proj.weight  [intermediate, H]
          mlp.up_proj.weight    [intermediate, H]
          mlp.down_proj.weight  [H, intermediate]
    dtype : ttnn.DataType
        Weight dtype (default bfloat16).
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.dtype = dtype

        # Compute kernel
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._build_weights(state_dict)

    def _to_device(self, t: torch.Tensor) -> ttnn.Tensor:
        """Upload weight to mesh, replicated across all devices."""
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _build_weights(self, sd: dict):
        """Prepare and upload MLP weights.

        HF layout:
          gate_proj.weight: [intermediate, H] → ttnn linear needs [H, intermediate]
          up_proj.weight:   [intermediate, H] → ttnn linear needs [H, intermediate]
          down_proj.weight: [H, intermediate] → ttnn linear needs [intermediate, H]
        """
        gate_w = sd["mlp.gate_proj.weight"]  # [17408, 5120]
        up_w = sd["mlp.up_proj.weight"]  # [17408, 5120]
        down_w = sd["mlp.down_proj.weight"]  # [5120, 17408]

        # Transpose to (input_dim, output_dim) for ttnn.linear(x @ W)
        self.w_gate = self._to_device(gate_w.T.contiguous())  # [5120, 17408]
        self.w_up = self._to_device(up_w.T.contiguous())  # [5120, 17408]
        self.w_down = self._to_device(down_w.T.contiguous())  # [17408, 5120]

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Compute SwiGLU MLP forward.

        Args:
            x: [B, T, H] bfloat16 replicated across mesh.

        Returns:
            [B, T, H] bfloat16 replicated across mesh.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        ck = self.compute_kernel

        # gate_proj(x) then silu
        gate_out = ttnn.linear(x, self.w_gate, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        gate_act = ttnn.silu(gate_out, memory_config=mem)
        gate_out.deallocate(True)

        # up_proj(x)
        up_out = ttnn.linear(x, self.w_up, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)

        # Element-wise multiply
        ff = ttnn.multiply(gate_act, up_out, memory_config=mem)
        gate_act.deallocate(True)
        up_out.deallocate(True)

        # down_proj(ff)
        out = ttnn.linear(ff, self.w_down, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        ff.deallocate(True)

        return out
