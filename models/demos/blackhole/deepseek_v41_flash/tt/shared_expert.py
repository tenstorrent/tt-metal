# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shared expert of a DeepSeek-V4.1-Flash MoE layer, as ordinary matmuls instead of a ``moe_compute`` expert.

``moe_compute`` only takes bfloat4_b weights. The shared expert runs at weight 1.0 on every token, so its bfp4
re-quantisation error (11% relative on layer 2, more than all six routed experts together) dominates the MoE error.
Here it is replicated on every device in bfloat8_b (35 MB per layer per device, error negligible), with gate and up fused into one matmul (0.265 -> 0.155 ms/layer) and uses the
checkpoint's clamps: up in [-10, 10], gate <= 10, ``silu(gate) * up @ w2``.
"""

import torch

import ttnn


class DSV41SharedExpert:
    def __init__(self, mesh_device, w0, w1, w2, limit=10.0, dtype=ttnn.bfloat8_b):
        """w0 (gate), w1 (up): [1, 1, dim, inter]; w2 (down): [1, 1, inter, dim] as host tensors ([in, out])."""
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        up = lambda t: ttnn.from_torch(
            t.contiguous(),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.inter = w0.shape[-1]
        self.w01, self.w2 = up(torch.cat([w0, w1], dim=-1)), up(w2)  # gate|up fused into one matmul
        self.limit = float(limit)
        self.grid = ttnn.CoreGrid(y=8, x=8)
        self.silu = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, h):
        """h [1, 1, T, dim] bf16 -> [1, 1, T, dim] fp32."""
        gu = ttnn.linear(h, self.w01, dtype=ttnn.float32, compute_kernel_config=self.ckc, core_grid=self.grid)
        gate = ttnn.minimum(gu[:, :, :, : self.inter], self.limit)
        up = ttnn.clamp(gu[:, :, :, self.inter :], min=-self.limit, max=self.limit)
        act = ttnn.multiply(gate, up, input_tensor_a_activations=self.silu)  # silu(gate) * up
        return ttnn.linear(act, self.w2, dtype=ttnn.float32, compute_kernel_config=self.ckc, core_grid=self.grid)
