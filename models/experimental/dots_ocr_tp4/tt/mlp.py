# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 SwiGLU MLP for dots.ocr prefill (replicated-hidden Megatron design).

    gate = x @ gate_proj.T        # column-parallel, output sharded on I
    up   = x @ up_proj.T          # column-parallel, output sharded on I
    act  = silu(gate) * up        # elementwise, per-chip on its I/ndev slice
    out  = act @ down_proj.T      # row-parallel, partial sums per chip
    out  = all_reduce(out)        # -> full replicated hidden

Input  x   : replicated [B, S, H]
Output out : replicated [B, S, H]
"""

import ttnn

from models.experimental.dots_ocr_tp4.tt.common import all_reduce, mesh_num_devices, shard_to_mesh


class DotsOCRMLPTP4:
    def __init__(self, mesh_device, config, weight_dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.config = config
        self.weight_dtype = weight_dtype
        self.num_devices = mesh_num_devices(mesh_device)
        self.gate_w = None
        self.up_w = None
        self.down_w = None
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(cls, mesh_device, config, torch_mlp, weight_dtype=ttnn.bfloat16):
        m = cls(mesh_device, config, weight_dtype=weight_dtype)
        m.load_weights(torch_mlp)
        return m

    def load_weights(self, torch_mlp):
        nd = max(1, self.num_devices)
        I = self.config.intermediate_size
        assert I % nd == 0, f"intermediate_size {I} not divisible by {nd}"

        # ttnn.linear computes x @ W with W laid out [K, N]; torch weights are
        # [out, in], so transpose to [in, out] then shard.
        gate_w = torch_mlp.gate_proj.weight.data.t().contiguous()  # [H, I]
        up_w = torch_mlp.up_proj.weight.data.t().contiguous()  # [H, I]
        down_w = torch_mlp.down_proj.weight.data.t().contiguous()  # [I, H]

        # Column-parallel gate/up: shard output dim (I) across chips.
        self.gate_w = shard_to_mesh(gate_w, self.mesh_device, dim=-1, dtype=self.weight_dtype)
        self.up_w = shard_to_mesh(up_w, self.mesh_device, dim=-1, dtype=self.weight_dtype)
        # Row-parallel down: shard contraction dim (I = K, dim 0) across chips.
        self.down_w = shard_to_mesh(down_w, self.mesh_device, dim=0, dtype=self.weight_dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        gate = ttnn.linear(
            x,
            self.gate_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        up = ttnn.linear(
            x,
            self.up_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        act = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        out = ttnn.linear(
            act,
            self.down_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(act)

        out = all_reduce(out, self.mesh_device)
        return out
