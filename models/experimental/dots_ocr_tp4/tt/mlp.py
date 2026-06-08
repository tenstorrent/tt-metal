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

import torch
import ttnn

from models.experimental.dots_ocr_tp4.tt.common import (
    all_reduce,
    matmul_m_dim,
    mesh_num_devices,
    prefill_matmul_2d_config,
    shard_to_mesh,
)
from models.experimental.tt_symbiote.core.module import TTNNModule


class DotsOCRMLPTP4(TTNNModule):
    def __init__(self, mesh_device, config, weight_dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.weight_dtype = weight_dtype
        self.num_devices = mesh_num_devices(mesh_device)
        self.gate_up_w = None  # fused [H, 2*I/nd] per chip: [gate_shard | up_shard]
        self.down_w = None
        # Precision recipe matched to the production dots.ocr prefill profile:
        #   gate/up : BF16 x BFP4 -> BFP8 @ HiFi2
        #   silu*mul: BFP8 x BFP8 -> BFP8
        #   down    : BFP8 x BFP4 -> BFP8 @ LoFi
        self.gate_up_weight_dtype = ttnn.bfloat4_b
        self.down_weight_dtype = ttnn.bfloat4_b
        self.gate_up_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.down_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        config,
        torch_mlp,
        weight_dtype=ttnn.bfloat16,
        gate_up_weight_dtype=None,
        down_weight_dtype=None,
    ):
        m = cls(mesh_device, config, weight_dtype=weight_dtype)
        m.set_weight_dtype(gate_up_dtype=gate_up_weight_dtype, down_dtype=down_weight_dtype)
        m.load_weights(torch_mlp)
        m.to_device(mesh_device)
        m._preprocessed_weight = True
        m._weights_on_device = True
        return m

    def set_weight_dtype(self, gate_up_dtype=None, down_dtype=None):
        if gate_up_dtype is not None:
            self.gate_up_weight_dtype = gate_up_dtype
        if down_dtype is not None:
            self.down_weight_dtype = down_dtype
        return self

    def load_weights(self, torch_mlp):
        nd = max(1, self.num_devices)
        I = self.config.intermediate_size
        assert I % nd == 0, f"intermediate_size {I} not divisible by {nd}"
        i_shard = I // nd

        # ttnn.linear computes x @ W with W laid out [K, N]; torch weights are
        # [out, in], so transpose to [in, out].
        gate_w = torch_mlp.gate_proj.weight.data.t().contiguous()  # [H, I]
        up_w = torch_mlp.up_proj.weight.data.t().contiguous()  # [H, I]
        down_w = torch_mlp.down_proj.weight.data.t().contiguous()  # [I, H]

        # Fuse gate+up into ONE column-parallel matmul. Interleave per chip so a
        # single shard(dim=-1) lands [gate_shard_c | up_shard_c] on chip c, and a
        # chunk(2) on the output recovers gate/up. Columns ordered:
        #   [chip0_gate | chip0_up | chip1_gate | chip1_up | ...].
        blocks = []
        for c in range(nd):
            blocks.append(gate_w[:, c * i_shard : (c + 1) * i_shard])
            blocks.append(up_w[:, c * i_shard : (c + 1) * i_shard])
        gate_up = torch.cat(blocks, dim=-1).contiguous()  # [H, 2I]
        self.gate_up_w = shard_to_mesh(gate_up, self.mesh_device, dim=-1, dtype=self.gate_up_weight_dtype)
        # Row-parallel down: shard contraction dim (I = K, dim 0) across chips.
        self.down_w = shard_to_mesh(down_w, self.mesh_device, dim=0, dtype=self.down_weight_dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Decode (seq==1) keeps the tiny M=1 activations L1-resident; prefill stays
        # DRAM-interleaved (its large CBs clash with L1-staged in0 -> L1 OOM).
        is_decode = int(x.shape[-2]) == 1
        act_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=act_mc)

        m = matmul_m_dim(x)
        K = int(x.shape[-1])
        gate_up_n = int(self.gate_up_w.shape[-1])  # 2 * I/nd
        gate_up_pc = prefill_matmul_2d_config(self.mesh_device, m, K, gate_up_n, fp32_dest=True)

        # Single fused gate+up matmul, then split: per chip the output is
        # [gate_shard | up_shard], so chunk(2, dim=-1) recovers them. in0 stays
        # DRAM-interleaved -- L1-staging it clashed with this matmul's large CBs
        # on the 28-layer run (L1 OOM); the tuned in0_block_w is the real win.
        gate_up = ttnn.linear(
            x,
            self.gate_up_w,
            dtype=ttnn.bfloat8_b,
            memory_config=act_mc,
            compute_kernel_config=self.gate_up_compute,
            program_config=gate_up_pc,
        )
        gate, up = ttnn.chunk(gate_up, 2, dim=-1)
        ttnn.deallocate(gate_up)
        act = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=act_mc,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        I_shard = int(act.shape[-1])
        H = int(self.down_w.shape[-1])
        down_pc = prefill_matmul_2d_config(self.mesh_device, m, I_shard, H, fp32_dest=False)
        out = ttnn.linear(
            act,
            self.down_w,
            dtype=ttnn.bfloat8_b,
            memory_config=act_mc,
            compute_kernel_config=self.down_compute,
            program_config=down_pc,
        )
        ttnn.deallocate(act)

        out = all_reduce(out, self.mesh_device, output_memory_config=act_mc)
        return out
