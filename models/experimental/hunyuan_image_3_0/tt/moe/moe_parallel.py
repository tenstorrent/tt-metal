# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Expert-parallel MoE — shards the 64 experts across a mesh axis so each device
# holds 64/ndev experts RESIDENT in bf8 (no per-forward host streaming). This is
# the dominant residency lever for the 80B model (experts are ~97% of per-layer
# weight memory). See MEMORY_FIT_PLAN.md.
#
# Math (mirrors the dense single-device HunyuanTtMoE):
#     combined = sum_e expert_e(x) * combine_w[:, e]
# Each device computes the partial sum over ITS experts, then we all-reduce the
# partial across the mesh axis to get the full combined output on every device.
#
# Per-device expert identity in SPMD: an arange(E) sharded along the mesh axis
# gives each device the GLOBAL ids of its local experts, so the combine-weight
# selection (topk_idx == global_id) picks the right column per device.

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from .gate import HunyuanTtTopKGate
from .mlp import HunyuanTtMLP


class HunyuanTtMoEParallel(LightweightModule):
    def __init__(
        self,
        mesh_device,
        ccl_manager,
        state_dict,
        prefix,
        *,
        num_experts: int,
        hidden_size: int,
        moe_topk: int,
        num_shared_expert: int = 1,
        norm_topk_prob: bool = True,
        use_mixed_mlp_moe: bool = True,
        mesh_axis: int = 0,
        weight_dtype=ttnn.bfloat8_b,
        gate_dtype=ttnn.float32,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.ccl = ccl_manager
        self.mesh_axis = mesh_axis
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe

        ndev = mesh_device.shape[mesh_axis]
        assert num_experts % ndev == 0, f"{num_experts} experts not divisible by mesh axis {mesh_axis} size {ndev}"
        self.ndev = ndev
        self.experts_per_dev = num_experts // ndev

        # --- stacked expert weights, sharded along the expert dim ------------
        # gate_and_up: [E, H, 2I]  down: [E, I, H]  (transposed for ttnn.linear)
        wgu = torch.stack(
            [
                state_dict[f"{prefix}.experts.{e}.gate_and_up_proj.weight"].transpose(0, 1).contiguous()
                for e in range(num_experts)
            ]
        )
        wdn = torch.stack(
            [
                state_dict[f"{prefix}.experts.{e}.down_proj.weight"].transpose(0, 1).contiguous()
                for e in range(num_experts)
            ]
        )
        self.inter2 = wgu.shape[-1]  # 2I
        self.w_gate_up = ttnn.from_torch(
            wgu,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )  # per-device [epd, H, 2I]
        self.w_down = ttnn.from_torch(
            wdn,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )  # per-device [epd, I, H]

        # Per-device GLOBAL expert ids: arange(E) sharded along the mesh axis.
        # bf16 so it compares against the (bf16-cast) topk indices; ids <= 63 exact.
        ids = torch.arange(num_experts, dtype=torch.float32).reshape(num_experts, 1)
        self.local_ids = ttnn.from_torch(
            ids,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )  # per-device [epd, 1]

        # Gate + shared MLP run replicated (input is replicated). Gate signature is
        # positional: (device, num_experts, moe_topk, state_dict, weight_key, ...).
        self.gate = HunyuanTtTopKGate(
            mesh_device,
            hidden_size,
            num_experts,
            moe_topk,
            state_dict,
            f"{prefix}.gate.wg",
            norm_topk_prob=norm_topk_prob,
            weight_dtype=gate_dtype,
        )
        self.shared_mlp = None
        if use_mixed_mlp_moe:
            self.shared_mlp = HunyuanTtMLP(
                mesh_device, hidden_size, state_dict, f"{prefix}.shared_mlp", weight_dtype=weight_dtype
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _slice_expert(self, w, el):
        """Slice local expert el from a [epd, A, B] sharded weight -> [A, B]."""
        epd, A, Bd = w.shape
        s = ttnn.slice(w, [el, 0, 0], [el + 1, A, Bd])
        return ttnn.reshape(s, (A, Bd))

    def _expert(self, x, el):
        """Run local expert `el` (its weight slice differs per device)."""
        wgu = self._slice_expert(self.w_gate_up, el)  # [H, 2I] (different global expert per device)
        wdn = self._slice_expert(self.w_down, el)  # [I, H]
        gu = ttnn.linear(
            x, wgu, compute_kernel_config=self.compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x1, x2 = ttnn.chunk(gu, 2, dim=-1)
        ttnn.deallocate(gu)
        act = ttnn.silu(x2)
        h = ttnn.multiply(x1, act)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(act)
        out = ttnn.linear(
            h, wdn, compute_kernel_config=self.compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(h)
        ttnn.deallocate(wgu)
        ttnn.deallocate(wdn)
        return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        topk_w, topk_idx_raw = self.gate(x)  # [B,S,k] each, replicated
        topk_idx = ttnn.typecast(topk_idx_raw, ttnn.bfloat16)  # ids <= 63 exact in bf16
        ttnn.deallocate(topk_idx_raw)

        partial = None
        for el in range(self.experts_per_dev):
            gid = self.local_ids[el]  # [1] — global id of this device's el-th expert
            sel = ttnn.eq(topk_idx, gid)  # [B,S,k] (broadcast scalar differs per device)
            contrib = ttnn.multiply(sel, topk_w)
            ttnn.deallocate(sel)
            w_e = ttnn.sum(contrib, dim=-1, keepdim=True)  # [B,S,1]
            ttnn.deallocate(contrib)

            oe = self._expert(x, el)
            weighted = ttnn.multiply(oe, w_e)  # [B,S,H]
            ttnn.deallocate(oe)
            ttnn.deallocate(w_e)
            if partial is None:
                partial = weighted
            else:
                tmp = ttnn.add(partial, weighted)
                ttnn.deallocate(partial)
                ttnn.deallocate(weighted)
                partial = tmp

        ttnn.deallocate(topk_w)
        ttnn.deallocate(topk_idx)

        # All-reduce the partial sums across the expert-shard axis: all-gather the
        # ndev partials then sum them (gives the full combined output everywhere).
        gathered = self.ccl.all_gather(
            partial, dim=0, mesh_axis=self.mesh_axis, use_hyperparams=False
        )  # [ndev*B, S, H]
        ttnn.deallocate(partial)
        B = gathered.shape[0] // self.ndev
        S, H = gathered.shape[1], gathered.shape[2]
        gathered = ttnn.reshape(gathered, (self.ndev, B, S, H))
        combined = ttnn.sum(gathered, dim=0)  # [B,S,H]
        ttnn.deallocate(gathered)

        if self.shared_mlp is not None:
            shared = self.shared_mlp(x)
            out = ttnn.add(shared, combined)
            ttnn.deallocate(shared)
            ttnn.deallocate(combined)
            return out
        return combined
