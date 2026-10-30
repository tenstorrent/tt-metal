# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN MoE for GLM-4.7-Flash (TP-sharded, expert-parallel), fully on device.

MoE block: router -> sigmoid scores (+ e_score_correction_bias for the choice) -> top-k
-> normalised, scaled weights -> 64 routed SwiGLU experts + 1 shared expert (dense SwiGLU).

Expert-parallel: the 64 experts are sharded 64/TP per chip. Every chip runs ALL of its local
experts over ALL tokens as one batched matmul and weights each by its (mostly zero) routing weight,
so no token dispatch / host gather is needed; the partial sums are all_reduced over the TP axis.
At B=32 decode nearly every expert is hit anyway (64 * (1 - (60/64)^32) ~ 56), so this costs
little over a sparse dispatch.

Top-k is computed exactly in fp32 on device by k rounds of (row max -> knock out) and a final
`choice >= kth-max` mask, which selects the same experts as torch.topk (barring exact ties).

`router` (the topk_router stub) and `shared_expert` (the m_l_p stub) may be injected by the
pipeline; without them this block falls back to its own on-device router / shared expert so the
stub stays runnable on its own.
"""
from __future__ import annotations

import os

import torch

import ttnn


class TtMoE:
    def __init__(self, device, torch_module, router=None, shared_expert=None, weight_cache=None) -> None:
        self.device = device
        sd = torch_module.state_dict()

        self.n_experts = int(torch_module.n_routed_experts)
        self.top_k = int(torch_module.top_k)
        self.hidden_size = int(torch_module.config.hidden_size)
        self.moe_intermediate = int(torch_module.config.moe_intermediate_size)
        self.n_group = int(torch_module.n_group)
        self.topk_group = int(torch_module.topk_group)
        self.norm_topk_prob = bool(torch_module.norm_topk_prob)
        self.routed_scaling_factor = float(torch_module.routed_scaling_factor)
        if self.n_group != 1 and self.topk_group != self.n_group:
            raise NotImplementedError("group-limited routing (topk_group < n_group) is not ported")

        try:
            _is_mesh = isinstance(device, ttnn.MeshDevice)
        except AttributeError:
            _is_mesh = False
        _mesh_shape = list(device.shape) if _is_mesh else [1, 1]
        shard = bool(os.environ.get("TT_HW_PLANNER_SHARD_RUN")) and _is_mesh
        TP = _mesh_shape[-1] if shard else 1
        shard = shard and TP > 1 and (self.n_experts % TP == 0)
        self._shard = shard
        self._TP = TP if shard else 1
        self._tp_axis = len(_mesh_shape) - 1
        self._mesh_shape = _mesh_shape
        self._local_experts = self.n_experts // self._TP
        self._is_mesh_dev = _is_mesh

        def put(t, dtype, shard_dim=None, cache=None):
            kw = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if shard_dim is not None and shard:
                kw["mesh_mapper"] = ttnn.ShardTensor2dMesh(device, mesh_shape=_mesh_shape, dims=(None, shard_dim))
            elif _is_mesh:
                kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
            if cache is not None:
                return ttnn.as_tensor(t, cache_file_name=cache, **kw)
            return ttnn.from_torch(t, **kw)

        def cname(tag):
            if weight_cache is None:
                return None
            return f"{weight_cache}_{tag}_tp{self._TP}"

        # Routing constants (replicated, fp32 like the HF router)
        self._bias = put(sd["gate.e_score_correction_bias"].float().reshape(1, self.n_experts), ttnn.float32)
        # Selector: column block of I for this chip's experts -> w_local = w @ sel   ([N,64] -> [N,E_local])
        self._sel = put(torch.eye(self.n_experts), ttnn.float32, shard_dim=1)

        # Routed experts, expert-parallel. HF: gate_up_proj [E, 2I, H] (gate rows first), down_proj [E, H, I]
        self._w_gate_up = put(
            sd["experts.gate_up_proj"].transpose(1, 2).contiguous(), ttnn.bfloat4_b, 0, cname("gate_up_bf4")
        )
        self._w_down = put(sd["experts.down_proj"].transpose(1, 2).contiguous(), ttnn.bfloat4_b, 0, cname("down_bf4"))

        self.router = router
        self.shared_expert = shared_expert
        if router is None:
            self._gate_w = put(sd["gate.weight"].float().t().contiguous(), ttnn.float32)
        if shared_expert is None:
            self._shared_gate = put(sd["shared_experts.gate_proj.weight"].t().contiguous(), ttnn.bfloat8_b)
            self._shared_up = put(sd["shared_experts.up_proj.weight"].t().contiguous(), ttnn.bfloat8_b)
            self._shared_down = put(sd["shared_experts.down_proj.weight"].t().contiguous(), ttnn.bfloat8_b)

        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.ckc_expert = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        self.ckc_shared = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )

    @classmethod
    def build(cls, device, torch_module, **kwargs):
        return cls(device, torch_module, **kwargs)

    def attach(self, router=None, shared_expert=None):
        if router is not None:
            self.router = router
        if shared_expert is not None:
            self.shared_expert = shared_expert
        return self

    # ---- pieces ----

    def _router_logits(self, x, N):
        if self.router is not None:
            return self.router(x)  # [N, n_experts] fp32
        x32 = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
        x2 = ttnn.reshape(x32, [N, self.hidden_size])
        return ttnn.matmul(x2, self._gate_w, compute_kernel_config=self.ckc)

    def _routing_weights(self, logits):
        """[N, E] fp32 logits -> [N, E] fp32 dense weights (zero outside the top-k)."""
        scores = ttnn.sigmoid(logits)
        choice = ttnn.add(scores, self._bias)
        work = choice
        kth = None
        for k in range(self.top_k):
            kth = ttnn.max(work, dim=-1, keepdim=True)
            if k < self.top_k - 1:
                hit = ttnn.ge(work, kth)
                work = ttnn.subtract(work, ttnn.multiply(hit, 1.0e4))
        mask = ttnn.ge(choice, kth)
        w = ttnn.multiply(scores, mask)
        if self.norm_topk_prob:
            den = ttnn.add(ttnn.sum(w, dim=-1, keepdim=True), 1e-20)
            w = ttnn.multiply(w, ttnn.reciprocal(den))
        return ttnn.multiply(w, self.routed_scaling_factor)

    def _shared(self, x):
        if self.shared_expert is not None:
            return self.shared_expert(x)
        g = ttnn.matmul(x, self._shared_gate, compute_kernel_config=self.ckc_shared)
        u = ttnn.matmul(x, self._shared_up, compute_kernel_config=self.ckc_shared)
        return ttnn.matmul(ttnn.multiply(ttnn.silu(g), u), self._shared_down, compute_kernel_config=self.ckc_shared)

    # ---- forward ----

    def __call__(self, hidden_states_tt, **kwargs):
        x = hidden_states_tt
        if isinstance(x, torch.Tensor):
            kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            if self._is_mesh_dev:
                kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
            x = ttnn.from_torch(x.to(torch.bfloat16), **kw)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        orig_shape = list(x.shape)
        N = 1
        for d in orig_shape[:-1]:
            N *= d
        H, I, E = self.hidden_size, self.moe_intermediate, self._local_experts

        # ---- routing (on device, fp32) ----
        logits = self._router_logits(x, N)
        w = self._routing_weights(logits)  # [N, 64]
        w_loc = ttnn.matmul(w, self._sel, compute_kernel_config=self.ckc)  # [N, E_local]
        w_loc = ttnn.reshape(ttnn.transpose(w_loc, -2, -1), [E, N, 1])

        # ---- local experts: one batched SwiGLU over all tokens ----
        xe = ttnn.repeat(ttnn.reshape(x, [1, N, H]), ttnn.Shape([E, 1, 1]))  # [E, N, H]
        gu = ttnn.matmul(
            xe, self._w_gate_up, compute_kernel_config=self.ckc_expert, core_grid=ttnn.CoreGrid(y=8, x=8)
        )  # [E, N, 2I]
        ttnn.deallocate(xe)
        g = ttnn.slice(gu, [0, 0, 0], [E, N, I])
        u = ttnn.slice(gu, [0, 0, I], [E, N, 2 * I])
        ttnn.deallocate(gu)
        inter = ttnn.multiply(ttnn.silu(g), u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        y = ttnn.matmul(
            inter,
            self._w_down,
            compute_kernel_config=self.ckc_expert,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )  # [E, N, H]
        ttnn.deallocate(inter)
        y = ttnn.multiply(y, w_loc)
        routed = ttnn.sum(y, dim=0, keepdim=True)  # [1, N, H] fp32
        ttnn.deallocate(y)
        if self._shard:
            routed = ttnn.all_reduce(routed, cluster_axis=self._tp_axis, topology=ttnn.Topology.Linear)
        routed = ttnn.reshape(ttnn.typecast(routed, ttnn.bfloat16), orig_shape)

        shared = self._shared(x)
        if shared.dtype != ttnn.bfloat16:
            shared = ttnn.typecast(shared, ttnn.bfloat16)
        return ttnn.add(routed, ttnn.reshape(shared, orig_shape))


def build(device, torch_module=None, **kwargs):
    return TtMoE.build(device, torch_module, **kwargs)


def mo_e(device=None, torch_module=None, *args, **kwargs):
    return TtMoE.build(device, torch_module, **kwargs)
