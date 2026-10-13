# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 routed experts (128 x GeGLU 704, top-8) for prefill on the 8x4 Galaxy mesh, composed from stock TTNN ops.

Reference: out[t] = sum_e w[t, e] * down_e(gelu_tanh(gate_e(x[t])) * up_e(x[t])), w = the router's dense [S, 128]
(zero off the top-8).

Layout: the input is replicated. Each mesh row takes its S/8 tokens (mesh_partition on axis 0, a local slice), each
column c holds experts 32c..32c+31 (EP=4) and its 32 columns of the routing matrix. Per chip, dense over its 32 experts:
    gate = x @ Wg [H, 32*704], up = x @ Wu [H, 32*704]       (expert-major blocks of 704)
    act  = gelu_tanh(gate) * up * (w_local @ M)               (M [32, 32*704] 0/1: w[t, e] over expert e's block)
    part = act @ Wd [32*704, H]                               (the sum over the 32 experts is the matmul's K reduction)
then all_reduce over the columns (the 128-expert sum) and all_gather over the rows (the tokens back, replicated).
An unselected expert has w = 0 exactly, so it adds exactly 0. Every expert runs for every token (dense, 16x the
top-8 work); a dispatch/combine (token sort per expert) is the performance follow-up, the interface stays.
"""

import torch

import ttnn

from .settings import settings

_FIDELITY = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}
_GELU = {"Tanh": ttnn.GeluVariant.Tanh, "Accurate": ttnn.GeluVariant.Accurate}

SP_AXIS = 0
EP_AXIS = 1


class TtExperts:
    def __init__(self, mesh, gate_up, down):
        """gate_up: HF [E, 2I, H] (gate rows first, then up); down: HF [E, H, I]."""
        self.mesh = mesh
        ep = mesh.shape[EP_AXIS]
        n_exp, two_i, hidden = gate_up.shape
        inter = two_i // 2
        assert n_exp % ep == 0, f"{n_exp} experts over EP={ep}"
        assert inter % 32 == 0, f"expert intermediate {inter} not tile aligned"
        self.local = n_exp // ep
        self.inter = inter
        self.split_tokens = settings.get("EXPERTS_SPLIT_TOKENS")

        # [H, E*I], expert-major: column block e is expert e; ShardTensor2dMesh over the columns gives chip c its 32
        gate = gate_up[:, :inter, :].float().permute(2, 0, 1).reshape(hidden, n_exp * inter)
        up = gate_up[:, inter:, :].float().permute(2, 0, 1).reshape(hidden, n_exp * inter)
        dn = down.float().permute(0, 2, 1).reshape(n_exp * inter, hidden)  # [E*I, H], row block e is expert e
        self.w_gate = self._shard(gate, dim=-1)
        self.w_up = self._shard(up, dim=-1)
        self.w_down = self._shard(dn, dim=-2)
        del gate, up, dn

        sel = torch.zeros(self.local, self.local * inter)
        for e in range(self.local):
            sel[e, e * inter : (e + 1) * inter] = 1.0
        self.sel = ttnn.from_torch(
            sel.reshape(1, 1, *sel.shape),
            dtype=ttnn.bfloat16,  # 0/1, exact
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.mm_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=_FIDELITY[settings.get("EXPERTS_MATMUL_FIDELITY")],
            math_approx_mode=False,
            fp32_dest_acc_en=settings.get("EXPERTS_MATMUL_FP32_ACC"),
            packer_l1_acc=False,
        )
        self.gelu = _GELU[settings.get("MLP_GELU_VARIANT")]

    def _shard(self, w, dim):
        return ttnn.from_torch(
            w.reshape(1, 1, *w.shape),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, tuple(self.mesh.shape), dims=(None, dim)),
        )

    def __call__(self, x, routing):
        """x: [1, 1, S, H] bf16, routing: [1, 1, S, E] dense (fp32 or bf16), both replicated. Returns [1, 1, S, H]
        bf16, replicated."""
        split = self.split_tokens and self.mesh.shape[SP_AXIS] > 1
        xs = ttnn.mesh_partition(x, dim=2, cluster_axis=SP_AXIS) if split else x
        rs = ttnn.mesh_partition(routing, dim=2, cluster_axis=SP_AXIS) if split else routing
        w = ttnn.mesh_partition(rs, dim=3, cluster_axis=EP_AXIS)  # [1, 1, T, 32]: this chip's experts
        if split:
            ttnn.deallocate(rs)
        if w.dtype != ttnn.float32:
            wf = ttnn.typecast(w, ttnn.float32)
            ttnn.deallocate(w)
            w = wf

        gate = ttnn.linear(xs, self.w_gate, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        up = ttnn.linear(xs, self.w_up, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        if split:
            ttnn.deallocate(xs)
        act = ttnn.gelu(gate, variant=self.gelu)
        ttnn.deallocate(gate)
        h = ttnn.mul(act, up, dtype=ttnn.float32)
        ttnn.deallocate(act)
        ttnn.deallocate(up)
        wexp = ttnn.linear(w, self.sel, dtype=ttnn.float32, compute_kernel_config=self.mm_cfg)  # [1, 1, T, 32*I]
        ttnn.deallocate(w)
        hw = ttnn.mul(h, wexp, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        ttnn.deallocate(wexp)
        part = ttnn.linear(hw, self.w_down, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(hw)
        red = ttnn.all_reduce(part, cluster_axis=EP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(part)
        if not split:
            return red
        out = ttnn.all_gather(red, dim=2, cluster_axis=SP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(red)
        return out
