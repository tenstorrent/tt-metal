# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview router: 256 experts, top-8, sigmoid scores, e_score_correction_bias (selection only), norm_topk_prob, x routed_scaling_factor.

Score function and formulas are the ref-oracle's (reference/oracle.py ``router``):
    s = sigmoid(x @ W^T)                 [N, 256]  (fp32)
    sel = topk(s + e_score_correction_bias, 8)        (n_group = topk_group = 1: no group masking; bias picks, never scales)
    w = s[sel] / (sum s[sel] + 1e-20) * 2.827
The DSV4 hi/lo-bias trick is not needed: the whole router runs in fp32 (matmul, sigmoid, rank add) and the exact top-8 is taken on fp32 ranks
(``ttnn.topk`` or the JIT ``router_select`` program; both exact), like DSV4's ``DSV41_ROUTER=exact/fused`` path.

CPU part (no ttnn): ``route_cpu`` (fp32 oracle maths and a device-emulating variant: bf16 input and weight, fp32 accumulate).
Device part: ``Hy4Gate`` (imports ttnn lazily).
"""
import os

import torch

N_EXPERTS = 256
TOPK = 8
SCALE = 2.827
EPS = 1e-20


def route_cpu(x, w, bias, k=TOPK, scale=SCALE, emulate_device=False, eps=EPS):
    """x [N, D], w [E, D] (checkpoint layout), bias [E] fp32 -> (weights [N, k] fp32, ids [N, k] int64, rank [N, E] fp32).
    emulate_device: x and w rounded to bf16 first (what the device sees), matmul accumulated in fp32."""
    if emulate_device:
        x, w = x.to(torch.bfloat16), w.to(torch.bfloat16)
    s = torch.sigmoid(x.float() @ w.float().T)
    rank = s + bias.float()
    ids = rank.topk(k, dim=-1)[1]
    sel = s.gather(1, ids)
    return sel / (sel.sum(-1, keepdim=True) + eps) * scale, ids, rank


def set_match(ids_a, ids_b):
    """fraction of rows whose top-k SETS are equal (order ignored)."""
    a, b = ids_a.sort(-1)[0], ids_b.sort(-1)[0]
    return (a == b).all(-1).float().mean().item()


def near_tie_rows(rank, k=TOPK, tol=1e-3):
    """rows whose k-th and (k+1)-th rank values are within tol (a flip there is a legitimate tie, not a bug)."""
    v = rank.topk(k + 1, dim=-1)[0]
    return ((v[:, k - 1] - v[:, k]).abs() < tol).nonzero().flatten()


class Hy4Gate:
    """Device router. ``forward(tt_x [1,1,T,6144] bf16 TILE)`` -> (weights bf16 RM [T,1,1,8], indices uint16 RM [T,1,1,8]).
    mode (env HY4_ROUTER, default ``fused`` = router_select JIT program, ``exact`` = ttnn.topk + one-hot select)."""

    def __init__(self, mesh_device, torch_gate_weight, torch_gate_bias, k=TOPK, scale=SCALE, mode=None):
        import ttnn

        self.ttnn = ttnn
        self.md, self.k, self.scale = mesh_device, k, float(scale)
        self.mode = mode or os.environ.get("HY4_ROUTER", "fused")
        self._sk = None
        from models.demos.blackhole.hy4_preview.tt.hy4_skinny import enabled

        if (
            enabled("router") and self.mode == "fused"
        ):  # HY4_SKINNY=router (default OFF): bank-local logits + sigmoid + bias kernel, T <= 8
            from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyRouter

            self._sk = SkinnyRouter(mesh_device, torch_gate_weight, torch_gate_bias, k=k, scale=scale)
        E, D = torch_gate_weight.shape  # checkpoint layout [E, hidden]
        assert E == N_EXPERTS and torch_gate_bias.numel() == E
        self.E, self.D = E, D
        rep = ttnn.ReplicateTensorToMesh(mesh_device)

        def up(t, dt):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        self._w = up(torch_gate_weight.to(torch.bfloat16).T.contiguous().reshape(1, 1, D, E), ttnn.bfloat16)
        self._bias = up(torch_gate_bias.float().reshape(1, 1, 1, E), ttnn.float32)
        self._zero = up(torch.zeros(1, 1, 1, E), ttnn.float32)
        self._arange = up(torch.arange(E, dtype=torch.float32).reshape(1, 1, 1, E), ttnn.float32)
        self._arange_col = up(torch.arange(E, dtype=torch.float32).reshape(1, 1, E, 1), ttnn.float32)
        self._ones_k = up(torch.ones(1, 1, k, k), ttnn.float32)
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def route_words(self, tt_x):
        """TP32 route tensor int32 RM [1,1,1,16 T] straight from the skinny router (select merge writes the packed words), or None when this gate has no skinny router / T > 8."""
        if (
            self._sk is not None
            and self._sk.ksplit
            and self._sk.fsel
            and int(tt_x.shape[2]) <= (16 if os.environ.get("HY4_B30_ROUTE16", "0") == "1" else 8)
        ):
            return self._sk.route(tt_x, int(tt_x.shape[2]))
        return None

    def scores(self, tt_x, grid=(1, 8), mc=None):
        """fp32 sigmoid scores [1,1,T,E] (DRAM unless mc)."""
        ttnn = self.ttnn
        mc = mc or ttnn.L1_MEMORY_CONFIG
        logits = ttnn.matmul(
            tt_x,
            self._w,
            compute_kernel_config=self.ckc,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]),
            memory_config=mc,
        )
        score = ttnn.sigmoid(logits, memory_config=mc)
        ttnn.deallocate(logits)
        return score

    def forward(self, tt_x, grid=(1, 8)):
        if self._sk is not None and int(tt_x.shape[2]) <= 8:
            return self._sk.forward(tt_x, int(tt_x.shape[2]))
        if self.mode == "fused":
            return self._forward_fused(tt_x, grid)
        return self._forward_exact(tt_x, grid)

    def _forward_fused(self, tt_x, grid):
        from models.demos.blackhole.hy4_preview.tt.hy4_router_select import router_select

        ttnn = self.ttnn
        L1 = ttnn.L1_MEMORY_CONFIG
        score = self.scores(tt_x, grid)
        rank = ttnn.add(score, self._bias, memory_config=L1)
        w, idx = router_select(rank, score, self.k, EPS, self.scale)
        ttnn.deallocate(rank)
        ttnn.deallocate(score)
        return w, idx

    def _forward_exact(self, tt_x, grid):
        ttnn = self.ttnn
        T, E, k = tt_x.shape[2], self.E, self.k
        mc = ttnn.L1_MEMORY_CONFIG
        score = self.scores(tt_x, grid)
        rank = ttnn.add(score, self._bias, memory_config=mc)
        _, idx = ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True, memory_config=mc)
        ttnn.deallocate(rank)
        idxf = ttnn.typecast(idx, ttnn.float32, memory_config=mc)
        onehot = ttnn.eq(self._arange_col, ttnn.reshape(idxf, [T, 1, 1, k]), memory_config=mc)  # [T,1,E,k]
        sel = ttnn.matmul(
            ttnn.reshape(score, [T, 1, 1, E]),
            onehot,
            compute_kernel_config=self.ckc,
            dtype=ttnn.float32,
            memory_config=mc,
        )
        den = ttnn.matmul(sel, self._ones_k, compute_kernel_config=self.ckc, dtype=ttnn.float32, memory_config=mc)
        U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, EPS)],
            activations=[U(UT.MUL_UNARY_SFPU, self.scale)],
            dtype=ttnn.bfloat16,
            memory_config=mc,
        )
        weights = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        indices = ttnn.view(
            ttnn.to_layout(
                ttnn.typecast(idx, ttnn.uint16, memory_config=mc),
                ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            (T, 1, 1, k),
        )
        return weights, indices
