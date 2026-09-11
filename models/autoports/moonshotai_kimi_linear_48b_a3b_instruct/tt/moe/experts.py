# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Routed experts with ``ttnn.sparse_matmul`` (Gemma4/GPT-OSS active-expert pattern), TP over the expert intermediate dim.

Weights: gate/up [1, E, H, I/tp] (column-parallel), down [1, E, I/tp, H] (row-parallel). ``forward`` returns the per-chip
PARTIAL combined expert output [1,1,S,H]; the caller adds the shared-expert partial and all-reduces once.
Decode (S <= 32): sparsity = the dense routing tensor; ``nnz=None`` because on Blackhole a bf16 routing weight can flush to
zero and a wrong static nnz deadlocks the kernel. Prefill: 32-token groups with all-ones sparsity (every expert computed,
routing weights select) — functional baseline; token-sorted expert matmuls are the optimisation-stage replacement.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.weights import as_device_tensor, tp_of

TILE = 32


def _sparse_program_config(m: int, n: int, in0_block_w: int = 1):
    n_tiles = int(math.ceil(n / TILE))
    best = (1, 1, 1)
    for cores in range(1, min(65, n_tiles + 1)):
        if n_tiles % cores:
            continue
        for cy in range(1, 9):
            if cores % cy == 0 and cores // cy <= 8 and cores > best[0]:
                best = (cores, cores // cy, cy)
                break
    cores, cx, cy = best
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=n_tiles // cores,
        per_core_M=max(TILE, m) // TILE,
        per_core_N=n_tiles // cores,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class KimiExperts:
    def __init__(
        self,
        mesh_device,
        cfg: KimiLinearConfig,
        sd: dict | None,
        *,
        name: str,
        cache_path: Path | None,
        dtype=ttnn.bfloat8_b,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.E, self.H, self.I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.tp = tp_of(mesh_device)
        assert self.I % (self.tp * TILE) == 0, (self.I, self.tp)
        self.I_loc = self.I // self.tp

        def prep(t, kind):  # torch [E, out, in] -> [1, E, in, out]
            return None if t is None else t.transpose(-2, -1).unsqueeze(0).contiguous()

        g = None if sd is None else sd["moe.experts.gate"]
        u = None if sd is None else sd["moe.experts.up"]
        d = None if sd is None else sd["moe.experts.down"]
        kw = dict(cache_path=cache_path, dtype=dtype)
        self.gate = as_device_tensor(mesh_device, prep(g, "gate"), name=f"{name}.gate", shard_dim=-1, **kw)
        self.up = as_device_tensor(mesh_device, prep(u, "up"), name=f"{name}.up", shard_dim=-1, **kw)
        self.down = as_device_tensor(mesh_device, prep(d, "down"), name=f"{name}.down", shard_dim=-2, **kw)
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self._ones = None
        self._ones_sparsity()  # allocate before any trace capture: a later allocation can sit in a trace's scratch region

    # ---- shared group kernel ------------------------------------------------------------------
    def _group(self, x: ttnn.Tensor, routing: ttnn.Tensor, sparsity: ttnn.Tensor, nnz, mem) -> ttnn.Tensor:
        """x [1,1,S,H] (S <= 32 rows), routing [1,1,S,E], sparsity [1,1,1,E] ROW_MAJOR bf16 (which experts to run for this group).
        Every active expert is applied to all S rows; the per-row routing weights then keep only that row's top-k. Returns
        the per-chip partial [1,1,S,H]."""
        S = x.shape[2]
        E, I, H = self.E, self.I_loc, self.H
        tile = ttnn.Tile([32, 32])
        pc_gu = _sparse_program_config(S, I, self.in0_block_w)
        pc_d = _sparse_program_config(S, H, self.in0_block_w_down)
        kw = dict(
            sparsity=sparsity,
            nnz=nnz,
            memory_config=mem,
            output_tile=tile,
            compute_kernel_config=self.compute,
            dtype=ttnn.bfloat16,
        )
        g = ttnn.sparse_matmul(x, self.gate, program_config=pc_gu, **kw)  # [1,1,E,S_tile,I]
        g = (
            ttnn.reshape(ttnn.transpose(g, 1, 3), (1, E, S, g.shape[-1]))
            if len(g.shape) == 6
            else ttnn.reshape(g, (1, E, S, g.shape[-1]))
        )
        u = ttnn.sparse_matmul(x, self.up, program_config=pc_gu, **kw)
        u = (
            ttnn.reshape(ttnn.transpose(u, 1, 3), (1, E, S, u.shape[-1]))
            if len(u.shape) == 6
            else ttnn.reshape(u, (1, E, S, u.shape[-1]))
        )
        h = ttnn.multiply(ttnn.silu(g), u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        d = ttnn.sparse_matmul(h, self.down, program_config=pc_d, is_input_a_sparse=True, **kw)  # [1,E,S,H]
        ttnn.deallocate(h)
        d = ttnn.reshape(d, (1, E, S, H))
        d = ttnn.multiply(d, ttnn.permute(routing, (0, 3, 2, 1)))  # [1,E,S,1] per-row weights (0 for non-selected)
        out = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(d, dims=[1]))
        ttnn.deallocate(d)
        return ttnn.reshape(out, (1, 1, S, H))

    # ---- decode -----------------------------------------------------------------------------
    def forward_decode(self, x: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,B,H] (B <= 32), routing [1,1,B,E] -> partial [1,1,B,H]. Runs the UNION of the batch's active experts
        (exactly top-k for one user); nnz inferred on device (a static nnz that mismatches would deadlock)."""
        B = x.shape[2]
        if B < TILE:  # the sparse kernel works on 32-row tiles: pad rows (zero routing rows contribute nothing)
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, TILE - B), (0, 0)], value=0.0)
            routing = ttnn.pad(routing, [(0, 0), (0, 0), (0, TILE - B), (0, 0)], value=0.0)
        union = ttnn.sum(routing, dim=2, keepdim=True)  # [1,1,1,E] > 0 where any user picked the expert
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(union)
        out = self._group(x, routing, sparsity, None, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sparsity)
        if B < TILE:
            out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, B, self.H))
        return out

    # ---- prefill ----------------------------------------------------------------------------
    def _ones_sparsity(self):
        if self._ones is None:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if self.mesh_device.get_num_devices() > 1 else None
            self._ones = ttnn.from_torch(
                torch.ones(1, 1, 1, self.E),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=mapper,
            )
        return self._ones

    # sparse_matmul inner block widths (K tiles per block). 1 (the gpt-oss/gemma4 default) made the decode MoE 12.4 ms per
    # layer at 32 users; 24 gives 2.8 ms (8-expert single user: 2.2 -> 1.5 ms), identical outputs. K = 2304 = 72 tiles for
    # gate/up, K = I_loc = 256 = 8 tiles for down.
    in0_block_w = 24
    in0_block_w_down = 8
    prefill_group = 32  # tokens per all-experts sparse_matmul group (sparse path)
    prefill_impl = "dense"  # "dense": batched dense matmuls over all experts (whole core grid); "sparse": 32-token sparse_matmul groups
    prefill_dense_chunk = (
        1024  # tokens per dense sub-chunk: g/u/h are [1,E,S,I_loc] bf16 (134 MB at 1024); 0.10 s/layer at 2048 tokens
    )

    def forward_prefill_dense(self, x: ttnn.Tensor, routing: ttnn.Tensor, chunk: int | None = None) -> ttnn.Tensor:
        """Every expert is applied to every token in prefill anyway, so run it as dense matmuls that fill the core grid:
        g,u = x @ W[e] for all e (batch-broadcast matmul, [1,E,S,I]); h = silu(g)*u scaled by the per-row routing weight
        (0 for non-selected experts); permute to [S, E*I] and finish with ONE dense down projection against the down
        weights viewed as [E*I, H] (a free reshape of [1,E,I,H]). ~7 ops per sub-chunk instead of ~10 per 32-token group.
        """
        T = x.shape[2]
        E, I, H = self.E, self.I_loc, self.H
        step = (
            chunk or self.prefill_dense_chunk
        )  # sub-chunks of up to ``step`` rows; any tile multiple works for a dense matmul
        down_flat = ttnn.reshape(self.down, (1, 1, E * I, H))  # view; E,I contiguous in [1,E,I,H]
        outs = []
        for s in range(0, T, step):
            n = min(step, T - s)
            xs = x if n == T else ttnn.slice(x, (0, 0, s, 0), (1, 1, s + n, H))
            rs = routing if n == T else ttnn.slice(routing, (0, 0, s, 0), (1, 1, s + n, E))
            g = ttnn.matmul(
                xs,
                self.gate,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute,
            )  # [1,E,S,I]
            u = ttnn.matmul(
                xs,
                self.up,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute,
            )
            h = ttnn.multiply(ttnn.silu(g), u)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            h = ttnn.multiply(
                h, ttnn.permute(rs, (0, 3, 2, 1))
            )  # [1,E,S,1] per-row expert weights (0 for non-selected)
            hp = ttnn.permute(h, (0, 2, 1, 3))  # [1,S,E,I]
            ttnn.deallocate(h)
            hp = ttnn.reshape(hp, (1, 1, n, E * I))
            o = ttnn.matmul(
                hp,
                down_flat,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute,
            )  # [1,1,S,H]
            ttnn.deallocate(hp)
            outs.append(o)
            if xs is not x:
                ttnn.deallocate(xs)
            if rs is not routing:
                ttnn.deallocate(rs)
        out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        return out

    def forward_prefill(self, x: ttnn.Tensor, routing: ttnn.Tensor, chunk: int | None = None) -> ttnn.Tensor:
        if self.prefill_impl == "dense":
            return self.forward_prefill_dense(x, routing, chunk)
        return self.forward_prefill_sparse(x, routing, chunk)

    def forward_prefill_sparse(self, x: ttnn.Tensor, routing: ttnn.Tensor, chunk: int | None = None) -> ttnn.Tensor:
        """x [1,1,T,H] (T % 32 == 0), routing [1,1,T,E] -> partial [1,1,T,H]. All experts per ``chunk``-token group."""
        T = x.shape[2]
        chunk = chunk or min(self.prefill_group, T)
        while T % chunk:
            chunk //= 2
        ones = self._ones_sparsity()
        outs = []
        for s in range(0, T, chunk):
            # a full-range slice returns the input tensor itself -> never deallocate an alias of x / routing
            xs = x if T == chunk else ttnn.slice(x, (0, 0, s, 0), (1, 1, s + chunk, self.H))
            rs = routing if T == chunk else ttnn.slice(routing, (0, 0, s, 0), (1, 1, s + chunk, self.E))
            outs.append(self._group(xs, rs, ones, self.E, ttnn.DRAM_MEMORY_CONFIG))
            if xs is not x:
                ttnn.deallocate(xs)
            if rs is not routing:
                ttnn.deallocate(rs)
        out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        return out
