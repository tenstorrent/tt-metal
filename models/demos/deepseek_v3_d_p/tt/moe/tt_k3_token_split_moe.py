# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 MoE block, token-split layout (moe_token_split_tp_se_plan.md): every chip owns S whole token rows
([S, H] in, [S, H] out) and runs, per call:

  1 router          gate matmul (bf16 weight, fp32 accumulation) + moe_grouped_topk (sigmoid, bias, top-16,
                    renormalized): ids / weights of its own tokens
  2 routing gather  all-gather of every chip's ids / weights (small): the replicated routing table
  3 planner         combine_rf.combine_plan (32 cores): the flat expert's counts / regions / token index, the
                    combine's hand-off table and per-call counts -- every table on the device
  4 latent down     [S, 7168] x [7168, 3584] (bf8)
  5 dispatch        all-gather of the latent rows (dense, the baseline; sparse 2D next): the gathered tokens
  6 experts+combine flat_routed_expert_with_program: this chip's experts on the gathered tokens (indexed by the
                    token index), each y row handed from the down cores' L1 straight into the router-forwarded
                    combine (combine_rf), which sums gate weight x row into fp32 per own token
  7 latent norm/up  RMSNorm(3584) then [S, 3584] x [3584, 7168] (bf8)
  8 shared expert   SiTU-GLU MLP, 7168 -> 6144 -> 7168 (bf8), replicated weights on the own rows
  9 sum             routed + shared

Routed expert weights: bfp4 in the flat expert's bank layout. For bring-up the routed experts come from a pool of P
distinct experts (global expert g uses pool[g % P]; the routing is over all 896): 896 real experts do not fit on a host.
"""

import os
from dataclasses import dataclass, field

import torch
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, port_luts
from ttnn.bringup.combine_rf.l1_zero import l1_zero_program
from ttnn.bringup.combine_rf.sparse_combine_rf import sparse_combine_rf_prepare
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import W_DTYPES, FlatRoutedExpert, _bank_sharded, flat_weight_regions

import ttnn


@dataclass
class K3Dims:
    H: int = 7168
    latent: int = 3584
    I: int = 3072  # routed expert intermediate
    I_shared: int = 6144  # 2 shared experts as one MLP
    E: int = 896
    K: int = 16
    beta: float = 4.0
    linear_beta: float = 25.0
    eps: float = 1e-6
    route_scale: float = 1.0


@dataclass
class K3MoEWeights:
    """Torch weights (fp32), reference layout (out, in) for nn.Linear ones."""

    gate: torch.Tensor  # [E, H]
    bias: torch.Tensor  # [E]
    lat_down: torch.Tensor  # [latent, H]
    lat_up: torch.Tensor  # [H, latent]
    lat_norm: torch.Tensor  # [latent]
    sh_gate: torch.Tensor  # [I_s, H]
    sh_up: torch.Tensor  # [I_s, H]
    sh_down: torch.Tensor  # [H, I_s]
    pool: list = field(default_factory=list)  # P x (w1 gate [I, latent], w3 up [I, latent], w2 down [latent, I])


def random_k3_weights(d: K3Dims, pool_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.02: torch.randn(*s, generator=g) * sc  # noqa: E731
    pool = [(r(d.I, d.latent), r(d.I, d.latent), r(d.latent, d.I)) for _ in range(pool_size)]
    return K3MoEWeights(
        gate=r(d.E, d.H, sc=0.02),
        bias=r(d.E, sc=0.01),
        lat_down=r(d.latent, d.H),
        lat_up=r(d.H, d.latent),
        lat_norm=1.0 + r(d.latent, sc=0.1),
        sh_gate=r(d.I_shared, d.H),
        sh_up=r(d.I_shared, d.H),
        sh_down=r(d.H, d.I_shared),
        pool=pool,
    )


def reference_k3_moe(x, w: K3MoEWeights, d: K3Dims, ids=None, wts=None):
    """Host model of the block (the reference modeling_kimi_moe.KimiSparseMoeBlock math, pooled experts). x [T, H]
    fp32. ids / wts: use this routing instead of computing it (to check the rest against the device's routing).
    Returns y [T, H], ids [T, K], wts [T, K]."""
    x = x.float()
    if ids is None:
        scores = torch.sigmoid(x @ w.gate.T)
        _, ids = torch.topk(scores + w.bias, d.K, dim=-1)
        wts = scores.gather(1, ids)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * d.route_scale
    lat = x @ w.lat_down.T
    acc = torch.zeros(x.shape[0], d.latent)
    P = len(w.pool)
    for p in range(P):  # every expert of the pool slot at once
        sel = (ids % P) == p
        tok, kk = sel.nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w1, w3, w2 = w.pool[p]
        h = lat[tok]
        g_, u_ = h @ w1.T, h @ w3.T
        a = d.beta * torch.tanh(g_ / d.beta) * torch.sigmoid(g_) * (d.linear_beta * torch.tanh(u_ / d.linear_beta))
        acc.index_add_(0, tok, (a @ w2.T) * wts[tok, kk].float().unsqueeze(1))
    yr = acc * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + d.eps) * w.lat_norm
    yr = yr @ w.lat_up.T
    gs, us = x @ w.sh_gate.T, x @ w.sh_up.T
    a = d.beta * torch.tanh(gs / d.beta) * torch.sigmoid(gs) * (d.linear_beta * torch.tanh(us / d.linear_beta))
    return yr + a @ w.sh_down.T, ids, wts


class TtK3TokenSplitMoE:
    """The block on a mesh (R x C, FABRIC_2D). S own tokens per chip."""

    def __init__(self, mesh_device, w: K3MoEWeights, d: K3Dims, S, *, cap=8192, rows_cap=16384, planner_cores=32):
        self.mesh, self.d, self.S = mesh_device, d, S
        R, C = tuple(mesh_device.shape)
        self.R, self.C, self.N = R, C, R * C
        assert d.E % self.N == 0
        self.EPC = d.E // self.N
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        dram = ttnn.DRAM_MEMORY_CONFIG
        tw = lambda t, dt: ttnn.from_torch(  # noqa: E731  (a [in, out] weight, replicated)
            t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=dram, mesh_mapper=rep
        )
        # router (bf16), latent (bf8), shared (bf8)
        self.w_gate = tw(w.gate.T, ttnn.bfloat16)
        self.bias = ttnn.from_torch(
            w.bias.view(1, -1).expand(S, -1).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, memory_config=dram, mesh_mapper=rep,
        )
        self.w_ld = tw(w.lat_down.T, ttnn.bfloat8_b)
        self.w_lu = tw(w.lat_up.T, ttnn.bfloat8_b)
        self.g_ln = ttnn.from_torch(
            w.lat_norm.view(1, 1, -1, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
            memory_config=dram, mesh_mapper=rep,
        )
        self.w_sg, self.w_su = tw(w.sh_gate.T, ttnn.bfloat8_b), tw(w.sh_up.T, ttnn.bfloat8_b)
        self.w_sd = tw(w.sh_down.T, ttnn.bfloat8_b)
        self.cfg_hi = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.cfg = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # routed experts: the flat expert (fake-initialized, then the pool packed once and replicated)
        E, EPC = d.E, self.EPC
        self.op = FlatRoutedExpert(
            mesh_device, "fake", m=cap, H=d.latent, I=d.I, gids=[list(range(c * EPC, (c + 1) * EPC)) for c in range(self.N)],
            n_global=E, wdtype="bf4", act="situ", pin=1,
        )
        P = len(w.pool)
        assert EPC % P == 0, "every chip's local expert e uses pool[e % P]: EPC must be a multiple of the pool"
        lay = dict(self.op.plan)
        W_l = [tuple(t.T.contiguous() for t in (w.pool[e % P][0], w.pool[e % P][1])) + (w.pool[e % P][2].T.contiguous(),) for e in range(EPC)]
        regions, dregs, rregs = flat_weight_regions(W_l, lay)
        wdt = W_DTYPES["bf4"][0]
        for t in (self.op.w_gu, self.op.w_d, self.op.w_rd):  # (the fake tensors)
            if t is not None:
                ttnn.deallocate(t)
        self.op.w_gu = _bank_sharded([regions], lay["banks"], wdt, mesh_device)
        self.op.w_d = _bank_sharded([dregs], lay["banks"], wdt, mesh_device)
        if lay["rdown"]:
            self.op.w_rd = _bank_sharded([rregs], lay["banks"], wdt, mesh_device)
        del W_l, regions, dregs, rregs
        plan = self.op.plan
        self.writers = len(plan["down"]) + (len(plan["rdn"]) if plan["rdown"] else 0)
        grid = mesh_device.compute_with_storage_grid_size()
        taken = {tuple(c) for k in ("arena_cores", "gu", "down", "readers", "relays", "coords") for c in plan[k]}
        pcores = {(i % 8, i // 8) for i in range(planner_cores)}
        self.cells = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
        wcores = [tuple(c) for c in plan["down"]] + ([tuple(c) for c in plan["rdn"]] if plan["rdown"] else [])
        wcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in wcores])
        # (the writers' words first: persistent L1 above the arena, not stranded in the middle when it is freed)
        self.words = ttnn.from_torch(
            torch.zeros(len(wcores) * 192, 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device, memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(wcrs, [192, 16], ttnn.ShardOrientation.ROW_MAJOR),
            ), mesh_mapper=rep,
        )
        # the arena: the flat expert's buffers, the combine's state / rings / CBs (~all of L1 on every core). Per call:
        # allocated after the dense ops before the experts, its combine state zeroed on the device, freed after (a
        # persistent one leaves the other ops no L1 for their circular buffers). The dispatch all-gather between the
        # zeroing and the experts is the cross-chip barrier (no chip's combine writes into another's arena before
        # that one is zeroed); the combine's ports drain their credits, so nothing lands after the call either.
        # Allocated here once for its address (the kernels bake it: every call's arena must land there).
        self._warm_up()  # (the dense ops once first: what they keep in L1 across calls sits above the arena)
        a_rows = int(plan["arena_tiles"]) * 32
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self._arena_spec = (
            ttnn.Shape([grid.x * grid.y * a_rows, 16]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(all_cores, [a_rows, 16], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        self.arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        self.arena_addr = int(self.arena.buffer_address())
        # the planner's outputs (allocated once) then the combine (its accumulator count sets the planner's A)
        luts, nps = port_luts(mesh_device, 2)
        self.rows_cap = rows_cap
        chips = [(r, c) for r in range(R) for c in range(C)]
        self._planner_args = dict(S=S, K=d.K, EPC=EPC, L=2, port_luts=luts, n_ports=nps, rows_cap=rows_cap,
                                  cores=sorted(pcores, key=lambda c: (c[1], c[0])))
        probe = combine_planner_prepare(mesh_device, A=15, **self._planner_args)  # (outputs only: their addresses)
        self.combine = sparse_combine_rf_prepare(
            mesh_device, S, d.latent, {ch: [] for ch in chips}, num_links=2, n_acc=16, depth=int(os.environ.get("K3_D", "4")),
            cells=self.cells, arena=self.arena, writers=self.writers,
            l1=dict(writers=wcores, words=int(self.words.buffer_address())),
            devplan=dict(plan=probe.outs["plan"], table=probe.outs["table"]),
        )
        self.A = len(self.combine.cells["accs"])
        self.planner = combine_planner_prepare(mesh_device, A=self.A, **self._planner_args) if self.A != 15 else probe
        if self.planner is not probe:  # (the combine baked the probe's tensors: give the planner those)
            self.planner.outs.update(probe.outs)
        # per-call buffers (allocated once)
        sh = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        self.y = ttnn.from_torch(torch.zeros(self.N, rows_cap, d.latent, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram, mesh_mapper=sh)
        self.out = ttnn.from_torch(torch.zeros(self.N, 1, S, d.latent, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram, mesh_mapper=sh)
        self.desc = self.l1_args = None  # (built on the first call: its CBs bind that call's arena)
        self.coll = self.combine.collector
        zr = {c: [(self.arena_addr + o, b, st, n) for o, b, st, n in runs] for c, runs in self.combine.zero_runs.items()}
        for c in wcores:
            zr.setdefault(c, []).append((int(self.words.buffer_address()), 192 * 16 * 4, 0, 1))
        self.zero_desc = l1_zero_program(mesh_device, zr)
        ttnn.deallocate(self.arena)
        self.arena = None
        self.key = 7_000_000 + id(self) % 1_000_000

    def _warm_up(self):
        """Every dense op once on zero tokens (e.g. CCL semaphores are allocated on first use and kept)."""
        x = ttnn.from_torch(
            torch.zeros(self.N * self.S, self.d.H, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        ids, wts = self.router(x)
        gi, gw = self.gather_routing(ids, wts)
        lat = self.latent_down(x)
        xg = self.dispatch(lat)
        o = ttnn.to_layout(ttnn.reshape(lat, [self.S, self.d.latent]), ttnn.TILE_LAYOUT)
        n = ttnn.rms_norm(o, epsilon=self.d.eps, weight=self.g_ln, compute_kernel_config=self.cfg_hi)
        yr = ttnn.linear(n, self.w_lu, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)
        y = ttnn.add(yr, self.shared(x))
        ttnn.synchronize_device(self.mesh)
        for t in (x, ids, wts, gi, gw, lat, xg, o, n, yr, y):
            ttnn.deallocate(t)

    # -- the ops (per chip; tensors on the mesh) --
    def router(self, x):
        logits = ttnn.linear(x, self.w_gate, compute_kernel_config=self.cfg_hi, dtype=ttnn.bfloat16)
        wts, ids = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits, self.bias, n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=self.d.K,
            route_scale=self.d.route_scale, stable_sort=True, epsilon=1e-20, score_func="sigmoid",
            padding_config=None, weights_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.deallocate(logits)
        return ids, wts

    def gather_routing(self, ids, wts):
        """ids (uint16 TILE [S, K]) / weights (bf16 row-major [S, K]) -> the planner's single-page rows [1, N S K],
        home-major (an all-gather along the last dim over the columns, then the rows), replicated."""

        def ag(t):
            a = ttnn.all_gather(t, dim=1, cluster_axis=1)
            b = ttnn.all_gather(a, dim=1, cluster_axis=0)
            ttnn.deallocate(a)
            return b

        S, K = self.S, self.d.K
        t = ttnn.typecast(ids, ttnn.uint32)  # (row-major uint16 has no reshape: through uint32)
        t2 = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(t)
        t3 = ttnn.reshape(t2, [1, S * K])
        ids_row = ttnn.typecast(t3, ttnn.uint16)
        ttnn.deallocate(t2)
        w = wts if wts.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(wts, ttnn.ROW_MAJOR_LAYOUT)
        w_row = ttnn.reshape(w if w.dtype == ttnn.bfloat16 else ttnn.typecast(w, ttnn.bfloat16), [1, S * K])
        gi, gw = ag(ids_row), ag(w_row)
        ttnn.deallocate(ids_row)
        return gi, gw

    def latent_down(self, x):
        lat = ttnn.linear(x, self.w_ld, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)
        return ttnn.to_layout(lat, ttnn.ROW_MAJOR_LAYOUT)

    def dispatch(self, lat_rm):
        a = ttnn.all_gather(lat_rm, dim=0, cluster_axis=1)
        b = ttnn.all_gather(a, dim=0, cluster_axis=0)
        ttnn.deallocate(a)
        return b  # [N * S, latent]: gathered token home * S + t

    def alloc_arena(self):
        """The call's arena (uninitialized) with the combine's state zeroed."""
        arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        assert int(arena.buffer_address()) == self.arena_addr, (
            f"the arena landed at {int(arena.buffer_address())}, not {self.arena_addr}: an L1 tensor outlives its op"
        )
        if self.desc is None:
            self.desc, self.l1_args = self.combine.build_l1(self.y, self.out, arena)
        else:  # (the CBs bind the arena's buffer: a program-cache miss rebuilds them from this one)
            self.desc = self.combine.rebind(arena)
        ttnn.generic_op([arena, self.words], self.zero_desc)
        return arena

    def experts_combine(self, xg, pl, arena):
        return ttnn.bringup.flat_routed_expert_with_program(
            xg, pl["counts"], pl["regions"], self.op.gidx, self.op.w_gu, self.op.w_d, reader_down_weights=self.op.w_rd,
            done_words=self.op.done, intermediate=self.d.I, max_tokens_per_expert=self.op.m, arena=arena, y=self.y,
            extra=self.desc, extra_key=self.key, collector=self.coll["core"], counts_addr=self.coll["counts_addr"],
            writers=self.writers, activation=self.op.act, pin=1, token_index=pl["token_index"], down_fp32=True,
            l1_args=self.l1_args,
        )

    def latent_up(self):
        o = ttnn.to_layout(ttnn.reshape(self.out, [self.S, self.d.latent]), ttnn.TILE_LAYOUT)
        n = ttnn.rms_norm(o, epsilon=self.d.eps, weight=self.g_ln, compute_kernel_config=self.cfg_hi)
        ttnn.deallocate(o)
        y = ttnn.linear(n, self.w_lu, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(n)
        return y

    def shared(self, x):
        g = ttnn.linear(x, self.w_sg, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)
        u = ttnn.linear(x, self.w_su, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)
        a = ttnn.situ_glu(g, u, self.d.beta, self.d.linear_beta)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        y = ttnn.linear(a, self.w_sd, compute_kernel_config=self.cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(a)
        return y

    def __call__(self, x, keep=False, hook=None):
        """x: [S, H] bf16 TILE per chip. Returns y [S, H] (and the intermediates with keep=True). hook(stage name),
        if given, is called after every stage (e.g. to read the profiler per stage)."""
        mark = hook or (lambda _: None)
        os.environ["MIMO_FL_Y_L1"] = "1"
        ids, wts = self.router(x)
        mark("router")
        gi, gw = self.gather_routing(ids, wts)
        mark("routing_gather")
        pl = self.planner(gi, gw)
        mark("planner")
        lat = self.latent_down(x)
        mark("lat_down")
        arena = self.alloc_arena()
        mark("arena_zero")
        xg = self.dispatch(lat)  # (also the barrier: every chip zeroed its arena before any chip's experts start)
        mark("dispatch")
        self.experts_combine(xg, pl, arena)
        ttnn.deallocate(arena)
        mark("experts_combine")
        yr = self.latent_up()
        mark("lat_up")
        ys = self.shared(x)
        mark("shared")
        y = ttnn.add(yr, ys)
        mark("sum")
        inter = dict(ids=ids, wts=wts, gi=gi, gw=gw, lat=lat, xg=xg, yr=yr, ys=ys) if keep else None
        if not keep:
            for t in (ids, wts, gi, gw, lat, xg, yr, ys):
                ttnn.deallocate(t)
        return (y, inter) if keep else y
