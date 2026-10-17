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
from ttnn.bringup.combine_rf.dispatch_prep import all_chips_route_words, dispatch_prep_prepare, own_rows_copy_prepare
from ttnn.bringup.combine_rf.sparse_all_gather_2d import ensure_semaphores, manifest_spec, port_cores_all, sparse_all_gather_2d
from ttnn.bringup.combine_rf.fabric_all_gather_py import fabric_all_gather, worker_cores
from ttnn.bringup.combine_rf.fabric_all_gather_py import plan as fag_plan
from ttnn.bringup.combine_rf.l1_zero import l1_zero_program
from ttnn.bringup.combine_rf.layout_ops import layout_op_prepare
from ttnn.bringup.combine_rf.router_topk import router_topk_prepare
from ttnn.bringup.combine_rf.sparse_combine_rf import sparse_combine_rf_prepare
from ttnn.bringup.combine_rf.sum_ops import sum_parts_prepare
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import W_DTYPES, FlatRoutedExpert, _bank_sharded, flat_weight_regions

import models.demos.glm53_flash_d_p.tt.mm_configs as mm_configs
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
    eps: float = 1e-5  # (rms_norm_eps, reference/kimi_k3/config.json)
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
        # router selection: K3_TOPK=custom (default: sigmoid + bias add in fp32, router_topk writes the joint routing
        # rows) or ttnn (moe_grouped_topk + layout / concat ops)
        self.topk_custom = os.environ.get("K3_TOPK", "custom") == "custom"
        self.bias32 = ttnn.from_torch(
            w.bias.view(1, -1).float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
            memory_config=dram, mesh_mapper=rep,
        )
        # (one ttnn.add with a sigmoid on its input gives the biased scores; the top-k reads only those and the bias row,
        # each pick's score = biased - bias: one program and half the top-k's reads fewer)
        self.topk_bias = os.environ.get("K3_TOPK_BIAS", "1") == "1"
        self.bias_row = ttnn.from_torch(
            w.bias.view(1, -1).float().contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
            memory_config=dram, mesh_mapper=rep,
        )
        self.rtopk = router_topk_prepare(mesh_device, S=S, E=d.E, K=d.K, route_scale=d.route_scale, bias_mode=self.topk_bias)
        self.w_ld = tw(w.lat_down.T, ttnn.bfloat8_b)
        self.w_lu = tw(w.lat_up.T, ttnn.bfloat8_b)
        self.g_ln = ttnn.from_torch(
            w.lat_norm.view(1, 1, -1, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
            memory_config=dram, mesh_mapper=rep,
        )
        # shared expert: K3_SE=flat (default) runs it as the flat routed expert with one expert every own token goes
        # to (bf8 weights in its bank layout, fused gate/up + SiTU + down); dense: three ttnn.linear + situ_glu
        # latent norm: the combine's rows tilized straight into an 8 x 10 block-sharded L1 tensor (1 row tile x 14 tiles a
        # core), the sharded rms_norm (7 us vs 58 interleaved), back to DRAM interleaved for the latent up
        gx_ln = 8
        self.ln_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx_ln - 1, S // 32 - 1))]),
            [32, d.latent // gx_ln], ttnn.ShardOrientation.ROW_MAJOR))
        self.ln_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx_ln, S // 32), subblock_w=2, block_h=1,
            block_w=d.latent // gx_ln // 32, inplace=False)
        self.tilize_ln = layout_op_prepare(mesh_device, "tilize", out_memory_config=self.ln_mc)
        self.tilize = layout_op_prepare(mesh_device, "tilize")
        self.untilize = layout_op_prepare(mesh_device, "untilize")
        self.untilize7 = layout_op_prepare(mesh_device, "untilize", pages_per_row=7)
        self.se_mode = os.environ.get("K3_SE", "flat")  # flat (local) | tp (TP over each mesh row) | dense
        self.se_flat = self.se_mode in ("flat", "tp")
        self.se_parts = int(os.environ.get("K3_SE_PARTS", "3"))
        if self.se_mode == "tp":
            self._build_shared_tp(mesh_device, w, d, S)
        elif self.se_flat:
            self._build_shared_flat(mesh_device, w, d, S)
        else:
            self.w_sg, self.w_su = tw(w.sh_gate.T, ttnn.bfloat8_b), tw(w.sh_up.T, ttnn.bfloat8_b)
            self.w_sd = tw(w.sh_down.T, ttnn.bfloat8_b)
        self.cfg_hi = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.cfg = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # the latent down / up linears (bf8 weights): K3_LAT_FID (HiFi2 default; LoFi as TtMoe's default config)
        self.cfg_lat = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("K3_LAT_FID", "HiFi2")), math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        # routed experts: the flat expert (fake-initialized, then the pool packed once and replicated). Its tokens and
        # its gate/up -> down intermediate in bfp8 (K3_X_BF16 / K3_H_BF16 = 1: bf16, accuracy probes)
        self.x_bf16 = os.environ.get("K3_X_BF16") == "1"
        self.h_bf16 = os.environ.get("K3_H_BF16") == "1"
        self.wdtype = os.environ.get("K3_WDTYPE", "bf4")  # routed expert weights: bf4 (default) or bf8
        self.act = os.environ.get("K3_ROUTED_ACT", "situ")  # (silu: an accuracy probe of the device SiTU)
        E, EPC = d.E, self.EPC
        self.op = FlatRoutedExpert(
            mesh_device, "fake", m=cap, H=d.latent, I=d.I, gids=[list(range(c * EPC, (c + 1) * EPC)) for c in range(self.N)],
            n_global=E, wdtype=self.wdtype, act=self.act, pin=1, x_bf16=self.x_bf16, h_bf16=self.h_bf16,
        )
        P = len(w.pool)
        assert EPC % P == 0, "every chip's local expert e uses pool[e % P]: EPC must be a multiple of the pool"
        lay = dict(self.op.plan)
        W_l = [tuple(t.T.contiguous() for t in (w.pool[e % P][0], w.pool[e % P][1])) + (w.pool[e % P][2].T.contiguous(),) for e in range(EPC)]
        regions, dregs, rregs = flat_weight_regions(W_l, lay)
        wdt = W_DTYPES[self.wdtype][0]
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
        self.sparse_dispatch = os.environ.get("K3_DISPATCH", "sparse") == "sparse"
        self.rows_pt = 32 if os.environ.get("K3_TOPK", "custom") == "custom" else 1  # (routing rows per page)
        # the latent down output untilized straight into this chip's rows of the dispatch buffer (layout_ops with a row
        # offset), dispatch_prep then only writes the route words: -18 us of own-row copy for +2.5 us of untilize
        self.lat_direct = self.sparse_dispatch and os.environ.get("K3_LAT_DIRECT", "1") == "1"
        self.dprep = (
            dispatch_prep_prepare(mesh_device, S=S, K=d.K, EPC=self.EPC, tokens_per_page=self.rows_pt, copy=not self.lat_direct)
            if self.sparse_dispatch
            else None
        )
        if self.lat_direct:  # (and both in one generic_op, the untilize on the cores the route words leave)
            C_ = tuple(mesh_device.shape)[1]
            self.untilize_x = layout_op_prepare(mesh_device, "untilize", row0_of=lambda r, c: (r * C_ + c) * S,
                                                exclude=self.dprep.cores)
            self._prep_merged = {}
        # the routing rows' all-gather: one sparse_all_gather_2d to every chip (default) or two ttnn.all_gather
        # (sparse: sparse_all_gather_2d to every chip, 157 us vs 140 for the two ttnn.all_gather at 64-B rows; fag: the
        # dense fabric all-gather, one snake)
        self.rgather = os.environ.get("K3_RGATHER", "fag")
        self.rgather_fabric = self.rgather == "sparse"
        if self.sparse_dispatch or self.rgather_fabric:
            ensure_semaphores(mesh_device, num_links=2)  # (persistent: before anything per call)
        if self.rgather_fabric:
            self.rcopy = own_rows_copy_prepare(mesh_device, S=S)
            self.route_all = all_chips_route_words(mesh_device, S)
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
                                  cores=sorted(pcores, key=lambda c: (c[1], c[0])), row_pages=True, joint=True,
                                  tokens_per_page=32 if os.environ.get("K3_TOPK", "custom") == "custom" else 1)
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
        # the shared expert's x untilize in another op's program, on the cores it leaves: the planner's (60 us on 32
        # cores; slows it to 76 - traced 3710 either way) or the sparse dispatch's (fabric bound, its port cores only)
        self.se_x_where = os.environ.get("K3_SE_X_IN", "own")  # own | plan | dispatch (dispatch: 230 -> 284 us)
        self.se_x_in_plan = self.se_x_where == "plan" and hasattr(self.planner, "programs")
        if self.se_x_in_plan:
            self.untilize_plan = layout_op_prepare(mesh_device, "untilize", exclude=self.planner.cores)
            self._plan_merged = {}
        if self.se_x_where == "gather":  # (the routing gather: a 160 KB fabric all-gather, latency bound, few cores)
            chips_, _ = fag_plan(mesh_device, cluster_axis=None, topology=ttnn.Topology.Linear, num_links=1)
            ex = [r_.start for r_ in worker_cores(chips_).ranges()]
            self.untilize_gather = layout_op_prepare(mesh_device, "untilize", exclude=ex)
        if self.se_x_where == "dispatch" and self.sparse_dispatch:
            ex = [ttnn.CoreCoord(*c) for c in port_cores_all(mesh_device, num_links=2)]
            self.untilize_disp = layout_op_prepare(mesh_device, "untilize", exclude=ex)
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

    @staticmethod
    def _linear(x, w, cfg, dtype=ttnn.bfloat16, memory_config=None):
        """ttnn.linear with the 2D-multicast config for the shape (models/demos/glm53_flash_d_p/tt/mm_configs.mm2d):
        2-4x the default at 320 rows (zz_scratch_k3_dense_mm.py)."""
        pc = mm_configs.linear_config(x, w, dtype)
        return ttnn.linear(x, w, compute_kernel_config=cfg, dtype=dtype, program_config=pc, memory_config=memory_config)

    def _warm_up(self):
        """Every dense op once on zero tokens (e.g. CCL semaphores are allocated on first use and kept)."""
        x = ttnn.from_torch(
            torch.zeros(self.N * self.S, self.d.H, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        ids, wts = self.router(x)
        gi, gw = self.gather_routing(ids, wts)
        lat = self.latent_down(x)
        xg = self.dispatch(lat, self.dispatch_prep(lat, gi))
        if self.lat_direct:  # (tiled, maybe L1: to DRAM before the shared expert's whole-L1 arena)
            o = ttnn.to_memory_config(lat, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(lat)
        else:
            o = ttnn.to_layout(ttnn.reshape(lat, [self.S, self.d.latent]), ttnn.TILE_LAYOUT)
        n = ttnn.rms_norm(o, epsilon=self.d.eps, weight=self.g_ln, compute_kernel_config=self.cfg_hi)
        yr = self._linear(n, self.w_lu, self.cfg)
        y = ttnn.add(yr, self.shared(x), dtype=ttnn.bfloat16)
        ttnn.synchronize_device(self.mesh)
        for t in (x, ids, wts, gi, lat, xg, o, n, yr, y):  # (gw is gi)
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)

    # -- the ops (per chip; tensors on the mesh) --
    def router(self, x, keep_logits=False):
        if self.topk_custom:  # (fp32 logits / scores / biased scores, router_topk: the joint rows; ids = the rows)
            mc = ttnn.L1_MEMORY_CONFIG if "router" in self.l1_set else None  # (the top-k's 64 B reads: L1 latency)
            logits = self._linear(x, self.w_gate, self.cfg_hi, ttnn.float32, memory_config=mc)
            if self.topk_bias:
                bs = ttnn.add(logits, self.bias32, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=mc)
                rows = self.rtopk(self.bias_row, bs)
            else:
                sc = ttnn.sigmoid(logits, memory_config=mc)
                bs = ttnn.add(sc, self.bias32, memory_config=mc)
                rows = self.rtopk(sc, bs)
                ttnn.deallocate(sc)
            ttnn.deallocate(bs)
            if keep_logits:
                if mc is not None:  # (no L1 tensor may outlive its op: the arena's address)
                    kept = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(logits)
                    logits = kept
                return rows, None, logits
            ttnn.deallocate(logits)
            return rows, None
        logits = self._linear(x, self.w_gate, self.cfg_hi)
        wts, ids = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits, self.bias, n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=self.d.K,
            route_scale=self.d.route_scale, stable_sort=True, epsilon=1e-20, score_func="sigmoid",
            padding_config=None, weights_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        if keep_logits:
            return ids, wts, logits
        ttnn.deallocate(logits)
        return ids, wts

    def gather_routing(self, ids, wts, x_se=None, x_rm=None):
        """ids (uint16 TILE [S, K]) / weights (bf16 row-major [S, K]) -> one [N S, 2 K] uint16 row-major tensor, each
        token's ids then its weights' bits (64 B a token), home-major (an all-gather over the columns, then the rows),
        replicated: the planner reads a page per token (row_pages, joint). With the custom top-k, ids are already the
        joint rows (wts None)."""
        if wts is None:
            both = ids
        else:
            ids_rm = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
            w = wts if wts.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(wts, ttnn.ROW_MAJOR_LAYOUT)
            w16 = ttnn.bitcast(w, ttnn.uint16)
            both = ttnn.concat([ids_rm, w16], dim=1)
            ttnn.deallocate(ids_rm)
        if self.rgather_fabric:  # (one sparse_all_gather_2d, every row to every chip: own rows in place first)
            N, S, K = self.N, self.S, self.d.K
            buf = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, N * S, 2 * K]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            self.rcopy(both, buf)
            ttnn.deallocate(both)
            man = ttnn.allocate_tensor_on_device(*manifest_spec(self.mesh, S, num_links=2))
            sparse_all_gather_2d(buf, self.route_all, num_links=2, manifest=man)
            ttnn.deallocate(man)
            g = ttnn.reshape(buf, [N * S, 2 * K])
            return g, g
        if self.rgather == "fag":  # (one dense fabric all-gather, a snake over the whole mesh: chip order kept)
            extra = None
            if x_se is not None:  # (and the shared expert's row-major x, untilized under it)
                ua = self.untilize_gather.programs(x_se, x_rm)
                extra = ([x_se, x_rm], lambda coord: (list(ua[coord].cbs), list(ua[coord].kernels)))
            g = fabric_all_gather(both, cluster_axis=None, topology=ttnn.Topology.Linear, num_links=1, dim=0, extra=extra)
            ttnn.deallocate(both)
            return g, g
        a = ttnn.all_gather(both, dim=0, cluster_axis=1)
        ttnn.deallocate(both)
        g = ttnn.all_gather(a, dim=0, cluster_axis=0)
        ttnn.deallocate(a)
        return g, g

    def plan(self, gi, gw, x):
        """The combine plan; with se_x_in_plan also the shared expert's row-major x ([S, H], DRAM), in one program."""
        if not (self.se_x_in_plan and self.se_mode == "flat"):
            return self.planner(gi, gw), None
        x_rm = ttnn.allocate_tensor_on_device(ttnn.Shape([self.S, self.d.H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh,
                                              ttnn.DRAM_MEMORY_CONFIG)
        k = tuple(int(t.buffer_address()) for t in (gi, gw, x, x_rm))
        if self._plan_merged.get("key") != k:
            pa, ua = self.planner.programs(gi, gw), self.untilize_plan.programs(x, x_rm)
            md = ttnn.MeshProgramDescriptor()
            for (r, c), pp in pa.items():
                pd, pu = ttnn.ProgramDescriptor(), ua[(r, c)]
                pd.cbs = list(pp.cbs) + list(pu.cbs)
                pd.semaphores = list(pp.semaphores)
                pd.kernels = list(pp.kernels) + list(pu.kernels)
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = pd
            self._plan_merged = {"key": k, "desc": md}
        ttnn.generic_op([gi, gw, x, x_rm] + self.planner.tensors, self._plan_merged["desc"])
        return self.planner.outs, x_rm

    def latent_down(self, x):
        """[S, latent]: tiled (lat_direct: dispatch_prep untilizes it into the dispatch buffer), else row-major."""
        l1 = "latdown" in self.l1_set  # (the tiled result is transient: L1; the row-major one feeds the dispatch: DRAM)
        lat = self._linear(x, self.w_ld, self.cfg_lat, memory_config=ttnn.L1_MEMORY_CONFIG if l1 else None)
        if self.lat_direct:
            return lat
        rm = ttnn.to_layout(lat, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(lat)
        return rm

    def dispatch_prep(self, lat_rm, gi):
        """Sparse dispatch: the buffer [1, 1, N S, latent] with this chip's own rows in place and the route words of
        its own tokens (dispatch_prep.py, one generic_op); the dense baseline (K3_DISPATCH=dense) needs nothing."""
        if not self.sparse_dispatch:
            return None
        N, S, W = self.N, self.S, self.d.latent
        dram = ttnn.DRAM_MEMORY_CONFIG
        xbuf = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, N * S, W]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, dram)
        route = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, S]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, dram)
        if self.lat_direct:  # (lat_rm is the tiled latent: own rows straight into the buffer; one program, both ops)
            k = tuple(int(t.buffer_address()) for t in (lat_rm, gi, xbuf, route))
            if self._prep_merged.get("key") != k:
                ua, da = self.untilize_x.programs(lat_rm, xbuf), self.dprep.programs(gi, None, xbuf, route)
                md = ttnn.MeshProgramDescriptor()
                for (r, c), pu in ua.items():
                    pd, pr = ttnn.ProgramDescriptor(), da[(r, c)]
                    pd.cbs = list(pu.cbs) + list(pr.cbs)
                    pd.kernels = list(pu.kernels) + list(pr.kernels)
                    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = pd
                self._prep_merged = {"key": k, "desc": md}
            ttnn.generic_op([lat_rm, gi, xbuf, route], self._prep_merged["desc"])
        else:
            self.dprep(gi, lat_rm, xbuf, route)
        return xbuf, route

    def dispatch(self, lat_rm, prep=None, x_se=None):
        """The gathered tokens [N S, latent] (row home * S + t). Sparse (default): each row only to the chips holding
        one of its experts (sparse_all_gather_2d, its neighbour fences and END markers make it a barrier as the
        dense all-gathers were); dense: two all-gathers."""
        if prep is None:
            a = ttnn.all_gather(lat_rm, dim=0, cluster_axis=1)
            b = ttnn.all_gather(a, dim=0, cluster_axis=0)
            ttnn.deallocate(a)
            return b
        xbuf, route = prep
        # (the port cores' manifest rings per call: a persistent one would hold ~123 KB of every core's L1)
        man = ttnn.allocate_tensor_on_device(*manifest_spec(self.mesh, self.S, num_links=2))
        extra, x_rm = None, None
        if x_se is not None:  # (the shared expert's row-major x, untilized under the dispatch)
            x_rm = ttnn.allocate_tensor_on_device(ttnn.Shape([self.S, self.d.H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                                  self.mesh, ttnn.DRAM_MEMORY_CONFIG)
            ua = self.untilize_disp.programs(x_se, x_rm)
            extra = ([x_se, x_rm], lambda coord: (list(ua[coord].cbs), list(ua[coord].kernels)))
        sparse_all_gather_2d(xbuf, route, num_links=2, manifest=man, extra=extra)
        ttnn.deallocate(man)
        ttnn.deallocate(route)
        g = ttnn.reshape(xbuf, [self.N * self.S, self.d.latent])
        return (g, x_rm) if x_se is not None else g

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
            x_bf16=self.x_bf16, h_bf16=self.h_bf16,
            l1_args=self.l1_args,
        )

    def latent_up(self, yr_l1=False):
        o = self.tilize_ln(ttnn.reshape(self.out, [self.S, self.d.latent]))  # (layout_ops: 19 us vs ttnn's 53)
        ns = ttnn.rms_norm(o, epsilon=self.d.eps, weight=self.g_ln, program_config=self.ln_pc, memory_config=self.ln_mc,
                           compute_kernel_config=self.cfg_hi)
        ttnn.deallocate(o)
        n = ttnn.sharded_to_interleaved(ns, ttnn.L1_MEMORY_CONFIG if "latup" in self.l1_set else ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ns)
        y = self._linear(n, self.w_lu, self.cfg_lat, memory_config=ttnn.L1_MEMORY_CONFIG if yr_l1 else None)
        ttnn.deallocate(n)
        return y

    def _se_knobs(self):
        self.se_fid = {"LoFi": 0, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}[os.environ.get("K3_SE_FID", "HiFi2")]
        self.se_xbf16 = os.environ.get("K3_SE_XBF16", "1") == "1"
        self.se_yrm = os.environ.get("K3_SE_YRM") == "1"
        self.se_ybf16 = os.environ.get("K3_SE_YBF16", "1") == "1" and not self.se_yrm
        self.fuse_sum = self.se_mode == "flat" and self.se_ybf16 and os.environ.get("K3_FUSE_SUM", "1") == "1"
        # transient intermediates in L1 (router logits / scores: the top-k 49.6 -> 36 us; latent up input / output, latent
        # down output: the linears write L1, -4..-8 us each); K3_L1= (empty) keeps them in DRAM
        self.l1_set = set(os.environ.get("K3_L1", "router,latup,yr,latdown").split(","))
        self.sum_parts = sum_parts_prepare(self.mesh, P=self.se_parts)
        self.se_srnd = os.environ.get("K3_SE_SRND") == "1"

    def _build_shared_tp(self, mesh_device, w, d, S):
        """TP shared expert over each mesh row (C chips): the row's tokens all-gathered ([C S, H] row-major, the dense
        fabric all-gather), chip column c holding intermediate columns c * I_s / C.. (one flat expert of I_s / C over
        all C S rows), the partial outputs reduce-scattered back to each chip's S rows."""
        R, C, N = self.R, self.C, self.N
        It = d.I_shared // C
        self._se_knobs()
        self.se = FlatRoutedExpert(
            mesh_device, "fake", m=C * S, H=d.H, I=It, gids=[[0] for _ in range(N)], n_global=1, wdtype="bf8",
            act="situ", pin=1, x_bf16=self.se_xbf16,
        )
        lay = dict(self.se.plan)
        per_col = []
        for c in range(C):
            sl = slice(c * It, (c + 1) * It)
            per_col.append(flat_weight_regions(
                [(w.sh_gate[sl].T.contiguous(), w.sh_up[sl].T.contiguous(), w.sh_down[:, sl].T.contiguous())], lay
            ))
        wdt = W_DTYPES["bf8"][0]
        for t in (self.se.w_gu, self.se.w_d, self.se.w_rd):
            if t is not None:
                ttnn.deallocate(t)
        dev_order = [per_col[i % C] for i in range(N)]  # (device r * C + c holds column c's slice)
        self.se.w_gu = _bank_sharded([g for g, _, _ in dev_order], lay["banks"], wdt, mesh_device)
        self.se.w_d = _bank_sharded([dd for _, dd, _ in dev_order], lay["banks"], wdt, mesh_device)
        if lay["rdown"]:
            self.se.w_rd = _bank_sharded([rr for _, _, rr in dev_order], lay["banks"], wdt, mesh_device)
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        row = lambda v: ttnn.from_torch(  # noqa: E731
            torch.tensor([v], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep,
        )
        self.se_counts, self.se_regions, self.se_tidx = row([C * S]), row([0]), None

    def _build_shared_flat(self, mesh_device, w, d, S):
        """The shared expert as SE_P experts of I_shared / SE_P intermediate columns each (the flat expert's gate/up
        split takes <= 64 tile columns per expert at H 7168 bf8), every own token to all of them (indexed rows: token
        index 0..S-1 per expert); y = the sum of their outputs (down over disjoint intermediate slices)."""
        N, P = self.N, self.se_parts
        Ip = d.I_shared // P
        # accuracy (zz_k3_moe_accuracy / se_err, vs the exact bf8-weight model): LoFi truncates bf8 weights to 5 bits
        # (3.3% rel, gain 0.982); HiFi2 + bf16 tokens + bf16 tiled y 1.2% (gain 0.997), as the ttnn.linear path; HiFi3 /
        # HiFi4 no better and slower. Knobs: K3_SE_FID (LoFi / HiFi2 / ...), K3_SE_XBF16 / K3_SE_YBF16 (default 1),
        # K3_SE_YRM bf16 row-major y, K3_SE_SRND stochastic packer rounding
        self._se_knobs()
        self.se = FlatRoutedExpert(
            mesh_device, "fake", m=S, H=d.H, I=Ip, gids=[list(range(P)) for _ in range(N)], n_global=P, wdtype="bf8",
            act="situ", pin=1, x_bf16=self.se_xbf16,
        )
        lay = dict(self.se.plan)
        sl = lambda i: slice(i * Ip, (i + 1) * Ip)  # noqa: E731
        regions, dregs, rregs = flat_weight_regions(
            [(w.sh_gate[sl(i)].T.contiguous(), w.sh_up[sl(i)].T.contiguous(), w.sh_down[:, sl(i)].T.contiguous()) for i in range(P)],
            lay,
        )
        wdt = W_DTYPES["bf8"][0]
        for t in (self.se.w_gu, self.se.w_d, self.se.w_rd):
            if t is not None:
                ttnn.deallocate(t)
        self.se.w_gu = _bank_sharded([regions], lay["banks"], wdt, mesh_device)
        self.se.w_d = _bank_sharded([dregs], lay["banks"], wdt, mesh_device)
        if lay["rdown"]:
            self.se.w_rd = _bank_sharded([rregs], lay["banks"], wdt, mesh_device)
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        row = lambda v: ttnn.from_torch(  # noqa: E731
            torch.tensor([v], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep,
        )
        self.se_counts, self.se_regions = row([S] * P), row([i * S for i in range(P)])
        self.se_tidx = row(list(range(S)) * P)

    def shared(self, x, yr=None, parts_only=False, x_rm=None):
        """The shared expert's y [S, H]; with yr (local flat SE, bf16 tile parts) the block's y = yr + the parts' sum
        in one pass (sum_ops) instead of ttnn.sum + ttnn.add."""
        if self.se_mode == "tp":  # (row all-gather, one flat expert on the row's tokens, reduce-scatter)
            x_rm = self.untilize7(x)  # (a row as 7 pages of 2 KB: the fabric all-gather sends whole pages, the flat reads 2 KB segments)
            xg = fabric_all_gather(x_rm, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=2, dim=0)
            ttnn.deallocate(x_rm)
            y = self.se(xg, self.se_counts, self.se_regions, x_pages_per_row=7, down_fp32=True,
                        pack_stochastic_rounding=self.se_srnd, y_bf16=self.se_ybf16, math_fidelity=self.se_fid)
            ttnn.deallocate(xg)
            ys = ttnn.reduce_scatter(y, 0, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear)
            ttnn.deallocate(y)
            return ys
        if self.se_flat:  # (x row major; y bfp8 tiles [P S, H]: the parts' outputs, summed)
            x_rm = self.untilize(x) if x_rm is None else x_rm  # (layout_ops: 32 us vs ttnn's 49; or the planner's)
            y = self.se(
                x_rm, self.se_counts, self.se_regions, token_index=self.se_tidx, down_fp32=True, y_row_major=self.se_yrm,
                pack_stochastic_rounding=self.se_srnd, y_bf16=self.se_ybf16, math_fidelity=self.se_fid,
            )
            ttnn.deallocate(x_rm)
            if self.se_yrm:
                yt = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
                ttnn.deallocate(y)
                y = yt
            if parts_only:
                return y
            if yr is not None and self.fuse_sum:
                out = self.sum_parts(yr, y)
                ttnn.deallocate(y)
                return out
            ys = ttnn.sum(ttnn.reshape(y, [self.se_parts, self.S, self.d.H]), dim=0)
            ttnn.deallocate(y)
            return ys
        g = self._linear(x, self.w_sg, self.cfg)
        u = self._linear(x, self.w_su, self.cfg)
        a = ttnn.situ_glu(g, u, self.d.beta, self.d.linear_beta)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        y = self._linear(a, self.w_sd, self.cfg)
        ttnn.deallocate(a)
        return y

    def __call__(self, x, keep=False, hook=None):
        """x: [S, H] bf16 TILE per chip. Returns y [S, H] (and the intermediates with keep=True). hook(stage name),
        if given, is called after every stage (e.g. to read the profiler per stage)."""
        mark = hook or (lambda _: None)
        os.environ["MIMO_FL_Y_L1"] = "1"
        ids, wts, *logits = self.router(x, keep_logits=keep)
        mark("router")
        x_rm = None
        if self.se_x_where == "gather" and self.rgather == "fag" and self.se_mode == "flat":
            x_rm = ttnn.allocate_tensor_on_device(ttnn.Shape([self.S, self.d.H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                                  self.mesh, ttnn.DRAM_MEMORY_CONFIG)
            gi, gw = self.gather_routing(ids, wts, x_se=x, x_rm=x_rm)
        else:
            gi, gw = self.gather_routing(ids, wts)
        mark("routing_gather")
        pl, x_rm2 = self.plan(gi, gw, x)
        x_rm = x_rm if x_rm is not None else x_rm2
        mark("planner")
        lat = self.latent_down(x)
        mark("lat_down")
        prep = self.dispatch_prep(lat, gi)
        if self.lat_direct and lat.memory_config().buffer_type == ttnn.BufferType.L1:  # (no L1 tensor past the arena)
            kept = ttnn.to_memory_config(lat, ttnn.DRAM_MEMORY_CONFIG) if keep else None
            ttnn.deallocate(lat)
            lat = kept
        mark("dispatch_prep")
        arena = self.alloc_arena()
        mark("arena_zero")
        if x_rm is None and self.se_x_where == "dispatch" and self.sparse_dispatch and self.se_mode == "flat":
            xg, x_rm = self.dispatch(lat, prep, x_se=x)  # (also the barrier: every chip zeroed its arena first)
        else:
            xg = self.dispatch(lat, prep)  # (also the barrier: every chip zeroed its arena before any chip's experts start)
        mark("dispatch")
        self.experts_combine(xg, pl, arena)
        ttnn.deallocate(arena)
        mark("experts_combine")
        if self.fuse_sum and not keep:  # (shared expert parts, latent up, then one sum; keep: ys separately, for checks)
            parts = self.shared(x, parts_only=True, x_rm=x_rm)  # (first: its arena takes all L1, so yr may live in L1)
            mark("shared")
            yr = self.latent_up(yr_l1="yr" in self.l1_set)
            mark("lat_up")
            y, ys = self.sum_parts(yr, parts), None
            ttnn.deallocate(parts)
        else:
            yr = self.latent_up()
            mark("lat_up")
            ys = self.shared(x, x_rm=x_rm)
            mark("shared")
            y = ttnn.add(yr, ys, dtype=ttnn.bfloat16)
        mark("sum")
        inter = dict(ids=ids, wts=wts, gi=gi, gw=gw, lat=lat, xg=xg, yr=yr, ys=ys, logits=logits[0], routed=self.out, pl=pl) if keep else None
        if not keep:
            for t in (ids, wts, gi, lat, xg, yr, ys):  # (gw is gi)
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
        return (y, inter) if keep else y
