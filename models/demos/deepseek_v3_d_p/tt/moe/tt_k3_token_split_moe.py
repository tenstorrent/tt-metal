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

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, port_luts
from ttnn.bringup.combine_rf.dispatch_prep import all_chips_route_words, dispatch_prep_prepare, own_rows_copy_prepare
from ttnn.bringup.combine_rf.fabric_all_gather_py import fabric_all_gather
from ttnn.bringup.combine_rf.fabric_all_gather_py import plan as fag_plan
from ttnn.bringup.combine_rf.fabric_all_gather_py import worker_cores
from ttnn.bringup.combine_rf.l1_zero import l1_zero_program
from ttnn.bringup.combine_rf.layout_ops import layout_op_prepare
from ttnn.bringup.combine_rf.lin2d import glu_weight, lin2d_prepare
from ttnn.bringup.combine_rf.lin2d import upload_stacked as lin2d_upload
from ttnn.bringup.combine_rf.router_topk import router_topk_prepare
from ttnn.bringup.combine_rf.routing_pack import routing_pack_prepare
from ttnn.bringup.combine_rf.sparse_all_gather_2d import (
    ensure_semaphores,
    manifest_spec,
    port_cores_all,
    route_words,
    sparse_all_gather_2d,
)
from ttnn.bringup.combine_rf.sparse_combine_rf import sparse_combine_rf_prepare
from ttnn.bringup.combine_rf.sum_ops import sum_parts_prepare
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import (
    W_DTYPES,
    FlatRoutedExpert,
    _bank_sharded,
    flat_weight_regions,
)
from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import flat_layout, k3_gu64
from ttnn.bringup.prepared_program import generic_op as prepared_generic_op

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
    act: str = "situ"  # (the routed experts' activation: situ (K3), clamped_silu (GLM-5.3-Flash))


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


def combine_cells(mesh_device, plan):
    """The cells the flat expert plan leaves free (logical (x, y), row-major): the combine's (and its ports')."""
    grid = mesh_device.compute_with_storage_grid_size()
    # (DRISC: only the reader cells still running Tensix readers / the launcher are taken; the rest go to combine)
    rd_key = "tensix_readers" if "tensix_readers" in plan else "readers"
    taken = {tuple(c) for k in ("arena_cores", "gu", "down", rd_key, "relays", "coords") for c in plan[k]}
    return [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]


def combine_port_cells(mesh_device, cells, torus):
    """The combine's port cells among `cells` (one a direction and link: 2 links)."""
    from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan, place_ports

    chips_, _, _ = mesh_plan(mesh_device, 2, torus)
    return {(v.x, v.y) for v in place_ports(mesh_device, chips_, cells).values()}


def planned_accumulators(mesh_device, plan, torus):
    """The combine's accumulator count in the ports-apart mode: every free cell not a port, at most A_MAX."""
    from ttnn.bringup.combine_rf.sparse_combine_rf import A_MAX

    cells = combine_cells(mesh_device, plan)
    return min(A_MAX, len(cells) - len(combine_port_cells(mesh_device, cells, torus)))


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
    """The block on a mesh (R x C, FABRIC_2D or a torus fabric). S own tokens per chip. torus = (wrap X, wrap Y): the
    rings the mesh fully spans (a submesh only has the wrap links of the axes it covers end to end)."""

    def __init__(
        self,
        mesh_device,
        w: K3MoEWeights,
        d: K3Dims,
        S,
        *,
        cap=8192,
        rows_cap=16384,
        planner_cores=32,
        deterministic=False,
        torus=(False, False),
        weight_cache=None,
        acc_on_ports=True,
        routed_only=False,
        flat_layout=None,
        gu64=True,
        combine_depth=4,
        x_once=None,
        dw_batch=None,
        layout_overrides=None,
        shared_tp=False,
        shared_axis=1,
        joint_routing=False,
        combine_auto_cfg=False,
        combine_accumulators=None,
        dispatch_in_program=False,
        combine_snap=None,
    ):
        """routed_only (GLM-5.3-Flash hack): only the routed experts and their combine - no router (the caller's top-k
        goes to routed()), no latent projections (d.latent = d.H: the dispatch rows are x), no shared expert; the
        flat expert's weights come per call (routed(op=...): one block for every layer). flat_layout: the routed
        experts' layout (flat_layouts.py preset) instead of K3's (gu64: flat_layouts.k3_gu64, else the plan's
        defaults). combine_depth: the combine's rings per source chip. x_once / dw_batch / layout_overrides: the routed
        experts' program options for the fused call only (x read from DRAM once - default off with gu64 / routed_only;
        down-weight blocks per read barrier; any other FlatLayoutOptions fields). shared_tp (routed_only): the shared
        expert folded in, split over each mesh row's chips (shared_axis 1; 0: each column's) - the dispatch also sends
        every token to its group-mates,
        routed(shared=) runs the chip's slice on the row's tokens between the dispatch and the experts, and the combine
        carries the partials to their owners (its shared_rows); the chip's own partial is added to the output.
        joint_routing (routed_only): routed() takes the router's joint rows (combine_rf/router_topk.py's 32-token pages:
        ids, then weights' bf16 bits) as ids, wts None. combine_auto_cfg: the combine's accumulators, port placement and
        rings from the free cells and the arena (sparse_combine_rf.combine_auto, the close-out sizing) instead of 16
        accumulators, acc_on_ports and combine_depth. dispatch_in_program (routed_only, sparse dispatch, combine_auto_cfg):
        routed_prepped's dispatch runs as the first phase of the combine's port kernels inside the experts' program (its
        ports the combine's; the x relays wait until every port of their chip has its rows in: the flat layout's x_ready)
        - the weights stream from the program's start, under the dispatch."""
        self.mesh, self.d, self.S, self.torus = mesh_device, d, S, tuple(torus)
        self.routed_only = routed_only
        assert not routed_only or d.latent == d.H, "routed_only: no latent projections (d.latent = d.H)"
        self.shared_tp = bool(shared_tp)
        self.shared_axis = int(shared_axis)
        assert not shared_tp or routed_only, "shared_tp: the routed_only block (the shared expert per call)"
        R, C = tuple(mesh_device.shape)
        self.R, self.C, self.N = R, C, R * C
        self.G = C if self.shared_axis == 1 else R  # (the shared expert's group: chips sharing its slices)
        assert d.E % self.N == 0
        self.EPC = d.E // self.N
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        dram = ttnn.DRAM_MEMORY_CONFIG
        tw = lambda t, dt: ttnn.from_torch(  # noqa: E731  (a [in, out] weight, replicated)
            t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=dram, mesh_mapper=rep
        )
        self.tilize = layout_op_prepare(mesh_device, "tilize", cw=16)  # (1 KB row pieces: GLM [640, 4096] 35 -> 26 us)
        self.untilize = layout_op_prepare(mesh_device, "untilize")
        self.untilize7 = layout_op_prepare(mesh_device, "untilize", pages_per_row=7)
        self.topk_custom = os.environ.get("K3_TOPK", "custom") == "custom" and not routed_only
        self.se_mode = "none"
        if not routed_only:
            # router (bf16), latent (bf8), shared (bf8)
            self.w_gate = tw(w.gate.T, ttnn.bfloat16)
            self.bias = ttnn.from_torch(
                w.bias.view(1, -1).expand(S, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=dram,
                mesh_mapper=rep,
            )
            # router selection: K3_TOPK=custom (default: sigmoid + bias add in fp32, router_topk writes the joint routing
            # rows) or ttnn (moe_grouped_topk + layout / concat ops)
            self.bias32 = ttnn.from_torch(
                w.bias.view(1, -1).float().contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=dram,
                mesh_mapper=rep,
            )
            # (one ttnn.add with a sigmoid on its input gives the biased scores; the top-k reads only those and the bias row,
            # each pick's score = biased - bias: one program and half the top-k's reads fewer)
            self.topk_bias = os.environ.get("K3_TOPK_BIAS", "1") == "1"
            self.bias_row = ttnn.from_torch(
                w.bias.view(1, -1).float().contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=dram,
                mesh_mapper=rep,
            )
            self.rtopk = router_topk_prepare(
                mesh_device, S=S, E=d.E, K=d.K, route_scale=d.route_scale, bias_mode=self.topk_bias
            )
            self.w_ld = tw(w.lat_down.T, ttnn.bfloat8_b)
            self.w_lu = tw(w.lat_up.T, ttnn.bfloat8_b)
            # latent down / up as combine_rf/lin2d (K3_LIN2D=1, default): x resident in L1, the weights streamed down each
            # column by fixed senders (top NOC0 / bottom NOC1, after x is in), each weight piece one DRAM read; 103 / 95 us
            # standalone vs ttnn mm2d's 117 / 122 (zz_scratch_k3_dense_mm.py::test_lin2d)
            self.lin2d = os.environ.get("K3_LIN2D", "1") == "1"
            if self.lin2d:
                fid = getattr(ttnn.MathFidelity, os.environ.get("K3_LAT_FID", "HiFi2"))
                self.ld_run, lay = lin2d_prepare(mesh_device, M=S, K=d.H, N=d.latent, fidelity=fid)
                self.w_ld2 = lin2d_upload(mesh_device, w.lat_down.T.contiguous(), lay)
                self.lu_run, lay = lin2d_prepare(mesh_device, M=S, K=d.latent, N=d.H, fidelity=fid)
                self.w_lu2 = lin2d_upload(mesh_device, w.lat_up.T.contiguous(), lay)
            self.g_ln = ttnn.from_torch(
                w.lat_norm.view(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=dram,
                mesh_mapper=rep,
            )
            # shared expert: K3_SE=flat (default) runs it as the flat routed expert with one expert every own token goes
            # to (bf8 weights in its bank layout, fused gate/up + SiTU + down); dense: three ttnn.linear + situ_glu
            # latent norm: the combine's rows tilized straight into an 8 x 10 block-sharded L1 tensor (1 row tile x 14 tiles a
            # core), the sharded rms_norm (7 us vs 58 interleaved), back to DRAM interleaved for the latent up
            gx_ln = 8
            self.ln_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx_ln - 1, S // 32 - 1))]),
                    [32, d.latent // gx_ln],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            self.ln_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx_ln, S // 32),
                subblock_w=2,
                block_h=1,
                block_w=d.latent // gx_ln // 32,
                inplace=False,
            )
            self.tilize_ln = layout_op_prepare(mesh_device, "tilize", out_memory_config=self.ln_mc)
            # lin2d (default: local, two combine_rf/lin2d calls, the down adding the latent up's output) | flat (local, the
            # flat expert) | tp (TP over each mesh row) | dense (ttnn.linear)
            self.se_mode = os.environ.get("K3_SE", "lin2d")
            self.se_flat = self.se_mode in ("flat", "tp")
            self.se_parts = int(os.environ.get("K3_SE_PARTS", "3"))
            if self.se_mode == "tp":
                self._build_shared_tp(mesh_device, w, d, S)
            elif self.se_mode == "lin2d":
                self._build_shared_lin2d(mesh_device, w, d, S)
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
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("K3_LAT_FID", "HiFi2")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # routed experts: the flat expert (fake-initialized, then the pool packed once and replicated). Its tokens and
        # its gate/up -> down intermediate in bfp8 (K3_X_BF16 / K3_H_BF16 = 1: bf16, accuracy probes)
        self.x_bf16 = os.environ.get("K3_X_BF16") == "1"
        self.h_bf16 = os.environ.get("K3_H_BF16") == "1"
        self.wdtype = os.environ.get("K3_WDTYPE", "bf4")  # routed expert weights: bf4 (default) or bf8
        self.act = os.environ.get("K3_ROUTED_ACT", d.act)  # (silu: an accuracy probe of the device SiTU)
        E, EPC = d.E, self.EPC
        # routed experts' layout: K3_GU64=1 - 64 gate/up cores (16-column units, 3 a core) with DRISC weight readers only
        # (the 16 Tensix reader cells go to the combine) and 28 down cores (4 columns each); both DRISC tiles of a bank
        # on every call. 2 x 4 overlap harness, fused us uniform / Zipf spread / packed: 2289 / 2590 / 5705 vs the
        # 48 gate/up Tensix-reader layout's 2442 / 2727 / 5920. (Plan options are process-wide env; explicit ones win.)
        # Default on (K3_GU64=0: the 48 gate/up Tensix-reader layout); with it the routed experts read x twice (one
        # multicaster per rectangle: a single one for 64 gate/up cores back-pressured the whole call; block traced
        # forward 3407 -> 3315 us, experts_combine 2446 -> 2380 us; with x read once 3453 / 2476). Two x helpers per
        # rectangle (MIMO_FL_NH=2, on the cells the accumulators-on-ports layout frees): x delivery (~20 GB/s a relay)
        # capped compute-bound calls - single chip M 1024 182.6 -> 165.2 us / expert, harness fused packed 5807 ->
        # 5709 us, uniform / spread and the block's traced forward unchanged (3312 us)
        self.gu64 = gu64 and flat_layout is None
        self.flat_layout = flat_layout if flat_layout is not None else (k3_gu64(acc_on_ports) if self.gu64 else {})
        self.x_once, self.dw_batch, self.layout_overrides = x_once, dw_batch, dict(layout_overrides or {})
        self.dispatch_in_program = bool(dispatch_in_program)
        if self.dispatch_in_program:  # (the x relays wait for the ports' count: sparse_all_gather_2d.XREADY_N)
            from ttnn.bringup.combine_rf.sparse_all_gather_2d import XREADY_N

            assert routed_only and combine_auto_cfg, "dispatch_in_program: routed_only, combine_auto_cfg"
            self.layout_overrides["x_ready"] = XREADY_N
        self.op = FlatRoutedExpert(
            mesh_device,
            "fake",
            m=cap,
            H=d.latent,
            I=d.I,
            gids=[list(range(c * EPC, (c + 1) * EPC)) for c in range(self.N)],
            n_global=E,
            wdtype=self.wdtype,
            act=self.act,
            pin=1,
            x_bf16=self.x_bf16,
            h_bf16=self.h_bf16,
            layout=self.flat_layout,
        )
        P = len(w.pool) if not routed_only else 1
        assert EPC % P == 0, "every chip's local expert e uses pool[e % P]: EPC must be a multiple of the pool"
        lay = dict(self.op.plan)
        wdt = W_DTYPES[self.wdtype][0]
        for t in (self.op.w_gu, self.op.w_d, self.op.w_rd):  # (the fake tensors)
            if t is not None:
                ttnn.deallocate(t)
        names = ("w_gu", "w_d", "w_rd") if lay["rdown"] else ("w_gu", "w_d")
        # weight_cache (a directory): the packed (bfp) host tensors, keyed on the plan, the dtype and a sample of the
        # pool's values - a rebuild with the same weights and layout loads them instead of packing again
        files = dict.fromkeys(names)
        if weight_cache is not None and not routed_only:
            sample = [(tuple(t.shape), t.reshape(-1)[::4093].float().numpy().tobytes()) for p_ in w.pool for t in p_]
            key = hashlib.sha1(repr((sorted((k, repr(v)) for k, v in lay.items()), self.wdtype, EPC, sample)).encode())
            files = {n: Path(weight_cache) / f"k3_flat.{key.hexdigest()[:16]}.{n}.tensorbin" for n in names}
        # (a loaded tensor is file-backed (mmap): its upload spins forever on KMD 2.9 unless the pinned-memory cache is
        # off - TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 before the device opens; otherwise pack again)
        no_pin = os.environ.get("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES") == "0"
        if routed_only:  # (the weights come per call: routed(op=...))
            self.op.w_gu = self.op.w_d = self.op.w_rd = None
        elif no_pin and all(f is not None and f.is_file() for f in files.values()):
            mc = lambda t: t.memory_config()  # noqa: E731
            loaded = {n: ttnn._ttnn.tensor.load_tensor_flatbuffer(str(files[n])) for n in names}
            for n, t in loaded.items():
                setattr(self.op, n, t.to(mesh_device, mc(t)))
        else:
            W_l = [
                tuple(t.T.contiguous() for t in (w.pool[e % P][0], w.pool[e % P][1]))
                + (w.pool[e % P][2].T.contiguous(),)
                for e in range(EPC)
            ]
            src = dict(zip(("w_gu", "w_d", "w_rd"), flat_weight_regions(W_l, lay)))
            del W_l
            for n in names:
                setattr(self.op, n, _bank_sharded([src[n]], lay["banks"], wdt, mesh_device, files[n]))
            del src
        plan = self.op.plan
        self.writers = len(plan["down"]) + (len(plan["rdn"]) if plan["rdown"] else 0)
        grid = mesh_device.compute_with_storage_grid_size()
        # (DRISC: only the reader cells still running Tensix readers / the launcher are taken; the rest go to combine)
        pcores = {(i % 8, i // 8) for i in range(planner_cores)}
        self.cells = combine_cells(mesh_device, plan)
        wcores = [tuple(c) for c in plan["down"]] + ([tuple(c) for c in plan["rdn"]] if plan["rdown"] else [])
        wcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in wcores])
        # (the writers' words first: persistent L1 above the arena, not stranded in the middle when it is freed)
        self.words = ttnn.from_torch(
            torch.zeros(len(wcores) * 192, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(wcrs, [192, 16], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=rep,
        )
        # routed_only: the x relays' words, persistent (the fused op otherwise allocates them per call, at an address the
        # program-cache key includes but the caller cannot see: _experts_combine predicts its hits)
        self.relay_words = None
        if routed_only:
            from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import crs_rects

            rl = [ttnn.CoreCoord(*c) for c in plan["relays"]]
            self.relay_words = ttnn.from_torch(  # (zero: the x-ready word counts from it - dispatch_in_program)
                torch.zeros(len(rl) * 32, 32, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(crs_rects(rl), (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            self.relays = [tuple(c) for c in plan["relays"]]
        self.disp_sems = None
        if self.dispatch_in_program:  # (the dispatch's global semaphores on the combine's port cores: persistent L1,
            # created before the arena; the ports as the combine places them - acc_on_ports off, asserted below)
            from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan as _mp
            from ttnn.bringup.combine_rf.sparse_all_gather_2d import place_ports as _pp

            chips_d, _, _ = _mp(mesh_device, 2, self.torus)
            pxy = sorted({(c.x, c.y) for c in _pp(mesh_device, chips_d, self.cells).values()})
            pcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in pxy])
            self.disp_cores = pxy
            self.disp_sems = tuple(ttnn.create_global_semaphore(mesh_device, pcrs, 0) for _ in range(4))
        # the arena: the flat expert's buffers, the combine's state / rings / CBs (~all of L1 on every core). Per call:
        # allocated after the dense ops before the experts, its combine state zeroed on the device, freed after (a
        # persistent one leaves the other ops no L1 for their circular buffers). The dispatch all-gather between the
        # zeroing and the experts is the cross-chip barrier (no chip's combine writes into another's arena before
        # that one is zeroed); the combine's ports drain their credits, so nothing lands after the call either.
        # Allocated here once for its address (the kernels bake it: every call's arena must land there).
        self.sparse_dispatch = os.environ.get("K3_DISPATCH", "sparse") == "sparse"
        self.joint_routing = bool(joint_routing)
        self.rows_pt = 32 if self.topk_custom or self.joint_routing else 1  # (routing rows per page)
        # the latent down output untilized straight into this chip's rows of the dispatch buffer (layout_ops with a row
        # offset), dispatch_prep then only writes the route words: -18 us of own-row copy for +2.5 us of untilize
        self.lat_direct = self.sparse_dispatch and os.environ.get("K3_LAT_DIRECT", "1") == "1"
        self.dprep = (
            dispatch_prep_prepare(
                mesh_device,
                S=S,
                K=d.K,
                EPC=self.EPC,
                tokens_per_page=self.rows_pt,
                copy=not self.lat_direct,
                torus=self.torus,
                row_mates=self.shared_tp,
                mates_axis=self.shared_axis,
            )
            if self.sparse_dispatch
            else None
        )
        if self.lat_direct:  # (and both in one generic_op, the untilize on the cores the route words leave)
            C_ = tuple(mesh_device.shape)[1]
            self.untilize_x = layout_op_prepare(
                mesh_device, "untilize", row0_of=lambda r, c: (r * C_ + c) * S, exclude=self.dprep.cores
            )
            self._prep_merged = {}
        # the routing rows' all-gather: one sparse_all_gather_2d to every chip (default) or two ttnn.all_gather
        # (sparse: sparse_all_gather_2d to every chip, 157 us vs 140 for the two ttnn.all_gather at 64-B rows; fag: the
        # dense fabric all-gather, one snake)
        self.rgather = os.environ.get("K3_RGATHER", "fag")
        self.rgather_fabric = self.rgather == "sparse"
        if self.sparse_dispatch or self.rgather_fabric:
            ensure_semaphores(mesh_device, num_links=2, torus=self.torus)  # (persistent: before anything per call)
        if self.rgather_fabric:
            self.rcopy = own_rows_copy_prepare(mesh_device, S=S)
            self.route_all = all_chips_route_words(mesh_device, S)
        if not routed_only:  # (routed_only: built after the caller's dense ops have run once)
            self._warm_up()  # (the dense ops once first: what they keep in L1 across calls sits above the arena)
        else:  # (the routing gather once: its first call's allocations sit above the arena)
            z = lambda dt: ttnn.from_torch(  # noqa: E731
                torch.zeros(self.N * S, d.K, dtype=torch.float32),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=dram,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
            if self.joint_routing:  # (the router's 32-token pages)
                ids0, wts0 = (
                    ttnn.from_torch(
                        torch.zeros(self.N * S // 32, 32 * 2 * d.K, dtype=torch.int16),
                        dtype=ttnn.uint16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh_device,
                        memory_config=dram,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
                    ),
                    None,
                )
            else:
                ids0, wts0 = z(ttnn.uint16), z(ttnn.bfloat16)
            g0, _ = self.gather_routing(ids0, wts0)
            ttnn.synchronize_device(mesh_device)
            for t in (ids0, wts0):
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
            if g0 is not getattr(self, "_rg_out", None):
                ttnn.deallocate(g0)
        # (allocated before the arena: above it, not capping the L1 below it)
        # (the combine's start barrier words: persistent, never rewritten - cumulative counts across calls)
        gcores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self.bar = ttnn.from_torch(
            torch.zeros(1, 1, grid.x * grid.y, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(gcores, [1, 16], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        a_rows = int(plan["arena_tiles"]) * 32
        if (
            combine_auto_cfg
        ):  # (the arena takes the L1 left, less the flat program's static CBs: the combine sizes from it)
            from ttnn.bringup.combine_rf.sparse_all_gather_2d import static_cb_bytes
            from ttnn.bringup.combine_rf.sparse_combine_rf import fused_arena_reserve

            mv = ttnn.device.get_memory_view(mesh_device, ttnn.BufferType.L1)
            # (below the arena, over the call: the writers' words (allocated after it) and whichever program's CBs are
            # larger - the fused op's static CBs, or the dispatch's on its port cores plus its per-call manifest)
            flat_cbs = int(plan["static_l1_bytes"])
            disp = 0
            if self.sparse_dispatch:
                ms = manifest_spec(mesh_device, S, num_links=2, torus=self.torus)
                grp = 8 if 2 * d.latent <= 7168 else 2  # (dispatch()'s row group)
                disp = (int(ms[0][-1]) * 4 + 63) // 64 * 64 + static_cb_bytes(self.N, S, d.latent, group=grp)
            reserve = fused_arena_reserve(max(flat_cbs, disp), l1=True) + 1024
            if self.dispatch_in_program:  # (one program: the manifest above both kinds of static CBs)
                man_b = (int(ms[0][-1]) * 4 + 63) // 64 * 64
                stat = static_cb_bytes(self.N, S, d.latent, group=grp)
                reserve = fused_arena_reserve(max(flat_cbs, stat), l1=True) + man_b + 1024
            left = (int(mv.largest_contiguous_bytes_free_per_bank) - reserve) // 64 // 32 * 32
            assert a_rows <= left, (
                f"the flat op's arena {a_rows * 64 >> 10} KB > the L1 left {left * 64 >> 10} KB (less the dispatch's / "
                f"the flat op's static CBs, {reserve >> 10} KB): their circular buffers would clash with it"
            )
            a_rows = left
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        self._arena_spec = (
            ttnn.Shape([grid.x * grid.y * a_rows, 16]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(all_cores, [a_rows, 16], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        self.arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        self.arena_addr = int(self.arena.buffer_address())
        self.arena_bytes = a_rows * 64  # (a core)
        # the planner's outputs (allocated once) then the combine (its accumulator count sets the planner's A)
        luts, nps = port_luts(mesh_device, 2, self.torus)
        self.rows_cap = rows_cap
        chips = [(r, c) for r in range(R) for c in range(C)]
        self._planner_args = dict(
            S=S,
            K=d.K,
            EPC=EPC,
            L=2,
            port_luts=luts,
            n_ports=nps,
            rows_cap=rows_cap,
            cores=sorted(pcores, key=lambda c: (c[1], c[0])),
            row_pages=True,
            joint=True,
            tokens_per_page=self.rows_pt,
            torus=self.torus,
        )
        # deterministic combine: the planner rebuilds every chip's flat-expert schedule (the plan's schedule parameters,
        # pin 1; the pinning ratio as the flat program reads it) and the accumulators take their rows in that order
        self.deterministic = deterministic
        if deterministic:
            self._planner_args["det"] = dict(
                rps=lay["sched_rps"],
                max_e=lay["sched_max_e"],
                gu_nreg=lay["sched_gu_nreg"],
                pin=1,
                pin_ratio=str(self.op.layout.pin_ratio),
            )
        if self.shared_tp:
            # the shared expert's input: the row's C S tokens from the dispatch buffer as tiles, the chip's own block
            # first (block j = column (c + j) % C: the order of the partial's blocks the combine sends), and its
            # partial row-major (the combine's shared_rows; persistent: the combine bakes its address)
            # (the shared expert runs before the arena is allocated - the fused op's own barrier replaces the dispatch
            # as the "every chip zeroed" sync - so it is one call over the group's G S rows, block j first (j = 0:
            # own); its partial goes to y_se: the combine sends blocks 1.. to their owners, its accumulators add block 0)
            G, ax = self.G, self.shared_axis
            mate = lambda ch, j: ch[0] * C + (ch[1] + j) % C if ax == 1 else ((ch[0] + j) % R) * C + ch[1]  # noqa: E731
            ST = S // 32
            cw = 16  # (1 KB row pieces: the gather-tilize 72 -> 50 us at 1280 x 4096; 32: no better)
            self.se_tilize = layout_op_prepare(
                mesh_device, "tilize", cw=cw, src_tile_of=lambda ch, tr: mate(ch, tr // ST) * ST + tr % ST
            )
            self.se_untilize = layout_op_prepare(mesh_device, "untilize", cw=cw)
            # (the unified-op shared expert: G "experts", block j at its slot of the dispatch buffer; the output's blocks
            # untilized into y_se in the partial's order)
            self.se_untilize_g = layout_op_prepare(
                mesh_device, "untilize", cw=cw, src_tile_of=lambda ch, tr: mate(ch, tr // ST) * ST + tr % ST
            )
            u32 = lambda t: ttnn.from_torch(  # noqa: E731
                t,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=dram,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
            self.se_ureg = u32(
                torch.tensor(
                    [[mate((d // C, d % C), j) * S for j in range(G)] for d in range(R * C)], dtype=torch.int32
                )
            )
            self.se_ucnt = u32(torch.full((R * C, G), S, dtype=torch.int32))
            self.se_ugidx = u32(torch.arange(G, dtype=torch.int32).repeat(R * C))
            # (the flat-op shared expert: its rows by a token index into the dispatch buffer, the same order)
            tix = torch.zeros(R, C, 1, G * S, dtype=torch.int32)
            for r_ in range(R):
                for c_ in range(C):
                    tix[r_, c_, 0] = torch.tensor([mate((r_, c_), j) * S + t for j in range(G) for t in range(S)])
            self.se_tidx = ttnn.from_torch(
                tix,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=dram,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1)),
            )
            self.y_se = ttnn.allocate_tensor_on_device(
                ttnn.Shape([self.G * S, d.H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device, dram
            )
        probe = combine_planner_prepare(mesh_device, A=15, **self._planner_args)  # (outputs only: their addresses)
        copt = dict(n_acc=16, depth=combine_depth)
        if combine_auto_cfg:  # (the close-out sizing: from the free cells and the arena)
            from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan, place_ports
            from ttnn.bringup.combine_rf.sparse_combine_rf import combine_auto

            port_cells = combine_port_cells(mesh_device, self.cells, self.torus)
            piece = max(7168, 2 * d.latent)
            # (combine_accumulators: the count the caller planned with - the MoE block's fused head planner bakes it:
            # the ports-apart mode at exactly that many)
            pin = dict(a_max=combine_accumulators, mode="ports apart") if combine_accumulators else {}
            auto = combine_auto(
                self.N, S, d.latent, len(self.cells), len(port_cells), arena_bytes=a_rows * 64, l1=True, depth=8,
                piece_max=piece, **pin,
            )  # fmt: skip
            assert auto[
                "fits"
            ], f"combine: does not fit ({len(self.cells)} cells, {a_rows * 64 >> 10} KB): {auto['why']}"
            assert (
                not combine_accumulators or auto["n_acc"] == combine_accumulators
            ), f"combine: {auto['n_acc']} accumulators, {combine_accumulators} planned"
            acc_on_ports = auto["acc_on_ports"]
            copt = dict(
                n_acc=auto["n_acc"], depth=auto["depth"], inj_slots=auto["inj_slots"], local_keep=auto["local_keep"]
            )
            from loguru import logger

            lay_ = auto["layout"]
            logger.info(
                f"K3 block combine (auto): {len(self.cells)} cells, {len(port_cells)} port cells, accumulators "
                f"{auto['n_acc']} (on the ports: {acc_on_ports}), landing pool {lay_['PSL']} slots (ring depth {lay_['D']}), "
                f"injection ring {lay_['QP']}, local ring {lay_['QL']}; arena {a_rows * 64 >> 10} KB"
            )
        self.combine = sparse_combine_rf_prepare(
            mesh_device,
            S,
            d.latent,
            {ch: [] for ch in chips},
            num_links=2,
            torus=self.torus,
            **copt,
            cells=self.cells,
            arena=self.arena,
            writers=self.writers,
            l1=dict(writers=wcores, words=int(self.words.buffer_address())),
            devplan=dict(plan=probe.outs["plan"], table=probe.outs["table"], dord=probe.outs.get("dord"), top_k=d.K),
            deterministic=deterministic,
            snap_q=combine_snap,
            acc_on_ports=acc_on_ports,
            piece_max=max(7168, 2 * d.latent),  # (the L1 hand-off: one piece a row; a fabric payload >= row + 32 B)
            shared_rows=dict(tensor=self.y_se, axis=self.shared_axis) if self.shared_tp else None,
            # (always: a port forwards ring layouts / rows / credits into another chip's arena only once every chip's
            # combine started - after that chip's per-call zeroing. The dispatch's END markers do not order it: a home
            # at the far end of a 1x4 line finished its dispatch and forwarded its layouts to chip 0 before chip 0's
            # zeroing ran, which wiped them - chip 0's ports then waited forever, ~1 call in 100)
            barrier=dict(tensor=self.bar),
            dispatch=(
                dict(
                    sems=self.disp_sems,
                    relays=self.relays,
                    words=int(self.relay_words.buffer_address()),
                    ready_off=2032,
                    group=8 if 2 * d.latent <= 7168 else 2,  # (dispatch()'s row group)
                )
                if self.dispatch_in_program
                else None
            ),
        )
        assert not (self.dispatch_in_program and acc_on_ports), "dispatch_in_program: the combine's ports per link"
        self.acc_on_ports = acc_on_ports
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
            ex = [ttnn.CoreCoord(*c) for c in port_cores_all(mesh_device, num_links=2, torus=self.torus)]
            self.untilize_disp = layout_op_prepare(mesh_device, "untilize", exclude=ex)
        # per-call buffers (allocated once)
        sh = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        self.y = ttnn.from_torch(
            torch.zeros(self.N, rows_cap, d.latent, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=dram,
            mesh_mapper=sh,
        )
        self.out = ttnn.from_torch(
            torch.zeros(self.N, 1, S, d.latent, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=dram,
            mesh_mapper=sh,
        )
        self.desc = self.l1_args = None  # (built on the first call: its CBs bind that call's arena)
        self.coll = self.combine.collector
        zr = {
            c: [(self.arena_addr + o, b, st, n) for o, b, st, n in runs] for c, runs in self.combine.zero_runs.items()
        }
        for c in wcores:
            zr.setdefault(c, []).append((int(self.words.buffer_address()), 192 * 16 * 4, 0, 1))
        self.zero_desc = l1_zero_program(mesh_device, zr)
        # the arena's per-call zeroing folded into the dispatch (routed_only, the separate dispatch): its kernels on the
        # cells (the dispatch's port cores' runs by other cells), each counting in on the chip's dispatch ports, whose
        # senders send no END before every one did - the dispatch stays the barrier that orders the zeroing before any
        # chip's combine writes (MoE block: one program less a layer, ~5 us)
        self.zero_gate = None
        if routed_only and self.sparse_dispatch and not dispatch_in_program and os.environ.get("K3_ZGATE", "1") == "1":
            from ttnn.bringup.combine_rf.l1_zero import l1_zero_kernels
            from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan, place_ports

            ch_, _, _ = mesh_plan(mesh_device, 2, self.torus)
            dpc = place_ports(mesh_device, ch_)
            busy = {(v.x, v.y) for v in dpc.values()}
            pcrs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(busy)]
            )
            self._zg_sems = [ttnn.create_global_semaphore(mesh_device, pcrs, 0) for _ in range(2)]
            ctr, ep = (int(ttnn.get_global_semaphore_address(x)) for x in self._zg_sems)
            ports_of = lambda coord: sorted({(v.x, v.y) for (c_, _, _), v in dpc.items() if c_ == coord})  # noqa: E731
            fn, nz = l1_zero_kernels(mesh_device, zr, busy, ports_of, ctr)
            self.zero_gate = dict(key=("zero_gate", ctr, ep, nz, self.arena_addr), ctr=ctr, ep=ep, n=nz, kernels=fn)
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
            torch.zeros(self.N * self.S, self.d.H, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
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
                bs = ttnn.add(
                    logits, self.bias32, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=mc
                )
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
            logits,
            self.bias,
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=self.d.K,
            route_scale=self.d.route_scale,
            stable_sort=True,
            epsilon=1e-20,
            score_func="sigmoid",
            padding_config=None,
            weights_layout=ttnn.ROW_MAJOR_LAYOUT,
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
        elif ids.layout == ttnn.TILE_LAYOUT and wts.layout == ttnn.TILE_LAYOUT and wts.dtype == ttnn.bfloat16:
            # (one generic_op: ttnn's untilizes, bitcast and concat were four ops and ~0.3 ms of host)
            want = [int(ids.shape[0]), 2 * int(ids.shape[1])]
            both = getattr(self, "_rg_in", None) if self.routed_only else None
            if both is None or not both.is_allocated() or list(both.shape) != want:
                both = ttnn.allocate_tensor_on_device(
                    ttnn.Shape(want), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
                )
                if self.routed_only:  # (one input buffer: the gather's descriptor keys on its address)
                    self._rg_in = both
            if getattr(self, "_rpack", None) is None:
                self._rpack = routing_pack_prepare(self.mesh)
            self._rpack(ids, wts, both)
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
            if both is not getattr(self, "_rg_in", None):
                ttnn.deallocate(both)
            man = ttnn.allocate_tensor_on_device(*manifest_spec(self.mesh, S, num_links=2, torus=self.torus))
            sparse_all_gather_2d(buf, self.route_all, num_links=2, manifest=man, torus=self.torus)
            ttnn.deallocate(man)
            g = ttnn.reshape(buf, [N * S, 2 * K])
            return g, g
        if self.rgather == "fag":  # (one dense fabric all-gather, a snake over the whole mesh: chip order kept)
            extra = None
            if x_se is not None:  # (and the shared expert's row-major x, untilized under it)
                ua = self.untilize_gather.programs(x_se, x_rm)
                extra = ([x_se, x_rm], lambda coord: (list(ua[coord].cbs), list(ua[coord].kernels)))
            out = None
            if self.routed_only and extra is None:  # (one gathered-routing buffer: the planner's descriptor keys on it)
                out = getattr(self, "_rg_out", None)
                want = [self.N * int(both.shape[0]), int(both.shape[1])]
                if out is None or not out.is_allocated() or list(out.shape) != want:
                    out = self._rg_out = ttnn.allocate_tensor_on_device(
                        ttnn.Shape(want), both.dtype, both.layout, self.mesh, ttnn.DRAM_MEMORY_CONFIG
                    )
            g = fabric_all_gather(
                both, cluster_axis=None, topology=ttnn.Topology.Linear, num_links=1, dim=0, extra=extra, output=out
            )
            if both is not getattr(self, "_rg_in", None):
                ttnn.deallocate(both)
            return g, g
        a = ttnn.all_gather(both, dim=0, cluster_axis=1)
        if both is not getattr(self, "_rg_in", None):
            ttnn.deallocate(both)
        g = ttnn.all_gather(a, dim=0, cluster_axis=0)
        ttnn.deallocate(a)
        return g, g

    def plan(self, gi, gw, x):
        """The combine plan; with se_x_in_plan also the shared expert's row-major x ([S, H], DRAM), in one program."""
        if not (self.se_x_in_plan and self.se_mode == "flat"):
            return self.planner(gi, gw), None
        x_rm = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.S, self.d.H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )
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
            ttnn.freeze_program_hashes(md)  # (hash stored once: not re-hashed per call)
            self._plan_merged = {"key": k, "desc": md}
        prepared_generic_op([gi, gw, x, x_rm] + self.planner.tensors, self._plan_merged["desc"])
        return self.planner.outs, x_rm

    def latent_down(self, x):
        """[S, latent]: tiled (lat_direct: dispatch_prep untilizes it into the dispatch buffer), else row-major."""
        l1 = "latdown" in self.l1_set  # (the tiled result is transient: L1; the row-major one feeds the dispatch: DRAM)
        mc = ttnn.L1_MEMORY_CONFIG if l1 else None
        lat = (
            self.ld_run(x, self.w_ld2, mc) if self.lin2d else self._linear(x, self.w_ld, self.cfg_lat, memory_config=mc)
        )
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
        # (routed_only: one buffer pair for every call - the fused experts op's program cache is keyed on its buffer
        # addresses, so a per-call dispatch buffer that lands elsewhere for every layer rebuilt its program each call)
        bufs = getattr(self, "_dprep_bufs", None) if self.routed_only else None
        if bufs is None:
            bufs = (
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, N * S, W]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, dram
                ),
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, 1, route_words(N) * S]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, dram
                ),
            )
            if self.routed_only:
                self._dprep_bufs = bufs
        xbuf, route = bufs
        if self.lat_direct:  # (lat_rm is the tiled latent: own rows straight into the buffer; one program, both ops)
            k = tuple(int(t.buffer_address()) for t in (lat_rm, gi, xbuf, route))
            # (routed_only: every key's descriptor kept - the caller's x and routing land at a few addresses per layer;
            # else the last one, as before)
            cache = self.__dict__.setdefault("_prep_descs", {}) if self.routed_only else {}
            if self._prep_merged.get("key") == k:
                md = self._prep_merged["desc"]
            elif k in cache:
                md = cache[k]
            else:
                ua, da = self.untilize_x.programs(lat_rm, xbuf), self.dprep.programs(gi, None, xbuf, route)
                md = ttnn.MeshProgramDescriptor()
                for (r, c), pu in ua.items():
                    pd, pr = ttnn.ProgramDescriptor(), da[(r, c)]
                    pd.cbs = list(pu.cbs) + list(pr.cbs)
                    pd.kernels = list(pu.kernels) + list(pr.kernels)
                    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = pd
                ttnn.freeze_program_hashes(md)  # (hash stored once: not re-hashed per call)
                if self.routed_only and len(cache) < 512:
                    cache[k] = md
            self._prep_merged = {"key": k, "desc": md}
            prepared_generic_op([lat_rm, gi, xbuf, route], md)
        else:
            self.dprep(gi, lat_rm, xbuf, route)
        return xbuf, route

    def dispatch(self, lat_rm, prep=None, x_se=None, zero=None):
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
        man = ttnn.allocate_tensor_on_device(*manifest_spec(self.mesh, self.S, num_links=2, torus=self.torus))
        extra, x_rm = None, None
        if x_se is not None:  # (the shared expert's row-major x, untilized under the dispatch)
            x_rm = ttnn.allocate_tensor_on_device(
                ttnn.Shape([self.S, self.d.H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            ua = self.untilize_disp.programs(x_se, x_rm)
            extra = ([x_se, x_rm], lambda coord: (list(ua[coord].cbs), list(ua[coord].kernels)))
        # (rows wider than K3's 7 KB: a 2-row group, its CBs (2 x group rows) below the arena and the manifest: GLM's 8 KB
        # rows at 8 a group clashed with the arena in the layer test, at 4 with the manifest in the whole model)
        group = 8 if 2 * self.d.latent <= 7168 else 2
        sparse_all_gather_2d(
            xbuf, route, num_links=2, manifest=man, extra=extra, torus=self.torus, group=group, zero=zero
        )
        ttnn.deallocate(man)
        if not self.routed_only:
            ttnn.deallocate(route)
        g = ttnn.reshape(xbuf, [self.N * self.S, self.d.latent])
        return (g, x_rm) if x_se is not None else g

    def arena_fits(self):
        """Would this call's arena land where the kernels expect it (no L1 allocated since the block was built)?"""
        arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        ok = int(arena.buffer_address()) == self.arena_addr
        ttnn.deallocate(arena)
        return ok

    def take_arena(self):
        """The call's arena taken ahead (the MoE block's one L1 allocation a forward: its fused head runs in it first);
        routed_prepped then runs on it (zeroing the combine's state there, after the head) and frees it."""
        arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        assert (
            int(arena.buffer_address()) == self.arena_addr
        ), f"the arena landed at {int(arena.buffer_address())}, not {self.arena_addr}: an L1 tensor outlives its op"
        self._taken = arena
        return arena

    def alloc_arena(self, zero=True):
        """The call's arena (uninitialized) with the combine's state zeroed (zero False: the dispatch's zero gate does
        it - self.zero_gate); take_arena's if taken ahead."""
        arena, self._taken = getattr(self, "_taken", None), None
        if arena is None:
            arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        assert (
            int(arena.buffer_address()) == self.arena_addr
        ), f"the arena landed at {int(arena.buffer_address())}, not {self.arena_addr}: an L1 tensor outlives its op"
        if self.desc is None:
            self.desc, self.l1_args = self.combine.build_l1(self.y, self.out, arena, getattr(self, "_disp", None))
            self._desc_arena = arena
        elif (
            not self.routed_only
        ):  # (the CBs bind the arena's buffer: a program-cache miss rebuilds them from this one)
            self.desc = self.combine.rebind(arena)
        # (routed_only: rebound only when the experts op will build a program - _experts_combine)
        if zero:
            prepared_generic_op([arena, self.words], self.zero_desc)
        return arena

    def experts_combine(self, xg, pl, arena, op=None):
        # x read from DRAM once (x_once): unset - the layout's default (FlatLayoutOptions.x_once -1: on with the
        # snakes up to 320 rows an expert, one snake through both rectangles). (The multicast form it replaced was
        # off with gu64 - one multicaster cannot feed 64 gate/up cores - and routed_only: GLM-5.3-Flash 1.99 -> 1.70
        # ms fused at S 640 without it.) dw_batch: down-weight blocks per read barrier (2 by default; 4: noise)
        lay = dict(self.flat_layout, **self.layout_overrides)
        if self.x_once is not None:
            lay["x_once"] = int(bool(self.x_once))
        if self.dw_batch:
            lay["dw_batch"] = int(self.dw_batch)
        return self._experts_combine(xg, pl, arena, op or self.op, flat_layout(lay))

    def _experts_combine(self, xg, pl, arena, op, layout):
        extra = self.desc
        if self.routed_only:
            # the fused op's program-cache key is its buffer addresses (+ the plan): a key seen before is a hit, where the
            # combine's descriptor is not needed - pass an empty one (no copy of ~a whole program set per call: ~7 ms of
            # host); a new key gets the descriptor rebound to this call's arena (C++ fails loudly on a miss without it)
            a = lambda t: 0 if t is None else int(t.buffer_address())  # noqa: E731
            key = tuple(
                a(t)
                for t in (xg, pl["counts"], pl["regions"], op.gidx, op.w_gu, op.w_d, op.w_rd, op.done, arena, self.y)
            ) + (a(pl["token_index"]), op.m, op.act, self.x_bf16, self.h_bf16)
            seen = self.__dict__.setdefault("_seen_keys", set())
            if os.environ.get("FLAT_WP_KEYLOG"):
                from loguru import logger

                logger.info(f"K3 routed key {'seen' if key in seen else 'new'}: {key}")
            if key in seen:
                extra = self.__dict__.setdefault("_empty_desc", ttnn.MeshProgramDescriptor())
            else:
                if getattr(self, "_desc_arena", None) is not arena:
                    self.desc = self.combine.rebind(arena)
                    self._desc_arena = arena
                extra = self.desc
                seen.add(key)
        return ttnn.bringup.flat_routed_expert_with_program(
            xg,
            pl["counts"],
            pl["regions"],
            op.gidx,
            op.w_gu,
            op.w_d,
            reader_down_weights=op.w_rd,
            done_words=op.done,
            intermediate=self.d.I,
            max_tokens_per_expert=op.m,
            arena=arena,
            y=self.y,
            extra=extra,
            extra_key=self.key,
            collector=self.coll["core"],
            counts_addr=self.coll["counts_addr"],
            writers=self.writers,
            activation=op.act,
            pin=1,
            token_index=pl["token_index"],
            down_fp32=True,
            x_bf16=self.x_bf16,
            h_bf16=self.h_bf16,
            l1_args=self.l1_args,
            relay_words=self.relay_words,
            y_l1=True,
            layout=layout,
        )

    def latent_up(self, yr_l1=False):
        o = self.tilize_ln(ttnn.reshape(self.out, [self.S, self.d.latent]))  # (layout_ops: 19 us vs ttnn's 53)
        ns = ttnn.rms_norm(
            o,
            epsilon=self.d.eps,
            weight=self.g_ln,
            program_config=self.ln_pc,
            memory_config=self.ln_mc,
            compute_kernel_config=self.cfg_hi,
        )
        ttnn.deallocate(o)
        n = ttnn.sharded_to_interleaved(
            ns, ttnn.L1_MEMORY_CONFIG if "latup" in self.l1_set else ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(ns)
        mc = ttnn.L1_MEMORY_CONFIG if yr_l1 else None
        y = self.lu_run(n, self.w_lu2, mc) if self.lin2d else self._linear(n, self.w_lu, self.cfg_lat, memory_config=mc)
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
            mesh_device,
            "fake",
            m=C * S,
            H=d.H,
            I=It,
            gids=[[0] for _ in range(N)],
            n_global=1,
            wdtype="bf8",
            act="situ",
            pin=1,
            x_bf16=self.se_xbf16,
        )
        lay = dict(self.se.plan)
        per_col = []
        for c in range(C):
            sl = slice(c * It, (c + 1) * It)
            per_col.append(
                flat_weight_regions(
                    [(w.sh_gate[sl].T.contiguous(), w.sh_up[sl].T.contiguous(), w.sh_down[:, sl].T.contiguous())], lay
                )
            )
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
            torch.tensor([v], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.se_counts, self.se_regions, self.se_tidx = row([C * S]), row([0]), None

    def _build_shared_lin2d(self, mesh_device, w, d, S):
        """The shared expert as two combine_rf/lin2d calls on the own S tokens: gate / up with the SiTU-GLU fused
        (gate / up tile pairs, the activation a [S, I] packed), then down; in the block's forward the down also adds
        the latent up's output (y = yr + a @ Wd: no separate sum). 458 us standalone vs the flat SE's ~525 + sum."""
        self._se_knobs()
        assert (d.beta, d.linear_beta) == (4.0, 25.0), "lin2d's SiTU is Kimi's (sfpu::SituGluConfigKimi)"
        fid = [ttnn.MathFidelity.LoFi, None, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4][
            self.se_fid
        ]
        self.se_gu_run, lay = lin2d_prepare(mesh_device, M=S, K=d.H, N=2 * d.I_shared, fidelity=fid, glu=True)
        self.se_w_gu = lin2d_upload(mesh_device, glu_weight(w.sh_gate.T.contiguous(), w.sh_up.T.contiguous()), lay)
        self.se_d_run, lay = lin2d_prepare(mesh_device, M=S, K=d.I_shared, N=d.H, fidelity=fid)
        self.se_dr_run, _ = lin2d_prepare(mesh_device, M=S, K=d.I_shared, N=d.H, fidelity=fid, residual=True)
        self.se_w_d = lin2d_upload(mesh_device, w.sh_down.T.contiguous(), lay)

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
            mesh_device,
            "fake",
            m=S,
            H=d.H,
            I=Ip,
            gids=[list(range(P)) for _ in range(N)],
            n_global=P,
            wdtype="bf8",
            act="situ",
            pin=1,
            x_bf16=self.se_xbf16,
        )
        lay = dict(self.se.plan)
        sl = lambda i: slice(i * Ip, (i + 1) * Ip)  # noqa: E731
        regions, dregs, rregs = flat_weight_regions(
            [
                (w.sh_gate[sl(i)].T.contiguous(), w.sh_up[sl(i)].T.contiguous(), w.sh_down[:, sl(i)].T.contiguous())
                for i in range(P)
            ],
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
            torch.tensor([v], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.se_counts, self.se_regions = row([S] * P), row([i * S for i in range(P)])
        self.se_tidx = row(list(range(S)) * P)

    def shared(self, x, yr=None, parts_only=False, x_rm=None):
        """The shared expert's y [S, H]; with yr (local flat SE, bf16 tile parts) the block's y = yr + the parts' sum
        in one pass (sum_ops) instead of ttnn.sum + ttnn.add."""
        if self.se_mode == "tp":  # (row all-gather, one flat expert on the row's tokens, reduce-scatter)
            x_rm = self.untilize7(
                x
            )  # (a row as 7 pages of 2 KB: the fabric all-gather sends whole pages, the flat reads 2 KB segments)
            xg = fabric_all_gather(x_rm, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=2, dim=0)
            ttnn.deallocate(x_rm)
            y = self.se(
                xg,
                self.se_counts,
                self.se_regions,
                x_pages_per_row=7,
                down_fp32=True,
                pack_stochastic_rounding=self.se_srnd,
                y_bf16=self.se_ybf16,
                math_fidelity=self.se_fid,
            )
            ttnn.deallocate(xg)
            ys = ttnn.reduce_scatter(y, 0, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear)
            ttnn.deallocate(y)
            return ys
        if self.se_mode == "lin2d":  # (a in L1; with yr: the block's y = yr + a @ Wd, in DRAM)
            a = self.se_gu_run(x, self.se_w_gu, ttnn.L1_MEMORY_CONFIG)
            y = self.se_d_run(a, self.se_w_d) if yr is None else self.se_dr_run(a, self.se_w_d, None, yr)
            ttnn.deallocate(a)
            return y
        if self.se_flat:  # (x row major; y bfp8 tiles [P S, H]: the parts' outputs, summed)
            x_rm = self.untilize(x) if x_rm is None else x_rm  # (layout_ops: 32 us vs ttnn's 49; or the planner's)
            y = self.se(
                x_rm,
                self.se_counts,
                self.se_regions,
                token_index=self.se_tidx,
                down_fp32=True,
                y_row_major=self.se_yrm,
                pack_stochastic_rounding=self.se_srnd,
                y_bf16=self.se_ybf16,
                math_fidelity=self.se_fid,
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

    def routed(self, x, ids, wts, op, hook=None, shared=None, tilize=True):
        """routed_only: x [S, H] bf16 TILE (this chip's own tokens), ids uint16 / wts bf16 [S, K] (the caller's top-k;
        weights already normalized / scaled) and op (a FlatRoutedExpert of this block's plan: the layer's weights) ->
        the routed experts' sum [S, H] bf16 TILE (a fresh tensor). shared (shared_tp: the layer's TtSharedTpRow): plus
        the shared expert's output."""
        assert (shared is not None) == self.shared_tp, "shared_tp blocks take the layer's shared expert, others none"
        mark = hook or (lambda _: None)
        if os.environ.get("K3_HOST_LOG") == "1":  # (host time per stage, no syncs: dispatch cost)
            import time as _t

            st = self.__dict__.setdefault("_host", {"n": 0})
            last = [_t.perf_counter()]

            def mark(name, _m=mark):
                now = _t.perf_counter()
                st[name] = st.get(name, 0.0) + now - last[0]
                last[0] = now
                _m(name)
                if name == "tilize":
                    st["n"] += 1
                    if st["n"] % 42 == 0:
                        from loguru import logger

                        logger.info(
                            "K3 routed host ms per call: "
                            + " ".join(f"{k} {1e3 * v / st['n']:.2f}" for k, v in st.items() if k != "n")
                        )

        gi, _ = self.gather_routing(ids, wts)
        mark("routing_gather")
        pl = self.planner(gi, gi)
        mark("planner")
        prep = self.dispatch_prep(x, gi)
        mark("dispatch_prep")
        own_se = None
        if shared is None:
            arena = self.alloc_arena()
            mark("arena_zero")
            xg = self.dispatch(
                x, prep
            )  # (also the barrier: every chip zeroed its arena before any chip's experts start)
            mark("dispatch")
        else:
            # (the shared expert between the dispatch and the arena: the fused op starts with its own barrier)
            xg = self.dispatch(x, prep)
            mark("dispatch")
            GS = self.G * self.S
            if hasattr(shared, "partials") and hasattr(shared, "G"):  # (the unified op: TILE out at the slots)
                o = shared.partials(xg, self.se_ureg, self.se_ucnt, self.se_ugidx, self.S)
                self.se_untilize_g(o, self.y_se, rows=GS)
                ttnn.deallocate(o)
            elif hasattr(shared, "partials"):  # (the flat op: rows straight from the dispatch buffer into y_se)
                shared.partials(xg, ttnn.reshape(self.se_tidx, [1, GS]), self.y_se)
            else:
                xs = self.se_tilize(xg, rows=GS)
                o = shared(ttnn.reshape(xs, [1, 1, GS, self.d.H]))
                ttnn.deallocate(xs)
                self.se_untilize(ttnn.reshape(o, [GS, self.d.H]), self.y_se)
                ttnn.deallocate(o)
            mark("shared")
            arena = self.alloc_arena()
            mark("arena_zero")
        n0 = self.mesh.num_program_cache_entries() if os.environ.get("K3_CACHE_LOG") == "1" else None
        self.experts_combine(xg, pl, arena, op)
        if n0 is not None and self.mesh.num_program_cache_entries() > n0:
            from loguru import logger

            logger.info(f"K3 routed: experts program cache miss ({self.mesh.num_program_cache_entries()} entries)")
        ttnn.deallocate(arena)
        if gi is not getattr(self, "_rg_out", None):
            ttnn.deallocate(gi)
        mark("experts_combine")
        if not tilize:  # (the combine's row-major output itself: the caller's fused tail tilizes it)
            assert own_se is None
            return self.out
        y = self.tilize(ttnn.reshape(self.out, [self.S, self.d.H]))
        mark("tilize")
        if own_se is not None:
            y2 = ttnn.add(y, own_se)
            ttnn.deallocate(y)
            ttnn.deallocate(own_se)
            y = y2
        return y

    def routed_prepped(self, prep, op, hook=None, tilize=True):
        """routed_only, with the routing gather, the planner and the dispatch prep done upstream (the MoE block's fused
        head): the plan already in self.planner.outs, prep = (the dispatch buffer [1, 1, N S, H] with this chip's own
        rows, the route words) -> the routed experts' sum [S, H] bf16 TILE (a fresh tensor); tilize False: the
        combine's row-major output itself ([1, 1, S, H], persistent - the MoE block's fused tail tilizes it)."""
        assert self.routed_only and self.sparse_dispatch
        mark = hook or (lambda _: None)
        if self.dispatch_in_program:
            return self._routed_in_program(prep, op, mark, tilize)
        # (the arena's zeroing in the dispatch's program when it can: its END markers wait for it - still the barrier
        # that orders every chip's zeroing before any chip's combine writes)
        zg = self.zero_gate
        arena = self.alloc_arena(zero=zg is None)
        if zg is None:
            mark("arena_zero")
        xg = self.dispatch(None, prep, zero=zg)  # (also the barrier: every chip zeroed its arena before any experts)
        mark("dispatch")
        self.experts_combine(xg, self.planner.outs, arena, op)
        ttnn.deallocate(arena)
        mark("experts_combine")
        if not tilize:
            return self.out
        y = self.tilize(ttnn.reshape(self.out, [self.S, self.d.H]))
        mark("tilize")
        return y

    def _routed_in_program(self, prep, op, mark, tilize=True):
        """routed_prepped with the dispatch inside the experts' program (dispatch_in_program): the arena zeroed (its own
        program: the dispatch's END barrier then also covers every chip's zeroing), the manifest for the call, one
        program for the dispatch + experts + combine, the tilize."""
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import manifest_layout, mesh_plan

        xbuf, route = prep
        R, C = self.R, self.C
        _, _, (wx, wy) = mesh_plan(self.mesh, 2, self.torus)
        words = manifest_layout(R, C, wx, wy, self.S)[2] * 4
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in self.disp_cores])
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
        )
        dkey = (int(xbuf.buffer_address()), int(route.buffer_address()))
        if getattr(self, "_disp_key", None) != dkey:  # (the dispatch's buffers are in the port kernels' args)
            self.desc, self.__dict__["_seen_keys"] = None, set()
            self._disp_key = dkey
        # (the arena first, then the manifest below it - the reserve counts both; the descriptor built with them)
        self._disp = None
        if self.desc is None:
            arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
            man = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, len(self.disp_cores), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, mc
            )
            self._man_addr = int(man.buffer_address())
            ttnn.deallocate(man)
            ttnn.deallocate(arena)
        arena = self.alloc_arena_disp(dict(output=xbuf, plan=route), mc, words)
        mark("arena_zero")
        xg = ttnn.reshape(xbuf, [self.N * self.S, self.d.latent])
        self.experts_combine(xg, self.planner.outs, arena, op)
        ttnn.deallocate(self._man)
        ttnn.deallocate(arena)
        mark("dispatch_experts_combine")
        if not tilize:
            return self.out
        y = self.tilize(ttnn.reshape(self.out, [self.S, self.d.H]))
        mark("tilize")
        return y

    def alloc_arena_disp(self, disp, mc, words):
        """alloc_arena for dispatch_in_program: the manifest after the arena, at the address the descriptor baked."""
        arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        assert int(arena.buffer_address()) == self.arena_addr, "the arena moved"
        self._man = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, len(self.disp_cores), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh, mc
        )
        assert int(self._man.buffer_address()) == self._man_addr, "the manifest moved"
        if self.desc is None:
            self.desc, self.l1_args = self.combine.build_l1(self.y, self.out, arena, dict(disp, manifest=self._man))
            self._desc_arena = arena
        prepared_generic_op([arena, self.words], self.zero_desc)
        return arena

    def __call__(self, x, keep=False, hook=None):
        """x: [S, H] bf16 TILE per chip. Returns y [S, H] (and the intermediates with keep=True). hook(stage name),
        if given, is called after every stage (e.g. to read the profiler per stage)."""
        mark = hook or (lambda _: None)
        ids, wts, *logits = self.router(x, keep_logits=keep)
        mark("router")
        x_rm = None
        if self.se_x_where == "gather" and self.rgather == "fag" and self.se_mode == "flat":
            x_rm = ttnn.allocate_tensor_on_device(
                ttnn.Shape([self.S, self.d.H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
            )
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
            xg = self.dispatch(
                lat, prep
            )  # (also the barrier: every chip zeroed its arena before any chip's experts start)
        mark("dispatch")
        self.experts_combine(xg, pl, arena)
        ttnn.deallocate(arena)
        mark("experts_combine")
        if self.se_mode == "lin2d" and not keep:  # (latent up, then the shared expert's down adds it: no sum op)
            yr = self.latent_up(yr_l1="yr" in self.l1_set)
            mark("lat_up")
            y, ys = self.shared(x, yr=yr), None
            mark("shared")
        elif (
            self.fuse_sum and not keep
        ):  # (shared expert parts, latent up, then one sum; keep: ys separately, for checks)
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
        inter = (
            dict(ids=ids, wts=wts, gi=gi, gw=gw, lat=lat, xg=xg, yr=yr, ys=ys, logits=logits[0], routed=self.out, pl=pl)
            if keep
            else None
        )
        if not keep:
            for t in (ids, wts, gi, lat, xg, yr, ys):  # (gw is gi)
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
        return (y, inter) if keep else y
