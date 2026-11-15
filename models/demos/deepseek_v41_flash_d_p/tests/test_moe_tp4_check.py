# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.58 (owner 2026-11-14, a go/no-go CHECK, no model change): would tensor-parallel routed experts (TP4: a mesh ROW of
4 chips a group, 8 groups of 48 experts, each expert split 4 ways along its FFN width I = 2304; TP8 (OV-41): a mesh
COLUMN of 8 chips, 4 groups of 96 experts split 8 ways) beat expert parallelism (EP=32, today) and EP + P.56's replicas?

Terms (tests/moe_tp_model.py has the CPU model's): chip k = r C + c; EP expert e on chip e // 12, slot e % 12.
  experts phase  = the routed_experts_combine program (the flat routed experts + the combine, one program a chip),
                   slowest chip, summed over the 40 MoE layers of a chunk (as P.55a / P.56 report it);
  dispatch       = the sparse_dispatch_2d program (a token's row to every chip that computes one of its experts);
  flat           = the standalone flat_routed_expert op (the experts' compute without the combine) on given counts;
  combine cost   = experts phase - flat on the same routing (what the combine adds to the compute);
  added          = TP's value - EP's value at the same routing (dispatch, combine), per layer at 5,120 tokens;
  reduce         = a separate TP sum: none, the partial rows are summed in the combine's fp32 accumulators (0 ms).

How TP is emulated with the block ops as they are (fork ttnn.bringup.moe_block_main, unchanged): each 1/g slice of an
expert is an expert of its own, placed so the g slices of expert e sit on the g chips of e's group, and the router
gives them the same row and bias (equal scores: a token picks every slice of its experts, each gate weight w_e / g).
The fused head then builds the TP route masks itself (every chip of the token's groups), the dispatch sends the rows
there, the flat experts compute the slices and the combine sends each (token, slice) partial row home and sums it in
fp32. Three limits of the ops shape the emulation (each is a change a real TP build needs):
  * the flat op splits one expert's width evenly over its gate/up cores (2 I / 32 tiles into a multiple of 16):
    576 / 288 cannot run; the slices run at 512 / 256 and their compute is scaled by 576 / 512 (x 1.125);
  * the fused head takes top-K <= 16 (even): TP4 runs at 2 / 3 / 4 experts a token (K 8 / 12 / 16), TP8 at 1 / 2
    (K 8 / 16); the combine cost is fitted linear in K and the dispatch linear in the chips a row goes to, then taken
    at the real 6 experts (K 24 / 48; the chips from the real layer's top-6 on the host);
  * the head's routing planner takes N x slots <= 1024: the head-driven runs use 32 slots a chip (TP4: 256 experts
    x 4, TP8: 128 x 8; the real layer's group loads, moe_tp_model / emul_target); the compute runs use the true 48
    slices a chip (TP8's 96 as two calls of 48: an upper bound).
Weights: seeded random, one chip's slots repeated on every chip (12 distinct quarter sets). Real routing: the router's
bias steered (CPU, moe_tp_model.steer_bias) so the experts' counts follow layer LAYER of chunk 10 (P.55a's counts,
novel and code); uniform: a near-zero bias.

Steps:
  1 CPU model: P.55a's and P.45's counts (every layer, chunks 0-10, both prompts), EP / TP4 / TP8 with P.55a's fit
    scaled (A / g, B / g) and with this run's fits of the flat op at the slice widths.
  2 device: (a) calibration sweeps of the flat op per layout (A: active slots at 32 tokens; B: every slot at more
    rows); (b) the flat op at every layer's real chunk-10 counts, both prompts; (c) head-driven routings: dispatch,
    experts phase, the flat op on the same counts; (d) accuracy at the uniform routing vs fp32.
  3 verdict: per chunk at 51,200, EP off (P.56 measured) vs EP + P.56 (measured) vs TP4 / TP8 (= EP off + their
    experts-phase difference + 40 x dispatch added).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_tp4_check.py
"""

from __future__ import annotations

import glob
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.tests import moe_tp_model as M

S = spec()
LAYER = 20  # the real routing's layer (P.55b's)
LEAD = 40  # matmuls (4096^2, 11 x 10 cores: rule 18) ahead of each timed window (DS41-248)
REPS = 5  # timed calls a routing
REPS_F = 3  # timed flat calls a layer
SAMPLE = 16  # tokens a chip in the accuracy check
NQ = 12  # distinct weight sets
SPC_E = 32  # TP emulation's slots a chip in the head-driven runs (the head's routing planner: N x slots <= 1024)
SPC_C = 48  # TP slices a chip in the compute runs (TP4: 48; TP8: 96 = two calls of 48; the flat op takes <= 64)
LAYERS_N = 40
P45_DIR = "/scratch/dnijemcevic/replication/counts"
# P.56 FINAL (BREADCRUMBS, gate unit P.56#test_moe_prev_replicas): chunks 1..10 mean, device ms a chunk
P56 = {
    "novel": {"device_off": 175.15, "device_prev": 156.10, "experts_off": 54.74, "experts_prev": 44.15},
    "code": {"device_off": 164.00, "device_prev": 153.08, "experts_off": 48.15, "experts_prev": 41.98},
}
K_COMBINE = "routed_experts_combine/device/kernels"
K_DISPATCH = "sparse_dispatch_2d/device/kernels"
K_FLAT = "flat_routed_expert/device/kernels"
MESH = pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
BOX = pytest.mark.parametrize("device_params", [harness.device_params(S)], indirect=True, ids=["box"])


def _p55a() -> dict:
    name = "P.55a_exact_bound.json"
    R, C = S.mesh
    for d in (
        metrics.results_dir(),
        Path(__file__).parents[1] / "bringup" / "ledgers" / f"galaxy_{R}x{C}-full" / "results",
    ):
        if (d / name).exists():
            return json.loads((d / name).read_text())
    raise FileNotFoundError(f"{name}: P.55a's record")


def _own(row) -> np.ndarray:
    return np.array([[int(v) for v in s.split()] for s in row["tokens_slot"]])


def _p45(wl: str) -> dict:
    """P.45's per-chunk count files {chunk: {layer: own [chips, 12]}} (empty when the files are not on this host)."""
    out = {}
    for f in sorted(glob.glob(f"{P45_DIR}/{wl}_chunk*.json")):
        r = json.loads(Path(f).read_text())
        spc = int(r["slots_per_chip"])
        out[int(r["chunk"])] = {
            int(L): np.array([row[k * spc : (k + 1) * spc] for k, row in enumerate(v["chips"])])
            for L, v in r["layers"].items()
        }
    return out


# ---------------------------------------------------------------------------------------------------- configurations
def flat_split_ok(width: int) -> bool:
    """The flat op's gate/up split rule (flat_routed_expert_plan.cpp: the 2 It weight tiles a K row over at most 64
    cores, a whole number of tiles or 16-column units a core, the core count a multiple of 16)."""
    It = width // 32
    tiles = 2 * It
    return width % 32 == 0 and any(
        tiles % per == 0 and (tiles // per) % 16 == 0 for per in range(max(1, (tiles + 63) // 64), 9)
    )


def flat_width(width: int) -> int:
    """The widest slice <= width the flat op can split (DeepSeek-V4.1's 576 / 288 cannot: 512 / 256)."""
    w = width - width % 32
    while w > 0 and not flat_split_ok(w):
        w -= 32
    return w


class Cfg:
    """One emulated layout. tp: 1 (EP), 4 (TP4, mesh rows), 8 (TP8 virtual: mesh columns, 192 virtual experts).
    n_e: experts a token; spc: slots a chip; E: slots; K: slots a token (n_e x slices: the head takes K <= 16);
    I: a slot's width (the flat op's, true_I / scale: the model's); NX: experts the router ranks (slot s copies
    expert xo[s]'s router row and bias)."""

    def __init__(self, name, tp, shape, n_exp, n_e, inter, spc=None):
        R, C = shape
        N = R * C
        self.name, self.tp, self.shape, self.N, self.n_e = name, tp, shape, N, n_e
        if tp == 1:
            self.spc, self.I, self.K, self.NX = n_exp // N, inter, n_e, n_exp
        elif tp == C:  # (a row: chip (r, c) holds slice c of the row's spc experts)
            self.spc, self.I, self.K = spc, flat_width(inter // C), n_e * C
            self.NX = spc * R
        else:  # tp == R (a column: chip (r, c) holds slice r of the column's spc experts)
            self.spc, self.I, self.K = spc, flat_width(inter // R), n_e * R
            self.NX = spc * C
        self.true_I = inter if tp == 1 else inter // tp
        self.scale = self.true_I / self.I  # (the flat op's compute at the true slice width: x scale)
        self.E = self.spc * N
        xo = np.zeros(self.E, dtype=np.int64)
        for k in range(N):
            r, c = divmod(k, C)
            for j in range(self.spc):
                if tp == 1:
                    xo[k * self.spc + j] = k * self.spc + j
                elif tp == C:
                    xo[k * self.spc + j] = r * self.spc + j
                else:
                    xo[k * self.spc + j] = c * self.spc + j
        self.xo = xo  # slot -> expert (router row)

    def wset(self, local: int) -> int:
        return local % NQ


def _expert_mats(Q, cfg: Cfg, l: int):
    """Slot-local l's (Wg [H, I], Wu [H, I], Wd [I, H]) for cfg (fp32)."""
    g, u, d = Q[cfg.wset(l)]
    if cfg.tp == 1:
        return torch.cat([g] * 4, 1), torch.cat([u] * 4, 1), torch.cat([d] * 4, 0)
    if cfg.I == g.shape[1]:
        return g, u, d
    w = cfg.I
    return g[:, :w].contiguous(), u[:, :w].contiguous(), d[:w].contiguous()


# ---------------------------------------------------------------------------------------------------- the test
@BOX
@MESH
@pytest.mark.timeout(5400)
def test_moe_tp4_check(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt import cores
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings
    from models.demos.moe_block_main.tt.flat_gather import flat_weights_module
    from models.demos.moe_block_main.tt.moe_block import MoEBlock
    from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

    t_start = time.time()
    mb, fw = ad._mb(), flat_weights_module()
    shape = tuple(mesh_device.shape)
    R, C = shape
    N = R * C
    cores.init_caps(mesh_device)
    ad._apply_core_cap(mesh_device)  # (rule 18: every block program inside 11 x 10 on the Galaxy)
    mcfg = hooks._config(S)
    chunk = int(S.get("target.chunk"))
    H, I, NE, TOPK = int(mcfg.dim), int(mcfg.inter), int(mcfg.n_experts), int(mcfg.topk_experts)
    assert (NE, TOPK, N) == (384, 6, 32) and chunk == 5120, (NE, TOPK, N, chunk)
    ep_dims, act_name = ad.block_dims(mcfg)
    act = fw.ACTS[act_name]
    split = settings.get("COMBINE_SPLIT") == "on"
    fails: list[str] = []
    rec: dict = {"shape": list(shape), "chunk": chunk, "layer": LAYER, "cap": str(mb.max_grid())}

    def check(ok: bool, what: str) -> None:
        logger.info(f"TP {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(what)

    # ------------------------------------------------------------------ STEP 1 (CPU): P.55a's fit, scaled
    p55 = _p55a()
    A0, B0, C00 = p55["fit"]["A_ms"], p55["fit"]["B_ms"], p55["fit"]["C0_ms"]
    real = {wl: {int(r["layer"]): _own(r) for r in p55["layers"][f"{wl}_p51200"]} for wl in ("novel", "code")}
    scaled = {"ep": (1, A0, B0, C00), "tp4": (C, A0 / C, B0 / C, C00), "tp8": (R, A0 / R, B0 / R, C00)}
    cpu: dict = {}
    for wl in ("novel", "code", "randids"):
        lay = real[wl] if wl in real else {int(r["layer"]): _own(r) for r in p55["layers"][f"{wl}_p51200"]}
        t = M.chunk_table(lay, shape, scaled)
        cpu[wl] = {k: {"ms": round(v["ms"], 2), "busy_over_mean": round(v["busy_over_mean"], 3)} for k, v in t.items()}
        cpu[wl]["padding"] = {
            str(tp): round(float(np.mean([M.padding_share(o, shape, tp) for o in lay.values()])), 4) for tp in (1, C, R)
        }
        logger.info(f"TP CPU scaled chunk 10 {wl}: {cpu[wl]}")
    p45 = {wl: _p45(wl) for wl in ("novel", "code")}
    rec["cpu_scaled"] = cpu

    # ------------------------------------------------------------------ host weights (12 quarter sets)
    g = torch.Generator().manual_seed(58)
    Iq = I // C
    Q = [
        (torch.randn(H, Iq, generator=g) * H**-0.5, torch.randn(H, Iq, generator=g) * H**-0.5,
         torch.randn(Iq, H, generator=g) * I**-0.5)
        for _ in range(NQ)
    ]  # fmt: skip
    x_host = torch.randn(1, 1, chunk, H, generator=g).to(torch.bfloat16)
    W_router = torch.randn(NE, H, generator=g) * H**-0.5
    W_virtual = torch.randn(SPC_E * C, H, generator=g) * H**-0.5
    Is = int(ep_dims.shared_expert_hidden_dim)
    shared = [torch.randn(Is, H, generator=g) * H**-0.5, torch.randn(Is, H, generator=g) * H**-0.5,
              torch.randn(H, Is, generator=g) * Is**-0.5]  # fmt: skip
    bias_uniform = torch.randn(NE, generator=g) * 0.01
    bias_virtual = torch.randn(SPC_E * C, generator=g) * 0.01
    # (router scores on the host, as the head computes them (bf16 x, bf16 router), for the bias steering)
    xs = x_host.reshape(chunk, H).float()
    scores = M.sqrtsoftplus((xs @ W_router.to(torch.bfloat16).float().T).numpy()).astype(np.float32)
    steer_cache: dict = {}

    def emul_target(own: np.ndarray, spc: int) -> np.ndarray:
        """A mesh row's 48 real expert counts -> its spc emulated experts (the row's spc largest, scaled to the row's
        total: the group loads of the real layer)."""
        out = []
        for r in range(R):
            row = np.sort(own[r * C : (r + 1) * C].reshape(-1))[::-1].astype(float)
            top = row[:spc]
            out.append(top * (row.sum() / max(top.sum(), 1.0)))
        return np.concatenate(out)

    def steer(wl: str, n_e: int, nx: int = NE) -> tuple:
        """(bias [nx], the host's top-n_e expert ids [T, n_e]) making n_e experts a token of the first nx router rows
        follow layer LAYER's real counts (wl; nx < 384: the TP4 emulation's experts, emul_target); "uniform": the
        near-zero bias."""
        key = (wl, n_e, nx)
        if key not in steer_cache:
            sc = scores[:, :nx]
            if wl == "uniform":
                b = bias_uniform.numpy()[:nx]
            else:
                own = real[wl][LAYER]
                tgt = own.reshape(-1).astype(float) if nx == NE else emul_target(own, nx // R)
                b, cnt = M.steer_bias(sc, tgt, n_e)
                logger.info(
                    f"TP steer {wl} L{LAYER} {n_e} of {nx} a token: |count - target| / total "
                    f"{np.abs(cnt - tgt * cnt.sum() / tgt.sum()).sum() / cnt.sum():.3f}"
                )
            top = np.argpartition(-(sc + b), n_e - 1, axis=1)[:, :n_e]
            steer_cache[key] = (torch.from_numpy(np.asarray(b, dtype=np.float32)), top)
        return steer_cache[key]

    def dests(top: np.ndarray, tp: int) -> float:
        """Mean chips a token's row goes to (its own chip excluded; token t on chip t // (chunk / N)): EP the experts'
        chips, TP every chip of the experts' groups."""
        grp = M.groups(shape, tp)
        g_of = {k: gi for gi, gg in enumerate(grp) for k in gg}
        tot = 0
        for t, row in enumerate(top):
            chips = set()
            for e in row.tolist():
                home = int(e) // M.EPC
                chips.update(grp[g_of[home]] if tp > 1 else [home])
            chips.discard(t // (chunk // N))
            tot += len(chips)
        return tot / len(top)

    lead = cores.from_torch(
        torch.randn(1, 1, 4096, 4096, generator=g) * 0.01, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )  # fmt: skip

    def window(run) -> dict:
        """run's programs on the real-time profiler after LEAD matmuls: {"dispatch"|"combine"|"flat"|"other": [us]}."""

        def go():
            for _ in range(LEAD):
                ttnn.deallocate(ttnn.matmul(lead, lead, core_grid=ttnn.CoreGrid(y=10, x=11)))
            run()

        _, per = profile_realtime_program_merged(mesh_device, go)
        out = {"dispatch": [], "combine": [], "flat": [], "other": []}
        for e in per.values():
            srcs = [s.replace("\\", "/") for s in e["kernel_sources"]]
            if any(K_COMBINE in s for s in srcs):
                k = "combine"
            elif any(K_DISPATCH in s for s in srcs):
                k = "dispatch"
            elif any(K_FLAT in s for s in srcs):
                k = "flat"
            else:
                k = "other"
            out[k].append(e["duration_ns"] / 1e6)
        return out

    def up(t, dtype, mapper=None, layout=ttnn.ROW_MAJOR_LAYOUT):
        return cores.from_torch(t, dtype=dtype, layout=layout, device=mesh_device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)  # fmt: skip

    # (layout name, tp, experts a token per run): EP at the model's 6; TP at what the head takes (K = n_e x slices <= 16)
    plan = [("ep", 1, [TOPK]), (f"tp{C}", C, [2, 3, 4]), (f"tp{R}v", R, [1, 2])]
    dev: dict = {}
    accuracy: dict = {}
    experts_cache: dict = {}  # (layout name, layout options, m) -> FlatRoutedExpert
    flat_x = {"t": None, "rows": 0}

    for name, tp, nes in plan:
        dev[name] = d = {"runs": {}}
        for n_e in nes:
            t0 = time.time()
            cf = Cfg(name, tp, shape, NE, n_e, I, spc=SPC_E)
            d.update(E=cf.E, spc=cf.spc, I=cf.I, true_I=cf.true_I, scale=cf.scale)
            dims = mb.MoEModelDims(
                num_experts=cf.E, top_k=cf.K, emb_dim=H, routed_expert_hidden_dim=cf.I,
                routed_expert_activation=ep_dims.routed_expert_activation, router=ep_dims.router,
                shared_expert_hidden_dim=Is, shared_expert_activation=ep_dims.shared_expert_activation,
            )  # fmt: skip
            bcfg = mb.MoEBlockConfig(chunk_tokens=chunk, num_links=ad.LINKS, combine_snap=ad.COMBINE_SNAP,
                                     combine_snap_split=split, persist_l1_small=True)  # fmt: skip
            dtypes = ad.block_dtypes()
            res = mb.MoEBlockResources(mesh_device, dims, bcfg, dtypes)
            blk = MoEBlock(mesh_device, res, dtypes=dtypes, activation=act, max_tokens_per_expert=max(chunk, 256),
                           layout=mb.FlatLayoutOptions(weights_nk=True))  # fmt: skip
            one_arena = blk._one_arena_fits()
            kw = blk.expert_kw
            ekey = (name, repr(kw["layout"]), int(kw["max_tokens_per_expert"]))
            if ekey not in experts_cache:
                for old in list(experts_cache):
                    o = experts_cache.pop(old)
                    ttnn.deallocate(o.w_gu)
                    ttnn.deallocate(o.w_d)
                local = [tuple(m.T.contiguous() for m in _expert_mats(Q, cf, l)) for l in range(cf.spc)]
                experts_cache[ekey] = fw.FlatRoutedExpert(
                    mesh_device, [local] * N, m=kw["max_tokens_per_expert"], H=H, I=cf.I,
                    gids=[list(range(k * cf.spc, (k + 1) * cf.spc)) for k in range(N)], n_global=cf.E,
                    act=act_name, pin=1, layout=kw["layout"], keep_packed=True,
                )  # fmt: skip
                del local
            ex = experts_cache[ekey]
            x = up(x_host, ttnn.bfloat16, ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(2, 3)),
                   ttnn.TILE_LAYOUT)  # fmt: skip
            Wr = W_router[: cf.NX] if tp != R else W_virtual
            f32 = lambda t: ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731  (host tensors: no device program)
            head_in = (f32(shared[0]), f32(shared[1]), f32(shared[2]), None, f32(Wr[torch.from_numpy(cf.xo)]))
            logger.info(f"TP {name} n_e {n_e}: E {cf.E} ({cf.spc} a chip), K {cf.K}, I {cf.I} (true {cf.true_I}), "
                        f"one arena {one_arena}, built in {time.time() - t0:.1f} s")  # fmt: skip
            state: dict = {}

            def route(bias_x):
                """The fused head + dispatch for a router bias over the experts (copied to their slots)."""
                hw = mb.prepare_fused_head_weights(*head_in, f32(bias_x[torch.from_numpy(cf.xo)]), resources=res,
                                                   dtypes=dtypes, accumulators=blk.accumulators)  # fmt: skip
                arena = ttnn.allocate_tensor_on_device(blk.arena_spec, mesh_device)
                x_full = (ttnn.all_gather(x, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                          if blk.x_dram else None)  # fmt: skip
                extra = {"xq": blk.to_xq(x)} if blk.bf8_x else {}
                head = mb.fused_head(x, hw, resources=res, dtypes=dtypes, accumulators=blk.accumulators,
                                     x_full=x_full, l1_free=blk.l1_free, arena=arena, **extra)  # fmt: skip
                if x_full is not None:
                    ttnn.deallocate(x_full)
                spec_ = blk.arena_spec
                if not one_arena:
                    ttnn.deallocate(arena)
                    spec_ = mb.routed_experts_combine_arena_spec(res, rows_cap=int(head.table.shape[-2]) * 32, **kw)
                    arena = ttnn.allocate_tensor_on_device(spec_, mesh_device)
                xg = mb.sparse_dispatch_2d(head.xbuf, head.route, res, arena=arena)
                ttnn.deallocate(arena)
                ttnn.synchronize_device(mesh_device)
                state.update(head=head, xg=xg, spec=spec_, hw=hw)

            def call(dispatch=True):
                h = state["head"]
                arena = ttnn.allocate_tensor_on_device(state["spec"], mesh_device)
                if dispatch:
                    state["xg"] = mb.sparse_dispatch_2d(h.xbuf, h.route, res, arena=arena)
                y = mb.routed_experts_combine(state["xg"], h.counts, h.regions, h.token_index, h.table, h.plan,
                                              ex.gidx, ex.w_gu, ex.w_d, res, arena=arena, **kw)  # fmt: skip
                ttnn.deallocate(arena)
                return y

            def routing_host():
                K, Sc = cf.K, chunk // N
                rows = torch.cat([ttnn.to_torch(t).reshape(-1, 2 * K).to(torch.int64)[:Sc]
                                  for t in ttnn.get_device_tensors(state["head"].rows)])  # fmt: skip
                ids = rows[:, :K] & 0xFFFF
                ws = ((rows[:, K:] & 0xFFFF) << 16).to(torch.int32).view(torch.float32)
                xin = torch.cat([ttnn.to_torch(t).reshape(-1, H)[me * Sc : (me + 1) * Sc].float()
                                 for me, t in enumerate(ttnn.get_device_tensors(state["xg"]))])  # fmt: skip
                return ids, ws, xin

            def flat_inputs(cnt, spc):  # cnt [N, spc] -> (counts [N, E], regions [N, E], rows needed)
                c_ = torch.zeros(N, spc * N, dtype=torch.int32)
                r_ = torch.zeros(N, spc * N, dtype=torch.int32)
                need = 0
                for k in range(N):
                    off = 0
                    for j in range(spc):
                        nn_ = int(cnt[k, j])
                        c_[k, k * spc + j], r_[k, k * spc + j] = nn_, off
                        off += -(-nn_ // 32) * 32
                    need = max(need, off)
                return c_, r_, need

            def flat_run(cnts: list, exo=None, reps=REPS_F) -> list:
                """The flat op alone (exo, default the head-driven layout's) on each per-chip counts set [N, slots]
                (reps calls each, one window): median ms."""
                exo = ex if exo is None else exo
                hostin = [flat_inputs(cn, int(cn.shape[1])) for cn in cnts]
                rows = max(h[2] for h in hostin) + 64
                if rows > flat_x["rows"]:
                    if flat_x["t"] is not None:
                        ttnn.deallocate(flat_x["t"])
                    rows = max(rows, 16384)
                    xr = torch.randn(rows, H, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
                    flat_x.update(t=up(xr, ttnn.bfloat16, ttnn.ReplicateTensorToMesh(mesh_device)), rows=rows)
                sh = ttnn.ShardTensorToMesh(mesh_device, dim=0)
                devin = [(up(c_, ttnn.uint32, sh), up(r_, ttnn.uint32, sh)) for c_, r_, _ in hostin]
                ttnn.deallocate(exo(flat_x["t"], *devin[0]))  # (warm)
                ttnn.synchronize_device(mesh_device)

                def run():
                    for c_d, r_d in devin:
                        for _ in range(reps):
                            ttnn.deallocate(exo(flat_x["t"], c_d, r_d))

                w = window(run)["flat"]
                assert len(w) == reps * len(devin), f"{name}: {len(w)} flat programs, {reps * len(devin)} calls"
                for c_d, r_d in devin:
                    ttnn.deallocate(c_d)
                    ttnn.deallocate(r_d)
                return [statistics.median(w[i * reps : (i + 1) * reps]) for i in range(len(devin))]

            if "cal" not in d:
                # the compute layout: EP's own; TP: SPC_C slices a chip at the same width (the real TP4 chip's 48)
                if tp == 1:
                    exc, spc_c = ex, cf.spc
                else:
                    spc_c = SPC_C
                    ccf = Cfg(name, tp, shape, NE, n_e, I, spc=spc_c)
                    local = [tuple(m.T.contiguous() for m in _expert_mats(Q, ccf, l)) for l in range(spc_c)]
                    exc = fw.FlatRoutedExpert(
                        mesh_device, [local] * N, m=kw["max_tokens_per_expert"], H=H, I=ccf.I,
                        gids=[list(range(k * spc_c, (k + 1) * spc_c)) for k in range(N)], n_global=spc_c * N,
                        act=act_name, pin=1, layout=kw["layout"], keep_packed=True,
                    )  # fmt: skip
                    del local
                d["spc_compute"] = spc_c
                # (a) calibration of the flat op at this width: A (active slots at 32 tokens), B (every slot, more rows)
                act_n, rows_n = ([3, 6, 12], [256, 512, 1024]) if tp == 1 else ([12, 24, 48], [64, 128, 256])
                ta = flat_run([np.where(np.arange(spc_c)[None, :] < n, 32, 0).repeat(N, 0) for n in act_n], exc)
                tb = flat_run([np.full((N, spc_c), r_) for r_ in rows_n], exc)
                a_slope = float(np.polyfit(act_n, ta, 1)[0])
                b_slope = float(np.polyfit([spc_c * r_ / 32 for r_ in rows_n], tb, 1)[0])
                d["cal"] = {"act": act_n, "act_ms": [round(v, 4) for v in ta], "rows": rows_n,
                            "rows_ms": [round(v, 4) for v in tb], "A_ms": a_slope, "B_ms": b_slope}  # fmt: skip
                logger.info(f"TP {name} calibration (width {cf.I}): A {a_slope * 1e3:.2f} us a slot, B "
                            f"{b_slope * 1e3:.3f} us a row tile; act {ta}, rows {tb}")  # fmt: skip
                # (b) the flat op at every layer's real chunk-10 counts (TP8: two calls of 48 slices, summed)
                d["flat_layers"] = {}
                for wl in ("novel", "code"):
                    Ls = sorted(real[wl])
                    cnts, parts = [], []
                    for L in Ls:
                        own = real[wl][L]
                        if tp == 1:
                            cnts.append(own)
                            parts.append(1)
                            continue
                        gc = M.group_counts(own, shape, tp)
                        per_chip = np.zeros((N, len(gc[0])), dtype=np.int64)
                        for gi, grp in enumerate(M.groups(shape, tp)):
                            per_chip[grp] = gc[gi]
                        n_p = per_chip.shape[1] // spc_c
                        cnts += [per_chip[:, q * spc_c : (q + 1) * spc_c] for q in range(n_p)]
                        parts.append(n_p)
                    ts = flat_run(cnts, exc)
                    out, i = [], 0
                    for n_p in parts:
                        out.append(round(float(sum(ts[i : i + n_p])), 4))
                        i += n_p
                    d["flat_layers"][wl] = out
                    logger.info(f"TP {name} flat at real counts {wl}: {sum(out):.2f} ms a chunk at width {cf.I} "
                                f"(x {cf.scale:.3f} -> {sum(out) * cf.scale:.2f})")  # fmt: skip
                if exc is not ex:
                    ttnn.deallocate(exc.w_gu)
                    ttnn.deallocate(exc.w_d)
                    del exc
            # (c) head-driven routings: dispatch, experts phase (fused), the flat op on the same counts
            rnames = ["uniform"] if tp == R else ["uniform", "novel", "code"]
            for rname in rnames:
                if tp == R:
                    bias = bias_virtual
                    top6 = None
                else:
                    bias, _ = steer(rname, n_e, cf.NX)
                    top6 = steer(rname, TOPK)[1]  # (the real 384-expert routing: the TP4 destinations at 6 experts)
                route(bias)
                ids, ws, xin = routing_host()
                cnt = torch.bincount(ids.reshape(-1), minlength=cf.E).reshape(N, cf.spc).numpy()
                per_chip = cnt.sum(1)
                call()  # (warm: programs built)
                ttnn.synchronize_device(mesh_device)

                def run():
                    for _ in range(REPS):
                        call()

                w = window(run)
                assert len(w["combine"]) == REPS and len(w["dispatch"]) == REPS, {k: len(v) for k, v in w.items()}
                fl = flat_run([cnt])[0]
                dsp, cmb = statistics.median(w["dispatch"]), statistics.median(w["combine"])
                ex_ids = torch.from_numpy(cf.xo)[ids]
                whole = int(sum(len(set(r_.tolist())) == n_e for r_ in ex_ids))
                chip_of = ids // cf.spc
                owner = torch.arange(ids.shape[0]) // (chunk // N)
                dst = float(np.mean([len(set(r_.tolist()) - {int(o)}) for r_, o in zip(chip_of, owner)]))
                run_rec = {
                    "n_e": n_e, "K": cf.K, "dispatch_ms": round(dsp, 4), "experts_ms": round(cmb, 4),
                    "flat_ms": round(fl, 4), "combine_cost_ms": round(cmb - fl, 4),
                    "busy_over_mean_tokens": round(float(per_chip.max() / per_chip.mean()), 3),
                    "dests_per_token": round(dst, 3), "tokens_whole": whole, "tokens": int(ids.shape[0]),
                    "dests_per_token_6": round(dests(top6, tp), 3) if top6 is not None else None,
                    "dispatch_runs_ms": [round(v, 4) for v in w["dispatch"]],
                    "experts_runs_ms": [round(v, 4) for v in w["combine"]],
                }  # fmt: skip
                d["runs"].setdefault(rname, []).append(run_rec)
                # (a token whose n_e-th and next expert score within the router's rounding may take a mix of their
                # slices: a few in 5120, harmless for the timing; a wrong layout would split most tokens)
                check(whole >= 0.995 * ids.shape[0], f"{name} n_e {n_e} {rname}: {whole} of {ids.shape[0]} tokens took "
                      f"every slice of {n_e} experts (>= 99.5 %)")  # fmt: skip
                logger.info(f"TP {name} n_e {n_e} {rname}: {run_rec}")
                if rname == "uniform" and n_e == nes[-1]:
                    # (d) accuracy vs fp32 on SAMPLE tokens a chip (bfp4-grid weights, the head's gate weights)
                    y = call(dispatch=False)
                    got = torch.cat([ttnn.to_torch(t).reshape(-1, H)[: chunk // N].float()
                                     for t in ttnn.get_device_tensors(y)])  # fmt: skip
                    gs = torch.Generator().manual_seed(77)
                    Sc = chunk // N
                    pick = torch.cat([me * Sc + torch.randperm(Sc, generator=gs)[:SAMPLE] for me in range(N)])
                    q4 = lambda w_: ttnn.to_torch(
                        ttnn.from_torch(w_, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
                    ).float()  # noqa: E731
                    wq = {}
                    want = torch.zeros(len(pick), H)
                    for i, t in enumerate(pick.tolist()):
                        for kk in range(cf.K):
                            l = int(ids[t, kk]) % cf.spc
                            if cf.wset(l) not in wq:
                                gm, um, dm = _expert_mats(Q, cf, l)
                                wq[cf.wset(l)] = tuple(q4(m.T.contiguous()).T for m in (gm, um, dm))
                            gq, uq, dq = wq[cf.wset(l)]
                            xr = xin[t : t + 1]
                            want[i] += float(ws[t, kk]) * (fw.act_ref(xr @ gq, xr @ uq, act_name) @ dq)[0]
                    err = float((got[pick] - want).norm() / want.norm())
                    accuracy[name] = {"rel_err_vs_fp32": err, "K": cf.K}
                    logger.info(
                        f"TP {name} accuracy (uniform, K {cf.K}, {len(pick)} tokens): rel err vs fp32 {err:.5f}"
                    )
                    check(err < 0.1, f"{name}: rel err vs fp32 {err:.5f} < 0.1 (a wrong sum or index gives ~1)")
            ttnn.deallocate(x)
            res.release()
            del blk, state
            ttnn.synchronize_device(mesh_device)
            logger.info(f"TP {name} n_e {n_e} done in {time.time() - t0:.1f} s")
    for o in experts_cache.values():
        ttnn.deallocate(o.w_gu)
        ttnn.deallocate(o.w_d)
    if flat_x["t"] is not None:
        ttnn.deallocate(flat_x["t"])

    # ---- extrapolation to the real K (6 experts a token): combine cost linear in K, dispatch linear in the chips a
    # token's row goes to (TP8 at the real routings: its uniform value x TP4's real / uniform ratio)
    def fit_at(xs_, ys_, x_at):
        if len(xs_) < 2:
            return float(ys_[0]), float("nan")
        k1, k0 = np.polyfit(xs_, ys_, 1)
        return float(k0 + k1 * x_at), float(k1)

    def _ratio(a_, b_):
        return a_ / b_ if b_ > 1e-3 else 1.0

    ext: dict = {}
    for name, tp, nes in plan:
        ext[name] = {}
        for rname, runs in dev[name]["runs"].items():
            K6 = TOPK * (1 if tp == 1 else tp)
            Ks = [r_["K"] for r_ in runs]
            cmb6, cmb_slope = fit_at(Ks, [r_["combine_cost_ms"] for r_ in runs], K6)
            if tp == R:  # (uniform only; the 6-expert destinations of this virtual layout from its own n_e runs)
                dsp6, dsp_slope = fit_at(Ks, [r_["dispatch_ms"] for r_ in runs], K6)
                d6 = None
            else:
                d6 = runs[0]["dests_per_token_6"]
                dsp6, dsp_slope = fit_at([r_["dests_per_token"] for r_ in runs], [r_["dispatch_ms"] for r_ in runs], d6)
            ext[name][rname] = {"K": K6, "combine_cost_ms": round(cmb6, 4), "combine_ms_per_K": round(cmb_slope, 5),
                                "dispatch_ms": round(dsp6, 4), "dispatch_slope": round(dsp_slope, 5),
                                "dests_per_token": d6, "measured_K": Ks}  # fmt: skip
        logger.info(f"TP extrapolated {name}: {ext[name]}")
    tpn, tp8n = f"tp{C}", f"tp{R}v"
    for rname in ("novel", "code"):
        u4, r4 = ext[tpn]["uniform"], ext[tpn][rname]
        u8 = ext[tp8n]["uniform"]
        ext[tp8n][rname] = {
            "K": u8["K"],
            "combine_cost_ms": round(u8["combine_cost_ms"] * _ratio(r4["combine_cost_ms"], u4["combine_cost_ms"]), 4),
            "dispatch_ms": round(u8["dispatch_ms"] * _ratio(r4["dispatch_ms"], u4["dispatch_ms"]), 4),
            "from": f"uniform x {tpn} {rname} / uniform",
        }
    rec["extrapolated"] = ext

    # ------------------------------------------------------------------ STEP 1 refined: fitted per layout
    fits = {}
    for name, d in dev.items():  # (at the true slice width: x scale)
        fits[name] = {"A_ms": d["cal"]["A_ms"] * d["scale"], "B_ms": d["cal"]["B_ms"] * d["scale"],
                      "A_ms_measured": d["cal"]["A_ms"], "B_ms_measured": d["cal"]["B_ms"], "width": d["I"]}  # fmt: skip
    fits["ep"]["A_over_A0"] = fits["ep"]["A_ms"] / A0
    for name in (tpn, tp8n):
        fits[name]["A_ratio_vs_ep"] = fits[name]["A_ms"] / fits["ep"]["A_ms"]
        fits[name]["B_ratio_vs_ep"] = fits[name]["B_ms"] / fits["ep"]["B_ms"]
    rec["fits"] = fits

    # ------------------------------------------------------------------ STEP 3: per chunk at 51,200
    ep_run = {r_: dev["ep"]["runs"][r_][0] for r_ in ("uniform", "novel", "code")}
    summary = {}
    for wl in ("novel", "code"):
        comb = {"ep": ep_run[wl]["combine_cost_ms"], "tp4": ext[tpn][wl]["combine_cost_ms"],
                "tp8": ext[tp8n][wl]["combine_cost_ms"]}  # fmt: skip
        disp = {
            "ep": ep_run[wl]["dispatch_ms"],
            "tp4": ext[tpn][wl]["dispatch_ms"],
            "tp8": ext[tp8n][wl]["dispatch_ms"],
        }
        experts = {
            "ep": sum(dev["ep"]["flat_layers"][wl]) + LAYERS_N * comb["ep"],
            "tp4": sum(dev[tpn]["flat_layers"][wl]) * dev[tpn]["scale"] + LAYERS_N * comb["tp4"],
            "tp8": sum(dev[tp8n]["flat_layers"][wl]) * dev[tp8n]["scale"] + LAYERS_N * comb["tp8"],
        }
        d_add = {k: disp[k] - disp["ep"] for k in ("tp4", "tp8")}
        c_add = {k: comb[k] - comb["ep"] for k in ("tp4", "tp8")}
        base = P56[wl]
        chunk_ms = {
            "ep_off_measured": base["device_off"],
            "ep_prev_measured": base["device_prev"],
            **{k: base["device_off"] + experts[k] - experts["ep"] + LAYERS_N * d_add[k] for k in ("tp4", "tp8")},
        }
        summary[wl] = {
            "experts_ms": {k: round(v, 2) for k, v in experts.items()},
            "flat_ms_chunk": {k: round(sum(dev[n]["flat_layers"][wl]) * dev[n]["scale"], 2)
                              for k, n in (("ep", "ep"), ("tp4", tpn), ("tp8", tp8n))},
            "combine_cost_ms_layer": {k: round(v, 4) for k, v in comb.items()},
            "dispatch_ms_layer": {k: round(v, 4) for k, v in disp.items()},
            "dispatch_added_ms_layer": {k: round(v, 4) for k, v in d_add.items()},
            "combine_added_ms_layer": {k: round(v, 4) for k, v in c_add.items()},
            "chunk_ms": {k: round(v, 2) for k, v in chunk_ms.items()},
            "experts_measured_p55a_ms": p55["summary"][f"{wl}_p51200"]["median"]["measured_rec_ms"],
            "experts_measured_p56_off_prev_ms": [base["experts_off"], base["experts_prev"]],
        }  # fmt: skip
        logger.info(f"TP VERDICT {wl}: {summary[wl]}")
        metrics.record(f"ep_experts_ms_{wl}", round(experts["ep"], 3))
        metrics.record(f"tp4_experts_ms_{wl}", round(experts["tp4"], 3))
        metrics.record(f"tp8_experts_ms_{wl}", round(experts["tp8"], 3))
        metrics.record(f"tp4_chunk_ms_{wl}", round(chunk_ms["tp4"], 2))
        metrics.record(f"tp8_chunk_ms_{wl}", round(chunk_ms["tp8"], 2))
        metrics.record(f"ep_off_chunk_ms_{wl}", base["device_off"])
        metrics.record(f"ep_prev_chunk_ms_{wl}", base["device_prev"])
        for k in ("tp4", "tp8"):  # (> 0: slower than EP + P.56)
            metrics.record(f"{k}_minus_prev_chunk_ms_{wl}", round(chunk_ms[k] - base["device_prev"], 2))
        for k in ("tp4", "tp8"):
            metrics.record(f"{k}_dispatch_added_ms_{wl}", round(d_add[k], 4))
            metrics.record(f"{k}_combine_added_ms_{wl}", round(c_add[k], 4))
    # (the per-layer headline numbers: the novel prompt's layer-LAYER routing, at 5,120 tokens)
    nv = summary["novel"]
    for k in ("tp4", "tp8"):
        metrics.record(f"{k}_dispatch_added_ms", nv["dispatch_added_ms_layer"][k])
        metrics.record(f"{k}_combine_added_ms", nv["combine_added_ms_layer"][k])
        metrics.record(f"{k}_reduce_ms", 0.0)  # (the TP sum is the combine's own fp32 accumulation)
    metrics.record("tp_fit_A_ms", {k: round(v["A_ms"], 5) for k, v in fits.items()})
    metrics.record("tp_fit_B_ms", {k: round(v["B_ms"], 6) for k, v in fits.items()})
    for k, v in accuracy.items():
        metrics.record(f"{k}_rel_err_vs_fp32", round(v["rel_err_vs_fp32"], 6))

    # ---- STEP 1 with the fitted constants, every recorded chunk (P.45) and chunk 10 (P.55a)
    def fitted(name, tp):
        f = fits[name]
        return (tp, f["A_ms"], f["B_ms"], 0.0)

    params = {"ep": fitted("ep", 1), "tp4": fitted(tpn, C), "tp8": fitted(tp8n, R)}
    cpu_fit = {}
    for wl in ("novel", "code"):
        rows = []
        src = p45[wl] or {10: real[wl]}
        for ch, lay in sorted(src.items()):
            t = M.chunk_table(lay, shape, params)
            rows.append({"chunk": ch, **{k: round(v["ms"], 2) for k, v in t.items()},
                         **{f"{k}_busy_over_mean": round(v["busy_over_mean"], 3) for k, v in t.items()}})  # fmt: skip
        cpu_fit[wl] = rows
        logger.info(f"TP CPU fitted (compute only, no C0) {wl}: {rows}")
    rec["cpu_fitted"] = cpu_fit
    rec["p45_chunks"] = {wl: sorted(p45[wl]) for wl in p45}

    # ---- the per-layer table (chunk 10 at 51,200): flat EP / TP4 / TP8 measured, the scaled CPU model
    table = {}
    for wl in ("novel", "code"):
        Ls = sorted(real[wl])
        table[wl] = []
        for i, L in enumerate(Ls):
            own = real[wl][L]
            table[wl].append({
                "layer": L,
                "flat_ms": {k: round(dev[k]["flat_layers"][wl][i] * dev[k]["scale"], 4) for k in dev},
                "cpu_scaled_ms": {k: round(M.layer_ms(own, shape, *scaled[k])["ms"] - C00, 4) for k in scaled},
                "busy_over_mean": {k: round(M.layer_ms(own, shape, *scaled[k])["busy_over_mean"], 3) for k in scaled},
                "tokens_busiest_chip": int(own.sum(1).max()),
            })  # fmt: skip
    rec.update(dev=dev, accuracy=accuracy, summary=summary, table=table, p56=P56)
    metrics.record_json("tp_check", rec)
    metrics.record("tp_checks_failed", len(fails))
    logger.info(f"TP total {time.time() - t_start:.0f} s; fails {fails}")
    ttnn.deallocate(lead)
    assert not fails, fails
