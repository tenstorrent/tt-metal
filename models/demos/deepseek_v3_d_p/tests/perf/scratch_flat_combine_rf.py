# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: the flat routed expert overlapped with the router-forwarded level-1 combine (ttnn.bringup.combine_rf), one
program per chip (ttnn.bringup.flat_routed_expert_with_program), on the LoudBox as a 2 x 4 FABRIC_2D mesh, K3 per-chip
load: 56 local experts per chip (448 in all), top-16, 320 home tokens per chip (~5120 rows per chip, as on a 4 x 4),
latent rows 3584 wide. The combine runs on the cells the flat plan leaves free (FC_CELLS), its state, rings and CBs in
the flat expert's arena (persistent, owned here); the flat y writers report each local expert to the combine's
collector, the combine reads an expert's rows once reported. Fake (uninitialised) expert weights: perf only; the first
fused call is checked against the sum of the y rows it read back (y zeroed before: a combine that reads early sums
zeros). Times: fused, flat alone (the same op, nothing appended), combine alone. Run WITHOUT TT_MESH_GRAPH_DESC_PATH."""

import dataclasses
import itertools
import os
import statistics
import time

import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, port_luts
from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan, place_ports
from ttnn.bringup.combine_rf.sparse_combine_l1 import route
from ttnn.bringup.combine_rf.sparse_combine_rf import combine_auto, sparse_combine_rf_prepare
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import CHUNKS, GALAXY, MODELS, fabric_payload, per_chip
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert, act_ref
from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import layout_from_env

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

R, C = 2, 4
N = R * C
L = 2
LINK_GBPS = 25.0


@dataclasses.dataclass
class FC:
    """One harness configuration: the per-chip expert load, the flat expert and the combine. n_acc None: the
    combine's configuration from the cells and the arena the flat plan leaves (combine_auto)."""

    S: int  # home tokens per chip
    K: int  # top-k
    EPC: int  # local experts per chip
    H: int
    I: int
    CAP: int  # the flat expert's capacity (tokens per expert)
    ACT: str
    payload: int  # fabric payload bytes
    ITERS: int = 10
    CELLS: list = None  # the combine's cells (default: every cell the flat plan leaves free)
    devplan: bool = True  # every per-call table from the device planner (with l1)
    det: bool = False
    per_dir_ports: bool = False
    acc_on_ports: bool = False
    l1: bool = True  # y rows straight from the down cores into the combine's L1 (no y in DRAM)
    n_acc: int = None
    depth: int = 8
    piece_max: int = None
    inj_slots: int = 64
    ring_depth: int = None
    local_keep: int = 8
    arena_kb: int = None
    arena_reserve_kb: int = 48
    det_repeat: int = 0
    pipe: bool = False
    timing: bool = True
    weights: str = "fake"
    gates: bool = True
    order: str = "token"
    routings: tuple = ("uniform",)
    plan_cores: int = 32
    token_major: bool = True
    layout: object = None  # the flat layout (FlatLayoutOptions / preset dict; None: the plan's defaults)

    @property
    def E(self):
        return N * self.EPC


def _env_cfg():
    """The scratch recipes' FC_* / SPC_* environment (defaults: K3 load on the LoudBox)."""
    e = os.environ
    cells = [tuple(int(v) for v in c.split(",")) for c in e["FC_CELLS"].split(";")] if e.get("FC_CELLS") else None
    devplan = e.get("FC_DEVPLAN") == "1"
    return FC(
        S=int(e.get("FC_S", "320")),
        K=int(e.get("FC_K", "16")),
        EPC=int(e.get("FC_EPC", "56")),
        H=int(e.get("FC_H", "3584")),
        I=int(e.get("FC_I", "3072")),
        CAP=int(e.get("FC_CAP", "8192")),
        ACT=e.get("FC_ACT", "situ"),
        payload=int(e.get("FC_PAYLOAD", "7232")),
        ITERS=int(e.get("FC_ITERS", "10")),
        CELLS=cells,
        devplan=devplan,
        det=devplan and e.get("FC_DET", "0") == "1",
        per_dir_ports=devplan and e.get("FC_PORTS", "link") == "dir",
        acc_on_ports=devplan and e.get("FC_ACC_ON_PORTS", "0") == "1",
        l1=e.get("FC_L1") == "1",
        # (FC_NACC unset with the device planner: the combine's configuration from the cells and the arena)
        n_acc=int(e["FC_NACC"]) if e.get("FC_NACC") else (None if devplan else 16),
        depth=int(e.get("FC_D", "4")),
        piece_max=int(e.get("SPC_PIECE", "7168")),
        inj_slots=int(e.get("SPC_QP", "64")),
        ring_depth=int(e["SPC_D"]) if e.get("SPC_D") else None,
        arena_kb=int(e["FC_ARENA_KB"]) if e.get("FC_ARENA_KB") else None,
        arena_reserve_kb=int(e.get("FC_ARENA_RESERVE_KB", "48")),
        det_repeat=int(e.get("FC_DET_REPEAT", "10")),
        pipe=e.get("FC_PIPE") == "1",
        weights=e.get("FC_WEIGHTS", "fake"),
        gates=e.get("FC_GATES", "1") != "0",
        order="home" if e.get("FC_ORDER") == "home" else "token",
        routings=tuple(e.get("FC_ROUTINGS", "uniform,zipf_spread_1.2,zipf_packed_1.2").split(",")),
        plan_cores=int(e.get("FC_PLAN_CORES", "32")),
        token_major=e.get("FC_PLAN_TM", "1") == "1",
        layout=layout_from_env(),  # (the harness's MIMO_FL_* recipe names, parsed into FlatLayoutOptions)
    )


def closeout_cfg(model, chunk, routings=("uniform",)):
    """A close-out case (EXPERTS_COMBINE_CLOSEOUT.md): the model's Galaxy 8x4 per-chip load (experts and home tokens a
    chip) on the LoudBox 2x4, every choice the planners' defaults (flat: DRISC only, even splits; combine: sized from
    the cells and the arena); real weights (one random expert set shared by all chips), correctness only."""
    md, pc = MODELS[model], per_chip(model, GALAXY, chunk)
    return FC(
        S=pc["S"],
        K=md["K"],
        EPC=pc["epc"],
        H=md["H"],
        I=md["I"],
        CAP=N * pc["S"],  # (every token at most once per expert)
        ACT=md["act"],
        payload=fabric_payload(md["H"]),
        piece_max=max(7168, 2 * md["H"]),  # (the L1 hand-off: one piece a row)
        weights="real",
        timing=False,
        routings=routings,
    )


_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_KEYS = itertools.count(1)
_PARAMS = [
    pytest.param(
        (R, C),
        fabric2d_device_params(fabric_payload_size=int(os.environ.get("FC_PAYLOAD", "7232")), l1_small_size=0),
        id="fabric2d-mesh-2x4-2link",
    )
]
_CLOSEOUT = [
    pytest.param(
        (R, C),
        fabric2d_device_params(fabric_payload_size=fabric_payload(MODELS[m]["H"]), l1_small_size=0),
        m,
        chunk,
        id=f"{m}-{chunk}",
    )
    for m in MODELS
    if m != "dsv41_flash"  # (host planning only: tests/unit/test_closeout_plans.py)
    for chunk in CHUNKS
]


def _chip_ns(mesh_device):
    """Device kernel time per chip of the programs since the last read (needs TT_METAL_DEVICE_PROFILER=1
    TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1)."""
    ttnn.ReadDeviceProfiler(mesh_device)
    chip_ns = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        ns = [
            float(p.program_analyses_results[_DURATION_KEY].duration)
            for p in programs
            if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
        ]
        if ns:
            chip_ns.append(sum(ns))
    assert chip_ns, "profiler produced no kernel durations"
    return chip_ns


def _ids(name, g, c):
    """[N, S, K] global expert ids per home chip and token."""
    S, K, E, EPC = c.S, c.K, c.E, c.EPC
    if name == "uniform":
        return torch.stack([torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(S)]) for _ in range(N)])
    if name.startswith("zipf"):
        s = float(name.split("_")[-1])
        r = torch.arange(E)
        perm = r if "packed" in name else (r % N) * EPC + r // N  # rank -> expert
        w = torch.zeros(E)
        w[perm] = 1.0 / (r.double() + 1).pow(s).float()
        return torch.multinomial(w.expand(N * S, E), K, replacement=False, generator=g).view(N, S, K)
    raise ValueError(name)


def _routing(name, c, seed=0):
    """rows_by_chip (y row, home, t, link, local expert) in expert order, per-chip counts / regions over the global
    experts, rows per chip (padded), the busiest link's rows."""
    S, K, E, EPC = c.S, c.K, c.E, c.EPC
    g = torch.Generator().manual_seed(seed)
    ids = _ids(name, g, c)
    # gate weights per (home, token, k): softmax over the token's experts, bf16 (gates off: all 1)
    gates = torch.softmax(torch.randn(N, S, K, generator=g), -1).to(torch.bfloat16)
    if not c.gates:
        gates = torch.ones(N, S, K, dtype=torch.bfloat16)
    gbits = gates.view(torch.int16).to(torch.int64) & 0xFFFF
    chips = [(r, c) for r in range(R) for c in range(C)]
    pend = {ch: [] for ch in chips}
    load = {}
    for o in range(N):
        for t in range(S):
            for k in range(K):
                e = int(ids[o, t, k])
                src = chips[e // EPC]
                link = (t + k) % L
                pend[src].append((e % EPC, o, t, link, int(gbits[o, t, k])))
                for hc, d in route(src, chips[o], R, C, False, False, link):
                    load[(hc, d, link)] = load.get((hc, d, link), 0) + 1
    rows, counts, regions, n_rows = (
        {},
        torch.zeros(N, 1, E, dtype=torch.int32),
        torch.zeros(N, 1, E, dtype=torch.int32),
        0,
    )
    for ci, ch in enumerate(chips):
        # expert order (as the flat expert emits them); within an expert token-major (FC_ORDER=home: by home chip,
        # which leaves the last homes' rows of the last expert for the very end)
        key = (lambda r: r) if c.order == "home" else (lambda r: (r[0], r[2], r[1]))
        lst = sorted(pend[ch], key=key)
        cnt = [0] * EPC
        for e, *_ in lst:
            cnt[e] += 1
        reg, acc = [], 0
        for e in range(EPC):
            reg.append(acc)
            acc += (cnt[e] + 31) // 32 * 32
        n_rows = max(n_rows, acc)
        seen = [0] * EPC
        rows[ch] = []
        for e, o, t, link, wb in lst:
            rows[ch].append((reg[e] + seen[e], chips[o], t, link, e, wb))
            seen[e] += 1
        counts[ci, 0, ci * EPC : (ci + 1) * EPC] = torch.tensor(cnt, dtype=torch.int32)
        regions[ci, 0, ci * EPC : (ci + 1) * EPC] = torch.tensor(reg, dtype=torch.int32)
    _routing.last = (ids, gates)  # (the device planner's inputs)
    return rows, counts, regions, max(n_rows, 32), max(load.values(), default=0)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(3600)
def test_flat_combine_rf(mesh_device, device_params):
    """Every routing of FC_ROUTINGS on one mesh and one flat expert (its weights packed and uploaded once)."""
    _run(mesh_device, _env_cfg())


@pytest.mark.parametrize(
    "mesh_device, device_params, model, chunk", _CLOSEOUT, indirect=["mesh_device", "device_params"]
)
@pytest.mark.timeout(3600)
def test_flat_combine_closeout(mesh_device, device_params, model, chunk):
    """A close-out case: the model's Galaxy per-chip load, the planners' defaults, the fused result checked."""
    c = closeout_cfg(model, chunk, routings=tuple(os.environ.get("FC_ROUTINGS", "uniform").split(",")))
    logger.info(f"closeout {model} chunk {chunk}: {c}")
    _run(mesh_device, c)


def _run(mesh_device, c):
    assert tuple(mesh_device.shape) == (R, C)
    H, I, EPC = c.H, c.I, c.EPC
    if c.weights == "real":  # (finite y for the check: one random expert shared by all)
        g = torch.Generator().manual_seed(1)
        w1 = tuple(torch.randn(*sh, generator=g) * 0.02 for sh in ((H, I), (H, I), (I, H)))
        weights = [[w1] * EPC for _ in range(N)]  # (the same tuple everywhere: packed once, replicated)
    else:
        weights = "fake"
    t0 = time.time()
    op = FlatRoutedExpert(
        mesh_device,
        weights,
        m=c.CAP,
        H=H,
        I=I,
        gids=[list(range(ch * EPC, (ch + 1) * EPC)) for ch in range(N)],
        n_global=c.E,
        wdtype="bf4",
        act=c.ACT,
        pin=1,
        layout=c.layout,
    )
    logger.info(f"flat layout {op.layout}")
    logger.info(f"flat expert built in {time.time() - t0:.1f} s (weights {c.weights})")
    for routing in c.routings:
        t0 = time.time()
        _flat_combine_rf_routing(mesh_device, op, routing, c)
        logger.info(f"{routing}: done in {time.time() - t0:.1f} s")


def _check_y(ys, x_host, tidx_t, cnts, regs, op, c, routing, per_chip_rows=96):
    """Sampled y rows of every chip against the bfp4-quantized reference expert (the harness's real weights: one
    expert on every chip); flat row r is x row token_index[r]."""
    g = torch.Generator().manual_seed(1)  # (as _run: the shared expert's weights)
    w = [torch.randn(*sh, generator=g) * 0.02 for sh in ((c.H, c.I), (c.H, c.I), (c.I, c.H))]
    q = lambda t: ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa: E731
    Wg, Wu, Wd = (q(t) for t in w)
    tix = [ttnn.to_torch(t).to(torch.int64).reshape(-1) for t in ttnn.get_device_tensors(tidx_t)]
    worst = 1.0
    for ci in range(N):
        act = [
            r_
            for gid in range(ci * c.EPC, (ci + 1) * c.EPC)
            for r_ in range(int(regs[ci][gid]), int(regs[ci][gid] + cnts[ci][gid]))
        ]
        sel = act[:: max(1, len(act) // per_chip_rows)][:per_chip_rows]
        xs = x_host[tix[ci][sel]].float()
        ref = act_ref(xs @ Wg, xs @ Wu, c.ACT) @ Wd
        ok, pcc = comp_pcc(ref, ys[ci][sel], 0.99)
        worst = min(worst, float(pcc))
        assert ok, f"{routing}: chip {ci} y vs the reference expert: PCC {pcc}"
    logger.info(f"{routing}: flat y vs the reference expert, min PCC {worst:.5f} over {N} chips")


def _flat_combine_rf_routing(mesh_device, op, routing, c):
    S, K, EPC, H, I, CAP, ITERS = c.S, c.K, c.EPC, c.H, c.I, c.CAP, c.ITERS
    torch.manual_seed(0)
    rows, counts, regions, n_rows, busiest = _routing(routing, c)
    floor = busiest * H * 2 / 1e6 / LINK_GBPS * 1e3
    plan = op.plan
    writers = len(plan["y_writers"])
    grid0 = mesh_device.compute_with_storage_grid_size()
    # (DRISC: only the reader cells that still run Tensix readers / the launcher are taken; the rest go to combine)
    rd_key = "tensix_readers" if "tensix_readers" in plan else "readers"
    taken = {tuple(v) for k in ("arena_cores", "gu", "down", rd_key, "relays", "coords") for v in plan[k]}
    cells = c.CELLS or [(x_, y_) for y_ in range(grid0.y) for x_ in range(grid0.x) if (x_, y_) not in taken]
    logger.info(
        f"{routing}: flat roles "
        + ", ".join(f"{k} {len(plan[k])}" for k in ("gu", "down", rd_key, "relays", "arena_cores") if k in plan)
        + f"; {len(cells)} cells left for the combine"
    )
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    rm = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram, mesh_mapper=shard
    )
    DEVPLAN, DET, PER_DIR_PORTS, L1 = c.devplan, c.det, c.per_dir_ports, c.l1
    assert not DEVPLAN or L1, "the device planner needs the L1 hand-off (l1)"
    # the arena: every worker core, zero at creation, kept for the test (the combine's state lives in it). It takes all
    # the L1 the allocator has left (the combine sizes its rings from it), less a reserve for the programs' static CBs
    # (the flat expert's few non-arena CBs, below the arena) and the L1 tensors allocated after it (the writers'
    # words); arena_kb: a fixed size instead. Sized before the device planner runs (its CBs must not sit under the
    # arena), allocated after it
    grid = mesh_device.compute_with_storage_grid_size()
    a_rows = int(plan["arena_tiles"]) * 32
    if c.arena_kb:
        a_rows = max(a_rows, c.arena_kb * 1024 // 64 // 32 * 32)
    else:
        mv = ttnn.device.get_memory_view(mesh_device, ttnn.BufferType.L1)
        a_rows = max(
            a_rows, (int(mv.largest_contiguous_bytes_free_per_bank) - c.arena_reserve_kb * 1024) // 64 // 32 * 32
        )
    arena_bytes = a_rows * 64
    copt = dict(
        n_acc=c.n_acc,
        acc_on_ports=c.acc_on_ports,
        inj_slots=c.inj_slots,
        depth=c.depth,
        local_keep=c.local_keep,
        piece_max=c.piece_max,
        ring_depth=c.ring_depth,
    )
    auto = None
    if c.n_acc is None:  # the combine's configuration from the cells and the arena (combine_auto)
        chips_, _, _ = mesh_plan(mesh_device, L)
        port_cells = {(v.x, v.y) for v in place_ports(mesh_device, chips_, cells).values()}
        auto = combine_auto(
            N, S, H, len(cells), len(port_cells), arena_bytes=arena_bytes, l1=L1, depth=c.depth, piece_max=c.piece_max
        )
        assert auto[
            "fits"
        ], f"{routing}: the combine does not fit ({len(cells)} cells, {arena_bytes >> 10} KB): {auto['why']}"
        copt.update({k: auto[k] for k in ("n_acc", "acc_on_ports", "inj_slots", "depth", "local_keep")})
        lay = auto["layout"]
        logger.info(
            f"{routing}: combine plan: {len(cells)} cells, {len(port_cells)} port cells, accumulators {auto['n_acc']} "
            f"(on the ports: {auto['acc_on_ports']}), landing pool {lay['PSL']} slots (ring depth {lay['D']}), "
            f"injection ring {lay['QP']}, local ring {lay['QL']} (kept {auto['local_keep']}), fp32 sums "
            f"{lay['acc_fp32'] >> 10} KB; arena {arena_bytes >> 10} KB, free: accumulator "
            f"{(arena_bytes - lay['acc_used']) >> 10} KB, port {(arena_bytes - lay['port_used']) >> 10} KB; flat "
            + ", ".join(f"{k} {(arena_bytes - v) >> 10} KB" for k, v in dict(plan.get("role_bytes", {})).items())
        )
    n_acc, ACC_ON_PORTS = copt["n_acc"], copt["acc_on_ports"]
    tidx = None
    if DEVPLAN:
        ids, gates = _routing.last
        rows_cap = (n_rows + 31) // 32 * 32
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        ids_t = ttnn.from_torch(
            ids.reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=dram,
            mesh_mapper=rep,
        )
        w_t = ttnn.from_torch(
            gates.reshape(1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=dram,
            mesh_mapper=rep,
        )
        luts, nps = port_luts(mesh_device, L, per_direction=PER_DIR_PORTS)
        planner = combine_planner_prepare(
            mesh_device,
            S=S,
            K=K,
            EPC=EPC,
            A=n_acc,
            L=L,
            port_luts=luts,
            n_ports=nps,
            rows_cap=rows_cap,
            cores=[(i % 8, i // 8) for i in range(c.plan_cores)],
            # (the deterministic combine order: the flat's schedule parameters; det off: the arrival-order sums)
            det=dict(
                rps=plan["sched_rps"],
                max_e=plan["sched_max_e"],
                gu_nreg=plan["sched_gu_nreg"],
                pin=1,
                pin_ratio=str(op.layout.pin_ratio),
            )
            if DET
            else None,
            token_major=c.token_major,  # (False: an expert's rows by home, then token)
        )
        pl = planner(ids_t, w_t)
        ttnn.synchronize_device(mesh_device)
        # x: the gathered tokens (every chip's S, replicated); flat row r reads x row token_index[r]
        x_host = (torch.randn(N * S, H) * 0.3).to(torch.bfloat16)
        x = ttnn.from_torch(
            x_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=dram,
            mesh_mapper=rep,
        )
        cd, rd, tidx = pl["counts"], pl["regions"], pl["token_index"]
        n_rows = rows_cap
    else:
        x = rm((torch.randn(N, n_rows, H) * 0.3).to(torch.bfloat16), ttnn.bfloat16)
        cd, rd = rm(counts, ttnn.uint32), rm(regions, ttnn.uint32)
    y = rm(torch.zeros(N, n_rows, H, dtype=torch.bfloat16), ttnn.bfloat16)
    out = rm(torch.zeros(N, 1, S, H, dtype=torch.bfloat16), ttnn.bfloat16)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    arena = ttnn.from_torch(  # (raw words, row-major: readable back core by core)
        torch.zeros(grid.x * grid.y * a_rows, 16, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(all_cores, [a_rows, 16], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    if "role_offsets" in plan:  # (what each flat role uses of the arena: the rest of every core's L1 is free)
        logger.info(
            f"{routing}: flat role offsets (KB) " + ", ".join(f"{k} {v >> 10}" for k, v in plan["role_offsets"].items())
        )
    common = dict(num_links=L, n_acc=n_acc, depth=copt["depth"], cells=cells, arena=arena)
    combine_kw = dict(
        piece_max=copt["piece_max"],
        inj_slots=copt["inj_slots"],
        ring_depth=copt["ring_depth"],
        local_keep=copt["local_keep"],
    )
    l1 = None
    if L1:
        wcores = [tuple(v) for v in plan["y_writers"]]  # (the factory's writer order: reader tails, then down)
        wcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*v), ttnn.CoreCoord(*v)) for v in wcores])
        words_t = ttnn.from_torch(  # the writers' grants / grant bases / table page / trailer stages
            torch.zeros(len(wcores) * 192, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(wcrs, [192, 16], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        l1 = dict(writers=wcores, words=int(words_t.buffer_address()))
    devplan = dict(plan=pl["plan"], table=pl["table"], dord=pl.get("dord"), top_k=K) if DEVPLAN else None
    fused = sparse_combine_rf_prepare(
        mesh_device,
        S,
        H,
        rows,
        writers=writers,
        l1=l1,
        devplan=devplan,
        deterministic=DET,
        ports_per_direction=PER_DIR_PORTS,
        acc_on_ports=ACC_ON_PORTS,
        **common,
        **combine_kw,
    )
    if DEVPLAN:  # (the planner was built for n_acc accumulators: the combine must have found that many cells)
        assert (
            len(fused.cells["accs"]) == n_acc
        ), f"combine has {len(fused.cells['accs'])} accumulators, the planner {n_acc} ({len(cells)} free cells)"
    if auto is not None:  # (combine_auto's layout is the combine's)
        assert fused.pool == auto["layout"]["PSL"], (fused.pool, auto["layout"]["PSL"])
    # (combine alone: the same cells in the same roles and ring depth - the arena's cumulative counters are shared; with
    # accumulators on the ports (L1 hand-off only) there is no alone variant)
    alone = (
        None
        if ACC_ON_PORTS or not c.timing
        else sparse_combine_rf_prepare(
            mesh_device,
            S,
            H,
            rows,
            writers=0,
            ports_per_direction=PER_DIR_PORTS,
            **{**common, "depth": fused.depth},
            pool_slots=fused.pool,  # (the fused build's pool exactly: its CBs must not sit on the fused ring slots)
            **combine_kw,
        )
    )
    coll = fused.collector
    logger.info(
        f"{routing}: flat drisc {plan.get('drisc')} dual {plan.get('dual')} down {len(plan['down'])} cells "
        f"{len(cells)}"
    )
    logger.info(
        f"{routing}: flat writers {writers}, arena {a_rows * 64 >> 10} KB/core, combine ports "
        f"{sorted(set(fused.cells['ports'].values()))} accs {fused.cells['accs']}"
    )
    logger.info(f"{routing}: ring depth fused {fused.depth}" + ("" if alone is None else f", alone {alone.depth}"))
    if L1:
        desc, l1_args = fused.build_l1(y, out)
    else:
        desc, l1_args = fused.build(y, out), []
    empty = ttnn.MeshProgramDescriptor()
    for r_ in range(R):
        for c_ in range(C):
            empty[
                ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r_, c_), ttnn.MeshCoordinate(r_, c_))
            ] = ttnn.ProgramDescriptor()
    keys = _KEYS  # (module-wide: the harness's collect pass and the real pass must not share program-cache entries)
    k_fused, k_flat = next(keys), next(keys)

    def call(extra, key, counts_addr):
        return ttnn.bringup.flat_routed_expert_with_program(
            x,
            cd,
            rd,
            op.gidx,
            op.w_gu,
            op.w_d,
            reader_down_weights=op.w_rd,
            done_words=op.done,
            intermediate=I,
            max_tokens_per_expert=CAP,
            arena=arena,
            y=y,
            extra=extra,
            extra_key=key,
            collector=coll["core"],
            counts_addr=counts_addr,
            writers=writers,
            activation=op.act,
            pin=1,
            token_index=tidx,
            down_fp32=True,
            l1_args=st["l1a"],
            y_l1=st["y_l1"],
            layout=op.layout,
        )

    st = dict(l1a=[], y_l1=False)
    if L1:  # the reference y: one flat call that still writes y (the y writers without the L1 hand-off)
        call(empty, next(keys), coll["counts_addr"] + 512)
        ttnn.synchronize_device(mesh_device)
        ys_ref = [ttnn.to_torch(t).reshape(n_rows, H).float() for t in ttnn.get_device_tensors(y)]
        st["y_l1"] = True  # (the L1 hand-off from here on: without its args the flat expert alone writes no y)
        k_flat = next(keys)  # (flat alone from here on: no y writes)

    def run_fused():
        st["l1a"] = l1_args
        return call(desc, k_fused, coll["counts_addr"])

    def run_flat():  # (dummy counts: the combine never reads them; in L1 mode no hand-off args: no y at all)
        st["l1a"] = []
        return call(empty, k_flat, coll["counts_addr"] + 512)

    # first fused call: the combine's sums against the y rows the flat expert wrote
    run_fused()
    ttnn.synchronize_device(mesh_device)
    ys = ys_ref if L1 else [ttnn.to_torch(t).reshape(n_rows, H).float() for t in ttnn.get_device_tensors(y)]
    if DEVPLAN:  # (rows as the planner placed them: its table gives home, token, weight per flat row)
        tabs = [ttnn.to_torch(t).to(torch.int64).reshape(-1) & 0xFFFFFFFF for t in ttnn.get_device_tensors(pl["table"])]
        cnts = [ttnn.to_torch(t).to(torch.int64).reshape(-1) for t in ttnn.get_device_tensors(pl["counts"])]
        regs = [ttnn.to_torch(t).to(torch.int64).reshape(-1) for t in ttnn.get_device_tensors(pl["regions"])]
        rows = {}
        for ci in range(N):
            ch = (ci // C, ci % C)
            rows[ch] = []
            for gid in range(ci * EPC, (ci + 1) * EPC):
                for r_ in range(int(regs[ci][gid]), int(regs[ci][gid] + cnts[ci][gid])):
                    w0, hdr = int(tabs[ci][2 * r_]), int(tabs[ci][2 * r_ + 1])
                    o, t_ = hdr & 0xFF, (hdr >> 8) & 0xFFFF
                    rows[ch].append((r_, (o // C, o % C), t_, 0, 0, w0 >> 16))
        if c.weights == "real":  # the flat expert's y against the reference (one expert's weights on every chip)
            _check_y(ys, x_host, pl["token_index"], cnts, regs, op, c, routing)
    outs = [ttnn.to_torch(t).reshape(S, H).float() for t in ttnn.get_device_tensors(out)]
    if os.environ.get("FC_OUT_DUMP"):  # (debug: the fused output and the flat expert's y rows, for A/B bit-exactness)
        torch.save({"outs": outs, "ys": ys, "plan": dict(plan)}, f"{os.environ['FC_OUT_DUMP']}.{routing}.pt")
    n_rep = c.det_repeat  # (bit-exact repeats of the fused call: the combine's sum order)
    if n_rep:
        o0 = ttnn.clone(out)
        bad = []
        for i_ in range(n_rep):
            run_fused()
            ne = ttnn.sum(ttnn.ne(out, o0, dtype=ttnn.float32))
            bad.append(sum(int(ttnn.to_torch(t).reshape(-1)[0].item()) for t in ttnn.get_device_tensors(ne)))
        logger.info(f"{routing}: FC_DET {DET}: {sum(b > 0 for b in bad)} of {n_rep} repeats differ (elements {bad})")
        assert not DET or not any(bad), f"{routing}: deterministic combine not bit-exact across repeats"
    chips = [(r_, c_) for r_ in range(R) for c_ in range(C)]
    ref = torch.zeros(N, S, H)
    for ci, ch in enumerate(chips):
        for yrow, home, t, _, _, wb in rows[ch]:
            w = torch.tensor([wb], dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()[0] if L1 else 1.0
            ref[chips.index(home), t] += w * ys[ci][yrow]
    fin = torch.isfinite(ref).all(2)
    worst = 0.0
    for o in range(N):
        f = fin[o]
        if f.any():
            d = (outs[o][f] - ref[o][f]).norm(dim=1) / ref[o][f].norm(dim=1).clamp(min=1e-30)
            worst = max(worst, d.max().item())
    logger.info(f"{routing}: fused first call, worst per-token rel err {worst:.2e} over {int(fin.sum())} finite tokens")
    if worst >= 2e-2:  # (which tokens are off: by home chip; and is it one bad row or garbage?)
        for o in range(N):
            d = (outs[o] - ref[o]).norm(dim=1) / ref[o].norm(dim=1).clamp(min=1e-30)
            bad = (d > 2e-2).nonzero().flatten().tolist()
            logger.info(
                f"  home {o}: {len(bad)} / {S} tokens off; first {bad[:8]}; out[0,:4] {outs[o][0, :4].tolist()} ref {ref[o][0, :4].tolist()}"
            )
    assert worst < 2e-2 or os.environ.get("SPC_NOYREAD") == "1", f"{routing}: combine result off ({worst:.2e})"

    def tprof(fn, run_, name):  # (SPC_TPROF: timeline of one call per chip, us from the combine's first start)
        fn()
        ttnn.synchronize_device(mesh_device)
        for ci, t in enumerate(ttnn.get_device_tensors(run_.tdbg)):
            d = (ttnn.to_torch(t).reshape(48, 128).to(torch.int64) & 0xFFFFFFFF).tolist()
            rdp = [r for r in d[:8] if r[0] >> 20 == 0x7E0]
            r1 = [r for r in d[:8] if r[0] >> 20 == 0x7E3]  # (L1 hand-off port readers: start, all landed, end)
            sdp = [r for r in d[8:16] if r[0] >> 20 == 0x7E1]
            adp = [r for r in d[16:48] if r[0] >> 20 == 0x7E2]
            t0 = min([r[1] for r in rdp + r1 + sdp + adp])
            us = lambda v: (v - t0) / 1350.0  # noqa: E731
            if r1:
                logger.info(
                    f"TPROF {routing} {name} chip {ci}: port rows all landed by {max(us(r[2]) for r in r1):.0f} | port "
                    f"readers end {max(us(r[3]) for r in r1):.0f} | senders end {max(us(r[2]) for r in sdp):.0f} (slot "
                    f"wait {max(r[3] for r in sdp) / 1350:.0f}) | accs local landed "
                    f"{max([us(r[2]) for r in adp if r[2]], default=0):.0f} end {max(us(r[3]) for r in adp):.0f}"
                )
            else:
                ready = {}
                for r in rdp:
                    for e in range(EPC):
                        if r[4 + e]:
                            ready[e] = max(ready.get(e, 0), r[4 + e])
                es = sorted(ready)
                q = [es[int(f * (len(es) - 1))] for f in (0, 0.25, 0.5, 0.75, 1)] if es else []
                logger.info(
                    f"TPROF {routing} {name} chip {ci}: experts ready (e: us) "
                    + " ".join(f"{e}:{us(ready[e]):.0f}" for e in q)
                    + f" | readers end {', '.join(f'{us(r[2]):.0f}' for r in rdp)} (rows {', '.join(str(r[3]) for r in rdp)})"
                    f" | senders end {max(us(r[2]) for r in sdp):.0f}"
                    f" | accs local done {max([us(r[2]) for r in adp if r[2]], default=0):.0f}"
                    f" end {max(us(r[3]) for r in adp):.0f}"
                )
            for r in sdp:  # (per sender: us busy until its end, slot wait, idle, credits, sending; rows, blocked)
                logger.info(
                    f"TPROF {routing} {name} chip {ci} sender {r[0] & 0xFF}: end {us(r[2]):.0f} slot {r[3] / 1350:.0f} "
                    f"idle {r[4] / 1350:.0f} cred {r[5] / 1350:.0f} send {r[8] / 1350:.0f} flush {r[10] / 1350:.0f} us | rows "
                    f"{r[7]} blocked {r[6]} empty {r[9]} credit packets {r[11]} | book {r[12] / 1350:.0f} blk {r[13] / 1350:.0f} us "
                    f"passes {r[14]} piggy {r[15]}"
                )
            ae = sorted((us(r[3]), r[5], r[4]) for r in adp)  # (accumulators: end, rows, local rows)
            logger.info(
                f"TPROF {routing} {name} chip {ci} accs: end min {ae[0][0]:.0f} med {ae[len(ae) // 2][0]:.0f} max "
                f"{ae[-1][0]:.0f} us | rows min {min(a_[1] for a_ in ae)} max {max(a_[1] for a_ in ae)} (local "
                f"{sum(a_[2] for a_ in ae)} of {sum(a_[1] for a_ in ae)})"
            )

    if os.environ.get("SPC_TPROF") == "1":
        tprof(run_fused, fused, "fused")
        if alone is not None:
            tprof(lambda: alone(y, out), alone, "alone")

    def timed(fn, name):
        ttnn.ReadDeviceProfiler(mesh_device)  # (flush)
        per = []
        for _ in range(ITERS):
            fn()
            ttnn.synchronize_device(mesh_device)
            per.append(max(_chip_ns(mesh_device)) / 1e3)
        med = statistics.median(per)
        logger.info(f"FC {routing:16s} {name:13s}: {med:8.1f} us (min {min(per):.1f} max {max(per):.1f})")
        return med

    def pipelined(fn, name):  # (FC_PIPE: ITERS calls queued back to back, one sync; per call, per chip)
        ttnn.ReadDeviceProfiler(mesh_device)
        for _ in range(ITERS):
            fn()
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        per_chip = []
        for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
            per_chip.append(
                [
                    float(p_.program_analyses_results[_DURATION_KEY].duration) / 1e3
                    for p_ in programs
                    if _DURATION_KEY in (getattr(p_, "program_analyses_results", None) or {})
                ]
            )
        calls = [max(c[i] for c in per_chip if len(c) > i) for i in range(min(len(c) for c in per_chip))]
        logger.info(
            f"FC {routing:16s} {name:13s} pipelined: per call (slowest chip) " + " ".join(f"{v:.0f}" for v in calls)
        )
        return statistics.median(calls[1:]) if len(calls) > 1 else calls[0]

    if not c.timing:  # (correctness only)
        return
    if c.pipe:
        t_fused = pipelined(run_fused, "fused")
        if os.environ.get("FC_DRISC_DUMP"):  # (DRISC per-section cycles of the last fused call: MIMO_FL_DRISC_PROF=1)
            ttnn._ttnn.operations.bringup.flat_drisc_dump(mesh_device)
        t_flat = pipelined(run_flat, "flat alone")
        if alone is None:
            logger.info(
                f"FC_SUMMARY {routing:16s} pipelined: flat {t_flat:.0f} | fused {t_fused:.0f} us (fused / flat {t_fused / t_flat:.3f})"
            )
            return
        t_comb = pipelined(lambda: alone(y, out), "combine alone")
        logger.info(
            f"FC_SUMMARY {routing:16s} pipelined: flat {t_flat:.0f} + combine {t_comb:.0f} = {t_flat + t_comb:.0f} serial vs "
            f"fused {t_fused:.0f} us ({100 * (t_flat + t_comb - t_fused) / t_comb:.0f}% of the combine hidden; fused / flat "
            f"{t_fused / t_flat:.3f})"
        )
        return
    t_fused = timed(run_fused, "fused")
    t_flat = timed(run_flat, "flat alone")
    t_comb = timed(lambda: alone(y, out), "combine alone")
    logger.info(
        f"FC_SUMMARY {routing:16s} E{EPC}/chip K{K} S{S} H{H}: flat {t_flat:.0f} + combine {t_comb:.0f} = "
        f"{t_flat + t_comb:.0f} serial vs fused {t_fused:.0f} us ({100 * (t_flat + t_comb - t_fused) / t_comb:.0f}% "
        f"of the combine hidden; fused / flat {t_fused / t_flat:.3f}) | combine link floor {floor:.0f} us"
    )
