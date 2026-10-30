# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: the flat routed expert overlapped with the router-forwarded level-1 combine (ttnn.bringup.combine_rf), one
program per chip (ttnn.bringup.flat_routed_expert_with_program), on the LoudBox as a 2 x 4 FABRIC_2D mesh, K3 per-chip
load: 56 local experts per chip (448 in all), top-16, 320 home tokens per chip (~5120 rows per chip, as on a 4 x 4),
latent rows 3584 wide. The combine runs on the cells the flat plan leaves free (FC_CELLS), its state, rings and CBs in
the flat expert's arena (persistent, owned here); the flat y writers report each local expert to the combine's
collector, the combine reads an expert's rows once reported. Fake (uninitialised) expert weights: perf only; the first
fused call is checked against the sum of the y rows it read back (y zeroed before: a combine that reads early sums
zeros). Times: fused, flat alone (the same op, nothing appended), combine alone. Run WITHOUT TT_MESH_GRAPH_DESC_PATH.

The close-out cases take the mesh as a parameter (MESHES: shape, fabric, wrap links where cabled); on a Galaxy the
smaller meshes are carves (TT_VISIBLE_DEVICES from galaxy_carve.py and the carve's single-mesh descriptor,
GALAXY_RUNBOOK.md), the 8x4 the whole Galaxy. Their per-chip load stays the Galaxy 8x4's (or K3's 4x4 stage, the
load4x4 ids) whatever the mesh, so the flat expert's work per chip is the same on every mesh."""

import collections
import dataclasses
import itertools
import os
import statistics
import time

import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, port_luts
from ttnn.bringup.combine_rf.routing_a2a import routing_a2a_planner_prepare, routing_a2a_reserve
from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan, place_ports
from ttnn.bringup.combine_rf.sparse_combine_l1 import route, tie_mode
from ttnn.bringup.combine_rf.sparse_combine_rf import (
    WRITER_WORDS_ROWS,
    combine_auto,
    fused_arena_reserve,
    sparse_combine_rf_prepare,
)
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import GALAXY, MODELS, fabric_payload, per_chip
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert, act_ref
from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import layout_from_env

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

L = 2
LINK_GBPS = 25.0
DRAM_GBS, BFP4_TILE = 512.0, 576  # (as test_flat_bench.py)
# the close-out meshes: shape, fabric, wrap links (X: along a row, the C axis; Y: along a column, the R axis)
MESHES = {
    "2x4": ((2, 4), ttnn.FabricConfig.FABRIC_2D, (False, False)),  # the LoudBox; Galaxy rows 0-1, line descriptor
    "2x4ring": ((2, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_X, (True, False)),  # Galaxy rows 0-1, the rows' wrap cables
    "4x4": ((4, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_X, (True, False)),  # Galaxy rows 0-3: LINE x RING
    "4x4torus": ((4, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY, (True, True)),  # rows 2-5, subtorus Galaxies only
    "8x4": ((8, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY, (True, True)),  # the whole Galaxy
    # (a LoudBox 2x2 square as a 1x4: p150_x4_line_1x4 / p150_x4_ring_1x4 and the square's TT_VISIBLE_DEVICES)
    "1x4": ((1, 4), ttnn.FabricConfig.FABRIC_2D, (False, False)),
    "1x4ring": ((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_X, (True, False)),
}


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
    arena_reserve_kb: int = None  # None: the flat plan's static CBs and the writers' words (fused_arena_reserve)
    det_repeat: int = 0
    soak: int = 0  # fused calls after the first, back to back (every 500th checked)
    landing: str = "rings"  # the combine's landing pools: "rings" (a ring a source) / "shared" (slot grants)
    pool_slots: int = None  # the landing pool's slots an accumulator (None: the planner's)
    pipe: bool = False
    trace: bool = False  # (traced replays back to back: wall, and the profiler's device time a replay)
    timing: bool = True
    weights: str = "fake"
    gates: bool = True
    order: str = "token"
    routings: tuple = ("uniform",)
    plan_cores: int = 32
    token_major: bool = True
    # the device tables from: "a2a" (routing_a2a - each chip's own routing, an all-to-all of the entries that point at
    # each chip's experts, the planner on local data) or "gather" (combine_plan on the replicated routing; also what
    # the deterministic combine and home-major rows use)
    routing_exchange: str = "a2a"
    layout: object = None  # the flat layout (FlatLayoutOptions / preset dict; None: the plan's defaults)
    closeout: bool = False  # a close-out case (closeout_cfg): the plan's defaults are checked
    combine_mode: str = None  # combine_auto's mode ("ports apart" / "acc_on_ports"); None: its choice
    a_max: int = 16  # combine_auto's accumulators at most (A_MAX)
    R: int = 2  # the mesh (default the LoudBox's 2 x 4, no wrap)
    C: int = 4
    torus: tuple = (False, False)  # (wrap X, wrap Y) where cabled

    @property
    def N(self):
        return self.R * self.C

    @property
    def wrap(self):  # (as mesh_plan: an axis of 2 has no wrap)
        return (self.torus[0] and self.C > 2, self.torus[1] and self.R > 2)

    @property
    def E(self):
        return self.N * self.EPC


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
        arena_reserve_kb=int(e["FC_ARENA_RESERVE_KB"]) if e.get("FC_ARENA_RESERVE_KB") else None,
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


def closeout_cfg(model, chunk, mesh="2x4", routings=("uniform",), load=GALAXY, **kw):
    """A close-out case (EXPERTS_COMBINE_CLOSEOUT.md): the model's per-chip load on stage `load` (default the Galaxy
    8x4: experts and home tokens a chip) on the mesh MESHES[mesh], every choice the planners' defaults (flat: DRISC only, even splits; combine: sized from
    the cells and the arena); real weights (one random expert set shared by all chips), correctness only."""
    md, pc = MODELS[model], per_chip(model, load, chunk)
    (R, C), _, torus = MESHES[mesh]
    # (A/B: FC_CLOSE_PAYLOAD / FC_CLOSE_PIECE - the fabric payload and the combine's row bytes a packet; pieces need
    # FC_CLOSE_L1=0 - no L1 hand-off, so host-planned tables and the ports-apart mode)
    if os.environ.get("FC_CLOSE_L1") == "0":
        kw.setdefault("l1", False)
        kw.setdefault("devplan", False)
        kw.setdefault("combine_mode", "ports apart")
    return FC(
        R=R,
        C=C,
        torus=torus,
        S=pc["S"],
        K=md["K"],
        EPC=pc["epc"],
        H=md["H"],
        I=md["I"],
        CAP=R * C * pc["S"],  # (every token at most once per expert)
        ACT=md["act"],
        payload=int(os.environ.get("FC_CLOSE_PAYLOAD", fabric_payload(md["H"]))),
        piece_max=int(os.environ.get("FC_CLOSE_PIECE", max(7168, 2 * md["H"]))),  # (the L1 hand-off: one piece a row)
        weights="real",
        timing=kw.pop("timing", False),
        routings=routings,
        closeout=True,
        # (the flat plan's hint: the expected rows per expert, chunk x top-k / experts - its sub-block size)
        layout=kw.pop("layout", dict(rows_hint=round(pc["M"]))),
        **kw,
    )


_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_KEYS = itertools.count(1)
_PARAMS = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(fabric_payload_size=int(os.environ.get("FC_PAYLOAD", "7232")), l1_small_size=0),
        id="fabric2d-mesh-2x4-2link",
    )
]


def _mesh_params(mesh, payload):
    shape, fabric, _ = MESHES[mesh]
    payload = int(os.environ.get("FC_CLOSE_PAYLOAD", payload))  # (the closeout payload A/B: the fabric opened at it)
    return shape, fabric2d_device_params(fabric_payload_size=payload, l1_small_size=0, fabric_config=fabric)


_SCALE_CHUNKS = (5120, 8192, 16384)
_LOADS = {GALAXY: "", "4x4": "-load4x4"}  # (the 4x4 stage load: K3 only)
_CLOSEOUT = [
    pytest.param(
        *_mesh_params(mesh, fabric_payload(MODELS[m]["H"])),
        m,
        chunk,
        mesh,
        load,
        id=f"{m}-{chunk}-{mesh}{_LOADS[load]}",
    )
    for m in MODELS
    if m != "dsv41_flash"  # (host planning only: tests/unit/test_closeout_plans.py)
    for chunk in _SCALE_CHUNKS
    if (m, chunk) == ("k27", 5120) or m != "k27"  # (K2.7: being deprecated, chunk 5120 only)
    for mesh in MESHES
    for load in _LOADS
    if load == GALAXY or m == "k3"
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
    S, K, E, EPC, N = c.S, c.K, c.E, c.EPC, c.N
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


YTIE = tie_mode()  # (the combine's Y tie split: the floor on the routes it sends)


def _routing(name, c, seed=0):
    """rows_by_chip (y row, home, t, link, local expert) in expert order, per-chip counts / regions over the global
    experts, rows per chip (padded), the busiest link's rows."""
    S, K, E, EPC, R, C, N = c.S, c.K, c.E, c.EPC, c.R, c.C, c.N
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
                for hc, d in route(src, chips[o], R, C, *c.wrap, link, YTIE):
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
    "mesh_device, device_params, model, chunk, mesh, load", _CLOSEOUT, indirect=["mesh_device", "device_params"]
)
@pytest.mark.timeout(3600)
def test_flat_combine_closeout(mesh_device, device_params, model, chunk, mesh, load):
    """A close-out case: the model's per-chip load (the Galaxy's, or K3's 4x4 stage), the planners' defaults, on the
    mesh `mesh`, the fused result checked."""
    c = closeout_cfg(model, chunk, mesh, routings=tuple(os.environ.get("FC_ROUTINGS", "uniform").split(",")), load=load)
    logger.info(f"closeout {model} chunk {chunk} on {mesh} (load {load}): {c}")
    _run(mesh_device, c)


# the combine's mode A/B (accumulators on the ports or apart, at the same accumulator count, and combine_auto's own
# choice): pipelined fused calls, uniform routing, chunk 5120 (needs TT_METAL_DEVICE_PROFILER=1
# TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1)
_AB = [
    ("glm53", "ports apart", 12),
    ("glm53", "acc_on_ports", 12),
    ("glm53", None, 16),
    ("k3", "ports apart", 16),
    ("k3", "acc_on_ports", 16),
]
# the scaling cases: the planners' defaults (combine_auto's choice), every mesh, chunk and routing
_SCALE_MODELS = ("k3", "glm53_flash", "glm53", "m3", "k27")
_ROUTINGS = ("uniform", "zipf_spread_1.2", "zipf_packed_1.2")
_PERF = [(m, mode, a, 5120, "uniform", GALAXY) for m, mode, a in _AB if mode is not None] + [
    (m, None, 16, chunk, routing, load)
    for m in _SCALE_MODELS
    for chunk in _SCALE_CHUNKS
    for routing in _ROUTINGS
    for load in _LOADS
    if load == GALAXY or (m == "k3" and chunk == 5120)
]


@pytest.mark.parametrize(
    "mesh_device, device_params, model, mode, a_max, chunk, routing, mesh, load",
    [
        pytest.param(
            *_mesh_params(mesh, fabric_payload(MODELS[m]["H"])),
            m,
            mode,
            a,
            chunk,
            routing,
            mesh,
            load,
            id=f"{m}-{chunk}-{routing}-{'auto' if mode is None else mode.replace(' ', '_')}-A{a}-{mesh}{_LOADS[load]}",
        )
        for m, mode, a, chunk, routing, load in _PERF
        for mesh in MESHES
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(3600)
def test_flat_combine_closeout_perf(mesh_device, device_params, model, mode, a_max, chunk, routing, mesh, load):
    """The close-out case timed (pipelined, ITERS calls): the combine mode A/B at chunk 5120, and the scaling cases
    (the planners' defaults over meshes, chunks, routings and K3's 4x4 stage load)."""
    c = closeout_cfg(
        model, chunk, mesh, routings=(routing,), load=load, combine_mode=mode, a_max=a_max, timing=True, pipe=True,
        ITERS=12,
    )  # fmt: skip
    logger.info(f"closeout perf {model} chunk {chunk} {routing} on {mesh} (load {load}) mode {mode} A <= {a_max}: {c}")
    _run(mesh_device, c)


# landing ring depth D per model (the combine's pool = D slots a source chip), at each model's Galaxy 8x4 per-chip load
# on this mesh: D forced through FC.ring_depth (None: the planner's choice here); A and everything else the planners'
# defaults. (The Galaxy 8x4 plans D: GLM-5.3 2, GLM-5.3-Flash 4, M3 2, K3 5, DSv4.1-Flash 3 at chunk 5120.)
_DEPTHS = (None, 1, 2, 3, 4, 5, 6, 8)
_DEPTH_MODELS = ("k3", "glm53_flash", "glm53", "m3", "dsv41_flash")


@pytest.mark.parametrize(
    "mesh_device, device_params, model, depth, mesh",
    [
        pytest.param(*_mesh_params(mesh, fabric_payload(MODELS[m]["H"])), m, d, mesh, id=f"{m}-D{d or 'auto'}-{mesh}")
        for m in _DEPTH_MODELS
        for d in _DEPTHS
        for mesh in ("2x4", "1x4ring")
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(3600)
def test_flat_combine_depth(mesh_device, device_params, model, depth, mesh):
    """Combine alone, flat alone and fused (pipelined) at a forced landing ring depth D."""
    c = closeout_cfg(model, 5120, mesh, routings=("uniform",), timing=True, pipe=True, ITERS=12, ring_depth=depth)
    logger.info(f"depth {model} D {depth} on {mesh}: {c}")
    _run(mesh_device, c)


@pytest.mark.parametrize(
    "mesh_device, device_params, model, mode, per_dir",
    [
        pytest.param(
            *_mesh_params("2x4", fabric_payload(MODELS[m]["H"])),
            m,
            mode,
            pd,
            id=f"{m}-{mode.replace(' ', '_')}-{'dirports' if pd else 'linkports'}",
        )  # fmt: skip
        for m in _DEPTH_MODELS
        for mode in ("acc_on_ports", "ports apart")
        for pd in (False, True)
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(3600)
def test_flat_combine_layout(mesh_device, device_params, model, mode, per_dir):
    """A combine layout forced (accumulators on the ports or apart, A <= 16; a port per link or per (direction, link)),
    the planner's D - the Galaxy 4x4's layout (acc_on_ports, A 16) on the 2x4."""
    c = closeout_cfg(
        model, 5120, "2x4", routings=("uniform",), timing=True, pipe=True, ITERS=12, combine_mode=mode, a_max=16,
        per_dir_ports=per_dir,
    )  # fmt: skip
    logger.info(f"layout {model} {mode} per_dir {per_dir}: {c}")
    _run(mesh_device, c)


# the landing pools: rings vs shared (slot grants) at the same pool, each model at its Galaxy 8x4 per-chip load; pool
# None: the planner's, else that many slots an accumulator (the Galaxy's slots a source on this mesh)
_LANDING = [
    (m, chunk, routing, pool)
    for m in ("glm53", "m3")
    for chunk, pools in ((5120, (None,)), (8192, (None, 11)), (16384, (None, 11)))
    for pool in pools
    for routing in ("uniform",) + (("zipf_spread_1.2",) if chunk == 8192 else ())
]
_LANDING_GALAXY = [  # (the whole Galaxy: 31 sources, the planner's pool)
    (m, 5120, "uniform", None, "8x4") for m in ("k3", "glm53_flash", "glm53", "m3", "k27")
]


@pytest.mark.parametrize(
    "mesh_device, device_params, model, chunk, routing, pool, landing, mesh",
    [
        pytest.param(
            *_mesh_params(mesh, fabric_payload(MODELS[m]["H"])),
            m,
            chunk,
            routing,
            pool,
            landing,
            mesh,
            id=f"{m}-{chunk}-{routing}-pool{pool or 'auto'}-{landing}" + ("" if mesh == "2x4" else f"-{mesh}"),
        )  # fmt: skip
        for m, chunk, routing, pool, mesh in [x + ("2x4",) for x in _LANDING] + _LANDING_GALAXY
        for landing in ("rings", "shared")
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(3600)
def test_flat_combine_landing(mesh_device, device_params, model, chunk, routing, pool, landing, mesh):
    """The combine's landing pools, rings vs shared (slot grants), ports apart, at one pool size."""
    c = closeout_cfg(
        model, chunk, mesh, routings=(routing,), timing=True, pipe=True, ITERS=12, combine_mode="ports apart",
        landing=landing, pool_slots=pool, a_max=int(os.environ.get("FC_AMAX", "16")),  # (FC_AMAX: accumulators at most)
    )  # fmt: skip
    logger.info(f"landing {model} {chunk} {routing} pool {pool} {landing}: {c}")
    _run(mesh_device, c)


@pytest.mark.parametrize(
    "mesh_device, device_params, model, depth, calls",
    [
        pytest.param(*_mesh_params("2x4", fabric_payload(MODELS[m]["H"])), m, d, 20000, id=f"{m}-D{d}-soak")
        for m, d in (("glm53", 2), ("glm53_flash", 4), ("k3", 5))
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(7200)
def test_flat_combine_depth_soak(mesh_device, device_params, model, depth, calls):
    """The fused call soaked at a model's Galaxy D (with CombineProbes.bk the Galaxy's source count's credit policy)."""
    c = closeout_cfg(model, 5120, "2x4", routings=("uniform",), timing=False, ring_depth=depth, soak=calls)
    _run(mesh_device, c)


@pytest.mark.parametrize(
    "mesh_device, device_params, model, chunk, routing, pool, calls",
    [
        pytest.param(
            *_mesh_params("2x4", fabric_payload(MODELS[m]["H"])),
            m,
            chunk,
            routing,
            11,
            calls,
            id=f"{m}-{chunk}-{routing}-pool11-soak{calls // 1000}k",
        )  # fmt: skip
        for m, chunk, routing, calls in (
            ("glm53", 8192, "uniform", 20000),
            ("glm53", 8192, "zipf_spread_1.2", 20000),
            ("m3", 16384, "uniform", 20000),
            ("glm53", 8192, "zipf_spread_1.2", 1_000_000),
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(7200)
def test_flat_combine_landing_soak(mesh_device, device_params, model, chunk, routing, pool, calls):
    """The shared landing pool soaked at the Galaxy's slots per source (fused calls back to back, ports apart)."""
    c = closeout_cfg(
        model, chunk, "2x4", routings=(routing,), timing=False, combine_mode="ports apart", landing="shared",
        pool_slots=pool, soak=calls,
    )  # fmt: skip
    _run(mesh_device, c)


_SOAK_GALAXY = [(m, r) for m in _SCALE_MODELS for r in ("uniform", "zipf_spread_1.0", "zipf_packed_1.0")]


@pytest.mark.parametrize(
    "mesh_device, device_params, model, routing",
    [
        pytest.param(*_mesh_params("8x4", fabric_payload(MODELS[m]["H"])), m, r, id=f"{m}-{r}-8x4-soak")
        for m, r in _SOAK_GALAXY
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(7200)
def test_flat_combine_soak_galaxy(mesh_device, device_params, model, routing):
    """The fused flat + combine soaked on the Galaxy (auto layout, the probes from the environment): FC_SOAK_CALLS fused
    calls back to back after the first, every 500th and the last checked against the reference."""
    c = closeout_cfg(
        model, 5120, "8x4", routings=(routing,), timing=False, soak=int(os.environ.get("FC_SOAK_CALLS", "20000"))
    )
    logger.info(f"galaxy soak {model} {routing}: {c.soak} calls")
    _run(mesh_device, c)


_TRACED = [
    pytest.param(
        _mesh_params(mesh, fabric_payload(MODELS[m]["H"]))[0],
        {**_mesh_params(mesh, fabric_payload(MODELS[m]["H"]))[1], "trace_region_size": 64 << 20},
        m,
        chunk,
        mesh,
        id=f"{m}-{chunk}-uniform-auto-{mesh}",
    )
    for m in _SCALE_MODELS
    for chunk in _SCALE_CHUNKS
    for mesh in MESHES
]


@pytest.mark.parametrize(
    "mesh_device, device_params, model, chunk, mesh", _TRACED, indirect=["mesh_device", "device_params"]
)
@pytest.mark.timeout(3600)
def test_flat_combine_closeout_traced(mesh_device, device_params, model, chunk, mesh):
    """The close-out case (defaults, uniform) traced: fused, flat alone and combine alone each captured once and replayed
    back to back (wall a replay; with TT_METAL_DEVICE_PROFILER=1 also each replay's device time, slowest chip)."""
    c = closeout_cfg(  # (FC_ROUTINGS: every routing on the one flat expert, its weights packed once)
        model, chunk, mesh, routings=tuple(os.environ.get("FC_ROUTINGS", "uniform").split(",")), timing=True, trace=True
    )
    logger.info(f"closeout traced {model} chunk {chunk} on {mesh}: {c}")
    _run(mesh_device, c)


def _run(mesh_device, c):
    N = c.N
    assert tuple(mesh_device.shape) == (c.R, c.C), (tuple(mesh_device.shape), (c.R, c.C))
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
    if c.closeout:  # (the plan's defaults must give the even split)
        assert op.plan["even_split"], "the gate/up split is not the even one"
    logger.info(f"flat expert built in {time.time() - t0:.1f} s (weights {c.weights})")
    for routing in c.routings:
        t0 = time.time()
        _flat_combine_rf_routing(mesh_device, op, routing, c)
        logger.info(f"{routing}: done in {time.time() - t0:.1f} s")


def _check_y(ys, x_host, tidx_t, cnts, regs, op, c, routing, per_chip_rows=96):
    """Sampled y rows of every chip against the bfp4-quantized reference expert (the harness's real weights: one
    expert on every chip); flat row r is x row token_index[r]."""
    g = torch.Generator().manual_seed(1)  # (as _run: the shared expert's weights)
    N = c.N
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
    ITERS = int(os.environ.get("FC_ITERS_FORCE", ITERS))  # (e.g. NoC-trace captures: few calls)
    R, C, N, TORUS = c.R, c.C, c.N, c.torus
    torch.manual_seed(0)
    rows, counts, regions, n_rows, busiest = _routing(routing, c)
    floor = busiest * H * 2 / 1e6 / LINK_GBPS * 1e3
    plan = op.plan
    writers = len(plan["y_writers"])
    grid0 = mesh_device.compute_with_storage_grid_size()
    fill_t = None
    if float(os.environ.get("SPC_FILLY", "0")):  # (probe: the Y-link filler's sink - 16 KB on every core, garbage)
        gcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid0.x - 1, grid0.y - 1))])
        fill_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([grid0.x * grid0.y, 4096]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(gcrs, [1, 4096], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
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
    RX = DEVPLAN and c.routing_exchange == "a2a" and not DET and not PER_DIR_PORTS and c.token_major
    rx_res = (
        routing_a2a_reserve(mesh_device, L=L, torus=TORUS) if RX else None
    )  # (its persistent L1: before the arena's sizing)
    # the arena: every worker core, zero at creation, kept for the test (the combine's state lives in it). It takes all
    # the L1 the allocator has left (the combine sizes its rings from it), less a reserve for the programs' static CBs
    # (the flat expert's non-arena CBs, below the arena: its plan's static_l1_bytes) and the L1 tensors allocated after
    # it (the writers' words) - fused_arena_reserve; arena_kb / arena_reserve_kb: a fixed size / reserve instead. Sized
    # before the device planner runs (its CBs must not sit under the arena), allocated after it
    grid = mesh_device.compute_with_storage_grid_size()
    a_rows = int(plan["arena_tiles"]) * 32
    if c.arena_kb:
        a_rows = max(a_rows, c.arena_kb * 1024 // 64 // 32 * 32)
    else:
        mv = ttnn.device.get_memory_view(mesh_device, ttnn.BufferType.L1)
        reserve = (
            c.arena_reserve_kb * 1024
            if c.arena_reserve_kb is not None
            else fused_arena_reserve(int(plan["static_l1_bytes"]), l1=c.l1)
        )
        a_rows = max(a_rows, (int(mv.largest_contiguous_bytes_free_per_bank) - reserve) // 64 // 32 * 32)
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
        chips_, _, _ = mesh_plan(mesh_device, L, TORUS)
        port_cells = {(v.x, v.y) for v in place_ports(mesh_device, chips_, cells).values()}
        auto = combine_auto(
            N, S, H, len(cells), len(port_cells), arena_bytes=arena_bytes, l1=L1, depth=c.depth, piece_max=c.piece_max,
            a_max=c.a_max, mode=c.combine_mode, landing=c.landing,
        )  # fmt: skip
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
        luts, nps = port_luts(mesh_device, L, TORUS, per_direction=PER_DIR_PORTS)
        if RX:  # each chip's own routing ([S / 32, 32 x 2 K] uint16: router_topk's joint rows), exchanged on device
            wb = gates.view(torch.int16).to(torch.int64) & 0xFFFF
            joint = torch.cat([ids.to(torch.int64), wb], -1).reshape(R, C, S // 32, 32 * 2 * K).to(torch.int32)
            own_t = ttnn.from_torch(
                joint,
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=dram,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1)),
            )
            planner = routing_a2a_planner_prepare(
                mesh_device,
                S=S,
                K=K,
                EPC=EPC,
                A=n_acc,
                L=L,
                torus=TORUS,
                port_luts=luts,
                n_ports=nps,
                rows_cap=rows_cap,
                tokens_per_page=32,
                reserved=rx_res,
            )
            pl = planner(own_t)
            logger.info(f"{routing}: device tables from the routing all-to-all ({planner.M} planner cores)")
        else:
            planner = combine_planner_prepare(
                mesh_device,
                S=S,
                K=K,
                EPC=EPC,
                A=n_acc,
                L=L,
                torus=TORUS,
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
    common = dict(num_links=L, torus=TORUS, n_acc=n_acc, depth=copt["depth"], cells=cells, arena=arena)
    if fill_t is not None:
        common["fill_sink"] = int(fill_t.buffer_address())
    combine_kw = dict(
        piece_max=copt["piece_max"],
        inj_slots=copt["inj_slots"],
        ring_depth=copt["ring_depth"],
        local_keep=copt["local_keep"],
        landing=c.landing,
    )
    if c.pool_slots:  # (the pool capped: the Galaxy's slots a source on this mesh)
        combine_kw["pool_slots"] = c.pool_slots
    l1 = None
    if L1:
        wcores = [tuple(v) for v in plan["y_writers"]]  # (the factory's writer order: reader tails, then down)
        wcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*v), ttnn.CoreCoord(*v)) for v in wcores])
        words_t = ttnn.from_torch(  # the writers' grants / grant bases / table page / trailer stages
            torch.zeros(len(wcores) * WRITER_WORDS_ROWS, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(wcrs, [WRITER_WORDS_ROWS, 16], ttnn.ShardOrientation.ROW_MAJOR),
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
    if auto is not None and c.ring_depth is None and not c.pool_slots:  # (combine_auto's - unless D / the pool forced)
        assert fused.pool == auto["layout"]["PSL"], (fused.pool, auto["layout"]["PSL"])
    # (combine alone: the same cells in the same roles and ring depth - the arena's cumulative counters are shared; with
    # accumulators on the ports (L1 hand-off only) there is no alone variant)
    alone = (
        None
        if ACC_ON_PORTS or not (c.timing or os.environ.get("FC_SOAK_FN") == "alone")
        else sparse_combine_rf_prepare(
            mesh_device,
            S,
            H,
            rows,
            writers=0,
            ports_per_direction=PER_DIR_PORTS,
            **{**common, "depth": fused.depth},
            **{**combine_kw, "pool_slots": fused.pool},  # (the fused build's pool exactly: its CBs off the fused slots)
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
    logger.info(
        f"{routing}: ring depth fused {fused.depth}" + ("" if alone is None else f", alone {alone.depth}")
        + f"; credit policy fused {getattr(fused, 'cred_policy', '?')}"
        + ("" if alone is None else f", alone {getattr(alone, 'cred_policy', '?')}")
    )
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
    assert (
        worst < 2e-2 or os.environ.get("SPC_NOYREAD") == "1" or os.environ.get("SPC_ABL", "0") != "0"
    ), f"{routing}: combine result off ({worst:.2e})"
    for i_ in range(
        c.soak
    ):  # (a soak: fused calls back to back, every 500th and the last checked against the reference)
        # (FC_SOAK_FN: fused (default) / alone (the combine without the flat) / flat (the flat without the combine: its
        # output is not the combine's - the checks see the last fused call's); FC_SOAK_EVERY: checks every N calls)
        {"alone": lambda: alone(y, out), "flat": run_flat}.get(os.environ.get("FC_SOAK_FN", ""), run_fused)()
        every_ = int(os.environ.get("FC_SOAK_EVERY", "500"))
        if os.environ.get("FC_SOAK_NOCHECK") == "1":  # (hangs only: no result checks)
            if i_ % every_ == every_ - 1:
                ttnn.synchronize_device(mesh_device)
                logger.info(f"{routing}: soak {i_ + 1} of {c.soak} calls (no check), worst per-token rel err 0")
            continue
        if i_ % every_ == every_ - 1 or i_ == c.soak - 1:
            ttnn.synchronize_device(mesh_device)
            o2 = [ttnn.to_torch(t).reshape(S, H).float() for t in ttnn.get_device_tensors(out)]
            w2 = max(
                [
                    ((o2[o][fin[o]] - ref[o][fin[o]]).norm(dim=1) / ref[o][fin[o]].norm(dim=1).clamp(min=1e-30))
                    .max()
                    .item()
                    for o in range(N)
                    if fin[o].any()
                ]
                + [0.0]
            )
            logger.info(f"{routing}: soak {i_ + 1} of {c.soak} fused calls, worst per-token rel err {w2:.2e}")
            assert w2 < 2e-2, f"{routing}: soak call {i_ + 1}: combine result off ({w2:.2e})"

    def tprof(fn, run_, name):  # (SPC_TPROF: timeline of one call per chip, us from the combine's first start)
        for _ in range(int(os.environ.get("SPC_TPROF_N", "1"))):  # (N > 1: the last of N back to back, as pipelined)
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
                    f"passes {r[14]} piggy {r[15]} | first row {us(r[16]) if r[16] else -1:.0f} last row "
                    f"{us(r[17]) if r[17] else -1:.0f} us | blocked: no layout {r[18]} full {r[19]} | kernel {us(r[22]):.0f} loop {us(r[1]):.0f} "
                    f"headers {us(r[20]):.0f} barrier {us(r[21]):.0f} loop end {us(r[23]):.0f}"
                )
            if os.environ.get("SPC_TPROF_ACC") == "1":  # (each accumulator: page, end, rows, local rows, local done)
                logger.info(
                    f"TPROF {routing} {name} chip {ci} acc list: "
                    + " ".join(f"{r[0] & 0xFF}:{us(r[3]):.0f}/{r[5]}/{r[4]}/{us(r[2]) if r[2] else -1:.0f}" for r in adp)
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
    if c.trace:  # (each variant captured once, after its eager calls above: every program built and loaded)

        def traced(fn, name, batches=(10, 40, 100), run_=None):
            fn()
            ttnn.synchronize_device(mesh_device)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            prof = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
            if prof:
                ttnn.ReadDeviceProfiler(mesh_device)  # (flush the eager calls and the first replay)
            walls, devs = [], []
            for n in batches:
                t0 = time.perf_counter()
                for _ in range(n):
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                walls.append((time.perf_counter() - t0) * 1e6 / n)
                if not prof:
                    continue
                ttnn.ReadDeviceProfiler(mesh_device)
                reps = collections.defaultdict(list)  # replay -> per chip its programs' device sum (us)
                for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
                    per = collections.defaultdict(float)
                    for p_ in programs:
                        res = (getattr(p_, "program_analyses_results", None) or {}).get(_DURATION_KEY)
                        uid = p_.program_execution_uid
                        if res is not None and int(uid.trace_id) == int(tid):
                            per[int(uid.trace_id_counter)] += float(res.duration) / 1e3
                    for r_, v in per.items():
                        reps[r_].append(v)
                d = sorted(max(v) for v in reps.values())
                devs.append(d)
                logger.info(
                    f"FC_TRACE {routing} {name:13s}: {n} replays back to back: wall {walls[-1]:.0f} us a replay; device "
                    f"(slowest chip) median {statistics.median(d) if d else 0:.0f} min {d[0] if d else 0:.0f} "
                    f"max {d[-1] if d else 0:.0f} ({len(d)} replays seen)"
                )
            if run_ is not None and os.environ.get("SPC_TPROF") == "1":  # (the last replay's timeline: traced, no host gaps)
                tprof(lambda: None, run_, f"{name.split()[0]}-traced")
            ttnn.release_trace(mesh_device, tid)
            if not prof:
                logger.info(f"FC_TRACE {routing} {name:13s}: wall a replay " + " ".join(f"{w:.0f}" for w in walls))
            return statistics.median(walls), (statistics.median(devs[-1]) if devs and devs[-1] else None)

        w_fused, d_fused = traced(run_fused, "fused", run_=fused)
        w_flat, d_flat = traced(run_flat, "flat alone")
        w_comb, d_comb = (None, None) if alone is None else traced(lambda: alone(y, out), "combine alone", run_=alone)
        fmt = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
        logger.info(
            f"FC_TRACE_SUMMARY {routing} {R}x{C} wrap {c.wrap} E{EPC}/chip K{K} S{S} H{H}: wall a replay fused "
            f"{fmt(w_fused)} flat {fmt(w_flat)} combine {fmt(w_comb)} | device fused {fmt(d_fused)} flat {fmt(d_flat)} "
            f"combine {fmt(d_comb)} us | combine link floor {floor:.0f} us"
        )
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
        # (the flat expert's DRAM traffic, the busiest chip: its bfp4 weights and its x rows; y stays in L1 with l1)
        w_bytes = EPC * 3 * (H // 32) * (I // 32) * BFP4_TILE
        dram = (w_bytes + n_rows * H * 2 * (1 if c.l1 else 2)) / (t_flat * 1e-6) / 1e9 / DRAM_GBS
        logger.info(
            f"FC_SUMMARY {routing:16s} {R}x{C} wrap {c.wrap} E{EPC}/chip K{K} S{S} H{H} pipelined: flat {t_flat:.0f} "
            f"(DRAM {dram:.0%} of {DRAM_GBS:.0f} GB/s) + combine {t_comb:.0f} = {t_flat + t_comb:.0f} serial vs fused "
            f"{t_fused:.0f} us ({100 * (t_flat + t_comb - t_fused) / t_comb:.0f}% of the combine hidden; fused / flat "
            f"{t_fused / t_flat:.3f}) | combine link floor {floor:.0f} us (combine / floor {t_comb / floor:.2f})"
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
