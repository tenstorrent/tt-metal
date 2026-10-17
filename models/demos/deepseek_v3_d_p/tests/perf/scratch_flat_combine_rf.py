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

import itertools
import os
import statistics

import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, port_luts
from ttnn.bringup.combine_rf.sparse_combine_l1 import route
from ttnn.bringup.combine_rf.sparse_combine_rf import sparse_combine_rf_prepare
from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

R, C = 2, 4
N = R * C
S = int(os.environ.get("FC_S", "320"))
K = int(os.environ.get("FC_K", "16"))
EPC = int(os.environ.get("FC_EPC", "56"))
E = N * EPC
H = int(os.environ.get("FC_H", "3584"))
I = int(os.environ.get("FC_I", "3072"))
CAP = int(os.environ.get("FC_CAP", "8192"))
ACT = os.environ.get("FC_ACT", "situ")
L = 2
ITERS = int(os.environ.get("FC_ITERS", "10"))
LINK_GBPS = 25.0
# FC_CELLS 'x,y;...': the combine's cells (default: every cell the flat plan leaves free)
CELLS = (
    [tuple(int(v) for v in c.split(",")) for c in os.environ["FC_CELLS"].split(";")]
    if os.environ.get("FC_CELLS")
    else None
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


def _ids(name, g):
    """[N, S, K] global expert ids per home chip and token."""
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


def _routing(name, seed=0):
    """rows_by_chip (y row, home, t, link, local expert) in expert order, per-chip counts / regions over the global
    experts, rows per chip (padded), the busiest link's rows."""
    g = torch.Generator().manual_seed(seed)
    ids = _ids(name, g)
    # gate weights per (home, token, k): softmax over the token's experts, bf16 (FC_GATES=0: all 1)
    gates = torch.softmax(torch.randn(N, S, K, generator=g), -1).to(torch.bfloat16)
    if os.environ.get("FC_GATES", "1") == "0":
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
        key = (lambda r: r) if os.environ.get("FC_ORDER") == "home" else (lambda r: (r[0], r[2], r[1]))
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
@pytest.mark.parametrize("routing", os.environ.get("FC_ROUTINGS", "uniform,zipf_spread_1.2,zipf_packed_1.2").split(","))
@pytest.mark.timeout(3600)
def test_flat_combine_rf(mesh_device, device_params, routing):
    assert tuple(mesh_device.shape) == (R, C)
    torch.manual_seed(0)
    rows, counts, regions, n_rows, busiest = _routing(routing)
    floor = busiest * H * 2 / 1e6 / LINK_GBPS * 1e3
    if os.environ.get("FC_WEIGHTS") == "real":  # (finite y for the check: one random expert shared by all)
        g = torch.Generator().manual_seed(1)
        w1 = tuple(torch.randn(*sh, generator=g) * 0.02 for sh in ((H, I), (H, I), (I, H)))
        weights = [[w1] * EPC for _ in range(N)]
    else:
        weights = "fake"
    op = FlatRoutedExpert(
        mesh_device,
        weights,
        m=CAP,
        H=H,
        I=I,
        gids=[list(range(c * EPC, (c + 1) * EPC)) for c in range(N)],
        n_global=E,
        wdtype="bf4",
        act=ACT,
        pin=1,
    )
    plan = op.plan
    writers = len(plan["down"]) + (len(plan["rdn"]) if plan["rdown"] else 0)
    grid0 = mesh_device.compute_with_storage_grid_size()
    # (DRISC: only the reader cells that still run Tensix readers / the launcher are taken; the rest go to combine)
    rd_key = "tensix_readers" if "tensix_readers" in plan else "readers"
    taken = {tuple(c) for k in ("arena_cores", "gu", "down", rd_key, "relays", "coords") for c in plan[k]}
    cells = CELLS or [(x_, y_) for y_ in range(grid0.y) for x_ in range(grid0.x) if (x_, y_) not in taken]
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    rm = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram, mesh_mapper=shard
    )
    DEVPLAN = os.environ.get("FC_DEVPLAN") == "1"  # (with FC_L1: every per-call table from the device planner)
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
        luts, nps = port_luts(mesh_device, L)
        planner = combine_planner_prepare(
            mesh_device,
            S=S,
            K=K,
            EPC=EPC,
            A=int(os.environ.get("FC_NACC", "15")),
            L=L,
            port_luts=luts,
            n_ports=nps,
            rows_cap=rows_cap,
            cores=[(i % 8, i // 8) for i in range(int(os.environ.get("FC_PLAN_CORES", "32")))],
        )
        pl = planner(ids_t, w_t)
        ttnn.synchronize_device(mesh_device)
        # x: the gathered tokens (every chip's 320, replicated); flat row r reads x row token_index[r]
        x = ttnn.from_torch(
            (torch.randn(N * S, H) * 0.3).to(torch.bfloat16),
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
    # the arena: every worker core, zero at creation, kept for the test (the combine's state lives in it)
    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    a_rows = int(plan["arena_tiles"]) * 32
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
    depth = int(os.environ.get("FC_D", "4"))
    # (the device planner's accumulator count must be the combine's: FC_NACC accumulators then, else up to 16)
    n_acc = int(os.environ.get("FC_NACC", "15")) if DEVPLAN else 16
    common = dict(num_links=L, n_acc=n_acc, depth=depth, cells=cells, arena=arena)
    L1 = os.environ.get("FC_L1") == "1"  # y rows straight from the down cores into the combine's L1 (no y in DRAM)
    l1 = None
    if L1:
        wcores = [tuple(c) for c in plan["down"]] + ([tuple(c) for c in plan["rdn"]] if plan["rdown"] else [])
        wcrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in wcores])
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
    devplan = dict(plan=pl["plan"], table=pl["table"]) if DEVPLAN else None
    if DEVPLAN:
        assert (
            l1 is not None and int(os.environ.get("FC_NACC", "15")) <= 16
        ), "FC_DEVPLAN needs FC_L1 (<= 16 accumulators)"
    fused = sparse_combine_rf_prepare(mesh_device, S, H, rows, writers=writers, l1=l1, devplan=devplan, **common)
    if DEVPLAN:  # (the planner was built for FC_NACC accumulators: the combine must have found that many cells)
        assert len(fused.cells["accs"]) == n_acc, (
            f"combine has {len(fused.cells['accs'])} accumulators, the planner {n_acc}: set FC_NACC "
            f"({len(cells)} free cells)"
        )
    alone = sparse_combine_rf_prepare(mesh_device, S, H, rows, writers=0, **common)
    coll = fused.collector
    logger.info(
        f"{routing}: flat drisc {plan.get('drisc')} dual {plan.get('dual')} down {len(plan['down'])} cells "
        f"{len(cells)}"
    )
    logger.info(
        f"{routing}: flat writers {writers}, arena {a_rows * 64 >> 10} KB/core, combine ports "
        f"{sorted(set(fused.cells['ports'].values()))} accs {fused.cells['accs']}"
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
        )

    st = dict(l1a=[])
    if L1:  # the reference y: one flat call that still writes y (before the hand-off sets MIMO_FL_Y_L1)
        os.environ.pop("MIMO_FL_Y_L1", None)
        call(empty, next(keys), coll["counts_addr"] + 512)
        ttnn.synchronize_device(mesh_device)
        ys_ref = [ttnn.to_torch(t).reshape(n_rows, H).float() for t in ttnn.get_device_tensors(y)]
        os.environ["MIMO_FL_Y_L1"] = "1"
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
    outs = [ttnn.to_torch(t).reshape(S, H).float() for t in ttnn.get_device_tensors(out)]
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

    if os.environ.get("SPC_TPROF") == "1":  # (timeline of one fused call per chip, us from the combine's first start)
        run_fused()
        ttnn.synchronize_device(mesh_device)
        for ci, t in enumerate(ttnn.get_device_tensors(fused.tdbg)):
            d = (ttnn.to_torch(t).reshape(48, 128).to(torch.int64) & 0xFFFFFFFF).tolist()
            rdp = [r for r in d[:8] if r[0] >> 20 == 0x7E0]
            r1 = [r for r in d[:8] if r[0] >> 20 == 0x7E3]  # (L1 hand-off port readers: start, all landed, end)
            if r1:
                sdp = [r for r in d[8:16] if r[0] >> 20 == 0x7E1]
                adp = [r for r in d[16:48] if r[0] >> 20 == 0x7E2]
                t0 = min([r[1] for r in r1 + sdp + adp])
                us = lambda v: (v - t0) / 1350.0  # noqa: E731
                logger.info(
                    f"TPROF {routing} chip {ci}: port rows all landed by {max(us(r[2]) for r in r1):.0f} | port readers end "
                    f"{max(us(r[3]) for r in r1):.0f} | senders end {max(us(r[2]) for r in sdp):.0f} (slot wait "
                    f"{max(r[3] for r in sdp) / 1350:.0f}) | accs local landed {max(us(r[2]) for r in adp if r[2]):.0f} "
                    f"end {max(us(r[3]) for r in adp):.0f}"
                )
                continue
            sdp = [r for r in d[8:16] if r[0] >> 20 == 0x7E1]
            adp = [r for r in d[16:48] if r[0] >> 20 == 0x7E2]
            t0 = min([r[1] for r in rdp + sdp + adp])
            us = lambda v: (v - t0) / 1350.0  # noqa: E731
            ready = {}
            for r in rdp:
                for e in range(EPC):
                    if r[4 + e]:
                        ready[e] = max(ready.get(e, 0), r[4 + e])
            es = sorted(ready)
            q = [es[int(f * (len(es) - 1))] for f in (0, 0.25, 0.5, 0.75, 1)] if es else []
            logger.info(
                f"TPROF {routing} chip {ci}: experts ready (e: us) "
                + " ".join(f"{e}:{us(ready[e]):.0f}" for e in q)
                + f" | readers end {max(us(r[2]) for r in rdp):.0f} | senders end {max(us(r[2]) for r in sdp):.0f}"
                f" (slot wait {max(r[3] for r in sdp) / 1350:.0f}, idle {max(r[4] for r in sdp) / 1350:.0f} us max)"
                f" | accs local done {max(us(r[2]) for r in adp if r[2]):.0f} end {max(us(r[3]) for r in adp):.0f}"
            )

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

    if os.environ.get("FC_PIPE") == "1":
        t_fused = pipelined(run_fused, "fused")
        if os.environ.get("FC_DRISC_DUMP"):  # (DRISC per-section cycles of the last fused call: MIMO_FL_DRISC_PROF=1)
            ttnn._ttnn.operations.bringup.flat_drisc_dump(mesh_device)
        t_flat, t_comb = pipelined(run_flat, "flat alone"), pipelined(lambda: alone(y, out), "combine alone")
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
