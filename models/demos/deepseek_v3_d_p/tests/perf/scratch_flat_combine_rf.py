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
    chips = [(r, c) for r in range(R) for c in range(C)]
    pend = {ch: [] for ch in chips}
    load = {}
    for o in range(N):
        for t in range(S):
            for k in range(K):
                e = int(ids[o, t, k])
                src = chips[e // EPC]
                link = (t + k) % L
                pend[src].append((e % EPC, o, t, link))
                for hc, d in route(src, chips[o], R, C, False, False, link):
                    load[(hc, d, link)] = load.get((hc, d, link), 0) + 1
    rows, counts, regions, n_rows = (
        {},
        torch.zeros(N, 1, E, dtype=torch.int32),
        torch.zeros(N, 1, E, dtype=torch.int32),
        0,
    )
    for ci, ch in enumerate(chips):
        lst = sorted(pend[ch])
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
        for e, o, t, link in lst:
            rows[ch].append((reg[e] + seen[e], chips[o], t, link, e))
            seen[e] += 1
        counts[ci, 0, ci * EPC : (ci + 1) * EPC] = torch.tensor(cnt, dtype=torch.int32)
        regions[ci, 0, ci * EPC : (ci + 1) * EPC] = torch.tensor(reg, dtype=torch.int32)
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
    taken = {tuple(c) for k in ("arena_cores", "gu", "down", "readers", "relays", "coords") for c in plan[k]}
    cells = CELLS or [(x_, y_) for y_ in range(grid0.y) for x_ in range(grid0.x) if (x_, y_) not in taken]
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    rm = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=dram, mesh_mapper=shard
    )
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
    common = dict(num_links=L, n_acc=16, depth=depth, cells=cells, arena=arena)
    fused = sparse_combine_rf_prepare(mesh_device, S, H, rows, writers=writers, **common)
    alone = sparse_combine_rf_prepare(mesh_device, S, H, rows, writers=0, **common)
    coll = fused.collector
    logger.info(
        f"{routing}: flat writers {writers}, arena {a_rows * 64 >> 10} KB/core, combine ports "
        f"{sorted(set(fused.cells['ports'].values()))} accs {fused.cells['accs']}"
    )
    desc = fused.build(y, out)
    empty = ttnn.MeshProgramDescriptor()
    for r_ in range(R):
        for c_ in range(C):
            empty[
                ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r_, c_), ttnn.MeshCoordinate(r_, c_))
            ] = ttnn.ProgramDescriptor()
    keys = itertools.count(1)
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
            down_fp32=True,
        )

    run_fused = lambda: call(desc, k_fused, coll["counts_addr"])  # noqa: E731
    run_flat = lambda: call(
        empty, k_flat, coll["counts_addr"] + 512
    )  # noqa: E731 (dummy counts: the combine never reads them)

    # first fused call: the combine's sums against the y rows the flat expert wrote
    run_fused()
    ttnn.synchronize_device(mesh_device)
    ys = [ttnn.to_torch(t).reshape(n_rows, H).float() for t in ttnn.get_device_tensors(y)]
    outs = [ttnn.to_torch(t).reshape(S, H).float() for t in ttnn.get_device_tensors(out)]
    chips = [(r_, c_) for r_ in range(R) for c_ in range(C)]
    ref = torch.zeros(N, S, H)
    for ci, ch in enumerate(chips):
        for yrow, home, t, _, _ in rows[ch]:
            ref[chips.index(home), t] += ys[ci][yrow]
    fin = torch.isfinite(ref).all(2)
    worst = 0.0
    for o in range(N):
        f = fin[o]
        if f.any():
            d = (outs[o][f] - ref[o][f]).norm(dim=1) / ref[o][f].norm(dim=1).clamp(min=1e-30)
            worst = max(worst, d.max().item())
    logger.info(f"{routing}: fused first call, worst per-token rel err {worst:.2e} over {int(fin.sum())} finite tokens")
    assert worst < 2e-2, f"{routing}: combine result off ({worst:.2e})"

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

    t_fused = timed(run_fused, "fused")
    t_flat = timed(run_flat, "flat alone")
    t_comb = timed(lambda: alone(y, out), "combine alone")
    logger.info(
        f"FC_SUMMARY {routing:16s} E{EPC}/chip K{K} S{S} H{H}: flat {t_flat:.0f} + combine {t_comb:.0f} = "
        f"{t_flat + t_comb:.0f} serial vs fused {t_fused:.0f} us ({100 * (t_flat + t_comb - t_fused) / t_comb:.0f}% "
        f"of the combine hidden; fused / flat {t_fused / t_flat:.3f}) | combine link floor {floor:.0f} us"
    )
