# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the sparse KV push's planning in the 2D dispatch's style (combine_rf/kv_push_plan.py), GLM-5.3 at a
1,044,480-row prefix, chunk 5120, top-k 2048.

test_kv_plan_chip      one chip as chip ME of the 8x4 Galaxy (SP8 x TP4: 160 queries, stripes of 160 rows): step 1
                       (top-k -> want bitmaps) and step 3 (bitmaps -> masks + order / link) timed and checked
                       exactly; step 3's input is every requester's slice as the exchange would deliver it.
test_kv_plan_loudbox   the whole chain on the LoudBox as a 2x4 FABRIC_2D mesh (SP2 x TP4: 640 queries per chip,
                       stripes of 640): step 1 on every chip, step 2 (sparse_all_gather_2d, each 4 KB page to its owner),
                       step 3; exact on every chip, slowest chip per step.
Top-k is synthetic with Iva's 1M statistics (real trace not on this box): per Galaxy chip ~30k distinct rows, a row
wanted by ~5.5 chips; the LoudBox chips are groups of 4 Galaxy chips. KVP_DISTINCT / KVP_MULT / KVP_RUN set them.

Run: TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
     scripts/run_safe_pytest.sh --mesh <this file> -s -k loudbox   (or --device auto -k chip)
"""
import os

import numpy as np
import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.kv_push_plan import (
    exchange_route_words,
    geometry,
    route_out_shape,
    route_program,
    want_program,
    want_split,
)
from ttnn.bringup.combine_rf.sparse_all_gather_2d import (
    ensure_semaphores,
    hbit,
    manifest_spec,
    span_table,
    sparse_all_gather_2d,
)

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

P, CHUNK, K = 1044480, 5120, 2048
SENTINEL = 0xFFFFFFFF
_KEY = "DEVICE KERNEL DURATION [ns]"


# ---- synthetic top-k (32 Galaxy chips; a LoudBox chip = a group of 4)
def make_topk(rng, groups, q_per_chip):
    nv, D = 32, int(os.environ.get("KVP_DISTINCT", "30000"))
    mult, run = float(os.environ.get("KVP_MULT", "5.5")), int(os.environ.get("KVP_RUN", "16"))
    u = int(nv * D / mult)
    starts = rng.integers(0, P, size=2 * u // run + 16)
    lens = rng.geometric(1 / run, size=len(starts))
    pool = np.unique(np.concatenate([np.arange(s, min(P, s + l)) for s, l in zip(starts, lens)]))
    pool = pool[rng.permutation(len(pool))[:u]]
    w = rng.lognormal(0.0, 1.0, size=len(pool))
    lo, hi = 0.0, 1e3
    for _ in range(60):  # mean inclusion -> D distinct rows per Galaxy chip
        c = (lo + hi) / 2
        lo, hi = (c, hi) if np.minimum(1, c * w).sum() < D else (lo, c)
    p = np.minimum(1, c * w)
    sets = [pool[rng.random(len(pool)) < p] for _ in range(nv)]
    out = []
    for g in range(nv // groups):
        s = np.unique(np.concatenate(sets[g * groups : (g + 1) * groups]))
        idx = s[rng.integers(0, len(s), size=(q_per_chip, K))].astype(np.uint64)
        idx[rng.random(idx.shape) < 0.005] = SENTINEL
        out.append(idx.astype(np.uint32))
    return np.stack(out)  # [N, Q, K]


# ---- host references
def owner_local(n, SC, N):
    blk = n // SC
    return blk % N, (blk // N) * SC + n % SC


def want_ref(idx, N, SC):
    """[N owners, WOP] uint32 bitmap of one requester."""
    _, wop, _, _ = geometry(N, P, SC)
    rows = np.unique(idx[idx < P]).astype(np.int64)
    o, lr = owner_local(rows, SC, N)
    bm = np.zeros((N, wop), dtype=np.uint32)
    np.bitwise_or.at(bm, (o, lr >> 5), np.left_shift(np.uint32(1), (lr & 31).astype(np.uint32)))
    return bm


def route_ref(wants, me, R, C, torus):
    """Owner me's sorted (row, mask, flags) from every requester's bitmap (wants [N, N, WOP])."""
    N = R * C
    wx, wy = bool(torus[0]) and C > 2, bool(torus[1]) and R > 2
    tx, ty = span_table(C, wx), span_table(R, wy)
    bits = np.unpackbits(wants[:, me, :].view(np.uint8), bitorder="little").reshape(N, -1).astype(np.uint64)
    m = (bits << np.arange(N, dtype=np.uint64)[:, None]).sum(axis=0) & ~np.uint64(1 << me)
    r0, c0 = me // C, me % C
    lk = lambda s: (s & 15) + (s >> 4)  # noqa: E731
    out = []
    for row in np.nonzero(m)[0]:
        mm, row = int(m[row]), int(row)
        tie = hbit(row, 30)
        xs = lambda pos, tm: tx[(pos * (1 << C) + tm) * 2 + tie]  # noqa: E731
        ys = lambda pos, tm: ty[(pos * (1 << R) + tm) * 2 + tie]  # noqa: E731
        cir = lambda r: (mm >> (r * C)) & ((1 << C) - 1)  # noqa: E731
        ric = lambda c: sum(((mm >> (r * C + c)) & 1) << r for r in range(R))  # noqa: E731
        cm = 0
        for r in range(R):
            cm |= cir(r)
        rm = sum(1 << r for r in range(R) if cir(r))
        hx, hy = lk(xs(c0, cm & ~(1 << c0))), lk(ys(r0, rm & ~(1 << r0)))
        for c in range(C):
            if ((cm | (1 << c0)) >> c) & 1:
                hx += lk(ys(r0, ric(c) & ~(1 << r0)))
        for r in range(R):
            if ((rm | (1 << r0)) >> r) & 1:
                hy += lk(xs(c0, cir(r) & ~(1 << c0)))
        h = (row * 2654435761) & 0xFFFFFFFF
        order = 0 if hx < hy else (1 if hy < hx else (h >> 31) & 1)
        out.append((row, mm, order | (((h >> 29) & 1) << 1)))
    return out


def decode_routes(o):
    """[n_cores, 16 + ...] uint32 -> sorted (row, mask, flags)."""
    got = []
    for i in range(o.shape[0]):
        n = int(o[i, 0])
        e = o[i, 16 : 16 + 2 * n].reshape(n, 2)
        got += [(int(w & 0xFFFFFF), int(m), int(w >> 24)) for m, w in e]
    return sorted(got)


# ---- device helpers
def _times(dev):
    ttnn.ReadDeviceProfiler(dev)
    rows = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        rows.append(
            [
                float(p.program_analyses_results[_KEY].duration) / 1e3
                for p in programs
                if _KEY in (p.program_analyses_results or {})
            ]
        )
    return rows


def timed(dev, fn, n=5):
    """Min over n calls of the slowest chip's duration of each program fn launches."""
    _times(dev)
    best = None
    for _ in range(n):
        fn()
        ttnn.synchronize_device(dev)
        rows = [r for r in _times(dev) if r]
        k = min(len(r) for r in rows)
        t = [max(r[-k + i] for r in rows) for i in range(k)]
        best = t if best is None or sum(t) < sum(best) else best
    return best


def mesh_desc(dev, build):
    R, C = tuple(dev.shape)
    md = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(C):
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = build(r * C + c)
    return md


def u32(dev, a, mapper=None):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(a).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def host(dev, t):
    R, C = tuple(dev.shape)
    if R * C == 1:
        return ttnn.to_torch(t).numpy().view(np.uint32)
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0)).numpy().view(np.uint32)


# ---- one chip as chip ME of the 8x4 Galaxy
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kv_plan_chip(mesh_device):
    R, C, ME, torus = 8, 4, 13, (True, True)
    N, Q = R * C, CHUNK // (R * C)
    SC = CHUNK // 8 // 4  # SP 8 x TP 4: 640 local rows / TP
    owned, wop, ppo, S = geometry(N, P, SC)
    rng = np.random.default_rng(0)
    idx = make_topk(rng, 1, Q)
    wants = np.stack([want_ref(idx[q], N, SC) for q in range(N)])  # [requester, owner, WOP]
    dist = [int(np.unique(idx[q][idx[q] < P]).size) for q in range(N)]
    logger.info(
        f"KVPLAN 8x4 chip {ME}: distinct rows per chip min / median / max {min(dist)} / {int(np.median(dist))} / {max(dist)}"
    )

    xbuf = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, N * S, 1024]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    tt_idx = u32(mesh_device, idx[ME].reshape(1, 1, Q, K))
    # step 1
    for n_prod, unroll in [
        (int(x), int(u))
        for x in os.environ.get("KVP_PROD", "64,80,120").split(",")
        for u in os.environ.get("KVP_UNROLL", "1").split(",")
    ]:
        status = u32(mesh_device, np.zeros((1, 1, 2 * n_prod, 16), dtype=np.uint32))
        md = mesh_desc(
            mesh_device,
            lambda me: want_program(
                mesh_device,
                R=R,
                C=C,
                me=ME,
                P=P,
                SC=SC,
                indices=tt_idx,
                xbuf=xbuf,
                status=status,
                n_prod=n_prod,
                unroll=unroll,
            ),
        )
        t = timed(mesh_device, lambda: ttnn.generic_op([tt_idx, xbuf, status], md))
        x = host(mesh_device, xbuf).reshape(N * S, 1024)
        got = x[ME * S : ME * S + N * ppo].reshape(N, ppo * 1024)[:, :wop]
        stv = host(mesh_device, status).reshape(2 * n_prod, 16).astype(np.int64)
        assert int(stv[:, 0].sum()) == 0, "inbox overflow"
        assert np.array_equal(got, wants[ME]), f"step 1 mismatch (n_prod {n_prod})"
        t0 = stv[:, 1].min()
        nc = want_split(n_prod)[1]
        ph = (
            lambda k: (stv[: 2 * nc if k >= 5 else 2 * n_prod, k] - t0) / 1350.0
        )  # noqa: E731 (cycles -> us at 1.35 GHz)
        logger.info(
            "KVPLAN 8x4 step1 phases (us from first start, median / max): "
            + "  ".join(
                f"{nm} {np.median(ph(k)):.1f}/{ph(k).max():.1f}"
                for k, nm in ((2, "read"), (3, "bucket"), (4, "sent"), (5, "all in"), (6, "slice out"))
            )
        )
        logger.info(f"KVPLAN 8x4 step1 want  producers {n_prod:3d} unroll {unroll}: {t[0]:7.1f} us (exact)")
    # step 3 on what the exchange delivers
    xin = np.zeros((N * S, 1024), dtype=np.uint32)
    for q in range(N):
        xin[q * S + ME * ppo : q * S + (ME + 1) * ppo] = np.pad(wants[q, ME], (0, ppo * 1024 - wop)).reshape(ppo, 1024)
    xbuf2 = u32(mesh_device, xin.reshape(1, 1, N * S, 1024))
    ref = route_ref(wants, ME, R, C, torus)
    logger.info(
        f"KVPLAN 8x4 owner {ME}: {len(ref)} rows to push of {owned}, mean requesters {np.mean([bin(m).count('1') for _, m, _ in ref]):.2f}"
    )
    for n_cores in [int(x) for x in os.environ.get("KVP_ROUTE_CORES", "16,32,64").split(",")]:
        out = u32(mesh_device, np.zeros(route_out_shape(N, P, SC, n_cores), dtype=np.uint32))
        ts = []
        for probe in (0, 2):
            md = mesh_desc(
                mesh_device,
                lambda me: route_program(
                    mesh_device,
                    R=R,
                    C=C,
                    me=ME,
                    torus=torus,
                    P=P,
                    SC=SC,
                    xbuf=xbuf2,
                    out=out,
                    n_cores=n_cores,
                    probe=probe,
                ),
            )
            ts.append(timed(mesh_device, lambda: ttnn.generic_op([xbuf2, out], md))[0])
            if probe == 0:
                ov = host(mesh_device, out).reshape(n_cores, -1)
                assert decode_routes(ov) == ref, f"step 3 mismatch ({n_cores} cores)"
                logger.info(f"KVPLAN 8x4 step3 rows per core: mean {ov[:, 0].mean():.0f} max {ov[:, 0].max()}")
        logger.info(f"KVPLAN 8x4 step3 route cores {n_cores:3d}: {ts[0]:7.1f} us (exact), masks only {ts[1]:7.1f} us")


# ---- the whole chain on the LoudBox as 2x4
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((2, 4), fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0), id="fabric-2d-mesh-2x4")],
    indirect=["mesh_device", "device_params"],
)
def test_kv_plan_loudbox(mesh_device, device_params):
    R, C = tuple(mesh_device.shape)
    torus = (False, False)
    N = R * C
    Q, SC = CHUNK // N, CHUNK // R // 4  # SP R x TP 4
    owned, wop, ppo, S = geometry(N, P, SC)
    rng = np.random.default_rng(1)
    idx = make_topk(rng, 32 // N, Q)
    wants = np.stack([want_ref(idx[q], N, SC) for q in range(N)])
    dist = [int(np.unique(idx[q][idx[q] < P]).size) for q in range(N)]
    logger.info(
        f"KVPLAN 2x4: {Q} queries per chip, stripes {SC}, distinct rows per chip min / median / max {min(dist)} / {int(np.median(dist))} / {max(dist)}"
    )
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    ensure_semaphores(mesh_device, num_links=2, torus=torus)
    tt_idx = u32(mesh_device, idx.reshape(N, 1, Q, K), shard)
    xbuf = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, N * S, 1024]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    n_prod = int(os.environ.get("KVP_PROD", "64").split(",")[0])
    n_route = int(os.environ.get("KVP_ROUTE_CORES", "64").split(",")[-1])
    status = u32(mesh_device, np.zeros((N, 1, 2 * n_prod, 16), dtype=np.uint32), shard)
    plan = ttnn.from_torch(exchange_route_words(R, C, P, SC).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard)  # fmt: skip
    man = ttnn.allocate_tensor_on_device(*manifest_spec(mesh_device, S, num_links=2, torus=torus))
    out = u32(mesh_device, np.zeros([N] + route_out_shape(N, P, SC, n_route)[1:], dtype=np.uint32), shard)
    md1 = mesh_desc(
        mesh_device,
        lambda me: want_program(
            mesh_device, R=R, C=C, me=me, P=P, SC=SC, indices=tt_idx, xbuf=xbuf, status=status, n_prod=n_prod
        ),
    )
    md3 = mesh_desc(
        mesh_device,
        lambda me: route_program(
            mesh_device, R=R, C=C, me=me, torus=torus, P=P, SC=SC, xbuf=xbuf, out=out, n_cores=n_route
        ),
    )

    def chain():
        ttnn.generic_op([tt_idx, xbuf, status], md1)
        sparse_all_gather_2d(xbuf, plan, num_links=2, manifest=man, torus=torus)
        ttnn.generic_op([xbuf, out], md3)

    t = timed(mesh_device, chain)
    assert int(host(mesh_device, status)[..., 0].sum()) == 0, "inbox overflow"
    x = host(mesh_device, xbuf).reshape(N, N * S, 1024)
    o = host(mesh_device, out).reshape(N, n_route, -1)
    for me in range(N):
        for q in range(N):  # every requester's slice of my rows arrived (and my own bitmap is in place)
            got = x[me, q * S + me * ppo : q * S + (me + 1) * ppo].reshape(-1)[:wop]
            assert np.array_equal(got, wants[q, me]), f"exchange: owner {me}, requester {q}"
        ref = route_ref(wants, me, R, C, torus)
        assert decode_routes(o[me]) == ref, f"step 3 mismatch on owner {me}"
        logger.info(
            f"KVPLAN 2x4 owner {me}: {len(ref)} rows to push of {owned}, mean requesters {np.mean([bin(m).count('1') for _, m, _ in ref]):.2f}"
        )
    names = ["want", "exchange", "route"] if len(t) == 3 else [f"p{i}" for i in range(len(t))]
    logger.info(
        "KVPLAN 2x4 chain (slowest chip, us, exact): "
        + "  ".join(f"{n} {v:.1f}" for n, v in zip(names, t))
        + f"  total {sum(t):.1f}"
    )
