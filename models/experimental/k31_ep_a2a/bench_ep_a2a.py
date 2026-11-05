# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness + latency bench for the K3.1 EP all-to-all prototype on a 1x8 row of a BH Galaxy.

python bench_ep_a2a.py --cases '[{"tokens":8,"links":2,"dtype":"fp8"}]' --payload 7232 --out r.jsonl
Timing = trace replay of n_ops back-to-back ops (same method as the ttnn baseline bench), per-op us.
"""
import argparse, json, os, sys, time

import torch
import ttnn
from loguru import logger

sys.path.insert(0, os.environ.get("TT_METAL_HOME", "/home/container_app_user/tt-metal"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conftest  # noqa: E402
from ep_a2a import EpA2A, META_BYTES  # noqa: E402

H = 7168


def time_trace(mesh, fn, n_ops, n_replays):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(n_ops):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(2):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    s = []
    for _ in range(n_replays):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        s.append((time.perf_counter() - t0) / n_ops * 1e6)
    ttnn.release_trace(mesh, tid)
    s.sort()
    return {"us_min": round(s[0], 2), "us_med": round(s[len(s) // 2], 2), "us_max": round(s[-1], 2)}


def gen_meta(N, T, K, E, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]) for _ in range(N)])  # N,T,K
    w = torch.softmax(torch.randn(N, T, K, generator=g), dim=-1).to(torch.bfloat16)
    raw = torch.cat([ids.to(torch.int16).view(torch.uint8).reshape(N, T, 2 * K), w.view(torch.uint8).reshape(N, T, 2 * K)], -1)
    assert raw.shape[-1] == META_BYTES
    return ids, w, raw.view(torch.int32).reshape(N * T, META_BYTES // 4)


def quant_fp8_rows(a):
    """per-token-scaled e4m3: row = 7168 B fp8 + fp32 scale + pad to 7232 B (64 B aligned)."""
    amax = a.float().abs().amax(-1, keepdim=True).clamp(min=1e-6)
    scale = amax / 448.0
    q = (a.float() / scale).to(torch.float8_e4m3fn)
    tail = torch.zeros(a.shape[0], 64, dtype=torch.uint8)
    tail[:, :4] = scale.view(torch.uint8).reshape(a.shape[0], 4)
    return torch.cat([q.view(torch.uint8), tail], -1)


def dequant_fp8_rows(b):
    q = b[:, :H].contiguous().view(torch.float8_e4m3fn).float()
    scale = b[:, H : H + 4].contiguous().view(torch.float32)
    return q * scale


def to_dev(mesh, t_int32):
    return ttnn.from_torch(
        t_int32, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def shards(t):
    return [ttnn.to_torch(s).view(torch.int32) for s in ttnn.get_device_tensors(t)]


def run_case(mesh, c, payload, recv_mem):
    N = mesh.shape[1]
    T, K, L = c["tokens"], c.get("K", 16), c["links"]
    EPD = c.get("epd", 28)
    E = N * EPD
    dtype = c.get("dtype", "fp8")
    Rd = 7232 if dtype == "fp8" else 2 * H
    Rc = 2 * H
    rec = dict(c, payload=payload, recv_mem=recv_mem, Rd=Rd, Rc=Rc)
    mem = ttnn.L1_MEMORY_CONFIG if recv_mem == "l1" else ttnn.DRAM_MEMORY_CONFIG
    ep = EpA2A(mesh, N, T, K, EPD, Rd, Rc, links=L, payload=payload, recv_mem=mem,
               depth=c.get("depth", 4), batch=c.get("batch", 2), ring=c.get("ring", False), flush_every=c.get("flush", 1))
    torch.manual_seed(1234)
    ids, w, meta_raw = gen_meta(N, T, K, E, 1234)
    act = (torch.randn(N * T, H) * 2).to(torch.bfloat16)
    if dtype == "fp8":
        xb = quant_fp8_rows(act)
        rec["fp8_pcc"] = round(float(torch.corrcoef(torch.stack([dequant_fp8_rows(xb).flatten(), act.float().flatten()]))[0, 1]), 5)
    else:
        xb = act.view(torch.uint8).reshape(N * T, 2 * H)
    x_dev = to_dev(mesh, xb.view(torch.int32))
    meta_dev = to_dev(mesh, meta_raw)
    yf = torch.randn(N * N * T, Rc // 2).to(torch.bfloat16)  # device d: rows [d*N*T, (d+1)*N*T) = its partials
    yb = yf.view(torch.int32)
    y_dev = to_dev(mesh, yb) if c.get("y_mem", "dram") == "dram" else ttnn.from_torch(
        yb, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    t0 = time.time()
    dpd = ep.dispatch_program(x_dev, meta_dev)
    cpd = ep.combine_program(y_dev)
    rec["build_s"] = round(time.time() - t0, 2)

    # ---- correctness: dispatch
    ep.dispatch(x_dev, meta_dev, dpd)
    ttnn.synchronize_device(mesh)
    rx, rm = shards(ep.recv_x), shards(ep.recv_meta)
    if os.environ.get("EP_REREAD"):
        for _ in range(3):
            _ = ttnn.to_torch(ttnn.get_device_tensors(meta_dev)[0])
        tmp = ttnn.add(ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)), 1.0)
        ttnn.synchronize_device(mesh)
        rx2, rm2 = shards(ep.recv_x), shards(ep.recv_meta)
        rec["reread_changed"] = [not torch.equal(a_, b_) for a_, b_ in zip(rx + rm, rx2 + rm2)]
        rx, rm = rx2, rm2
    xw = xb.view(torch.int32).reshape(N, T, Rd // 4)
    mw = meta_raw.reshape(N, T, 16)
    hit = torch.zeros(N, T, N, dtype=torch.bool)  # [src, t, dst]
    for s in range(N):
        for t in range(T):
            for e in ids[s, t].tolist():
                hit[s, t, e // EPD] = True
    bad, dbg = 0, []

    def note(kind, r, s, t, got, exp):
        if len(dbg) < 12:
            eq = (got == exp).float().mean().item()
            dbg.append(f"{kind} r{r} s{s} t{t} match={eq:.3f} zeros={(got == 0).float().mean().item():.3f}")

    for r in range(N):
        for s in range(N):
            g = rm[r].reshape(N, T, 16)[s]
            if not torch.equal(g, mw[s]):
                bad += 1
                note("meta", r, s, -1, g, mw[s])
            for t in range(T):
                g = rx[r].reshape(N, T, Rd // 4)[s, t]
                if hit[s, t, r] and not torch.equal(g, xw[s, t]):
                    bad += 1
                    note("x", r, s, t, g, xw[s, t])
    rec["dispatch_ok"] = bad == 0
    rec["dispatch_bad"] = bad
    rec["rows_per_dev_avg"] = round(float(hit.sum()) / N, 2)

    # ---- correctness: combine (expert side y; source receives comb)
    ep.combine(y_dev, cpd)
    ttnn.synchronize_device(mesh)
    cb = shards(ep.comb)
    yw = yb.reshape(N, N, T, Rc // 4)  # [device d, src s, t]
    bad = 0
    for s in range(N):
        cs = cb[s].reshape(N, T, Rc // 4)
        for d in range(N):
            for t in range(T):
                if hit[s, t, d] and not torch.equal(cs[d, t], yw[d, s, t]):
                    bad += 1
                    note("comb", s, d, t, cs[d, t], yw[d, s, t])
    rec["combine_ok"] = bad == 0
    rec["combine_bad"] = bad
    rec["dbg"] = dbg

    # ---- correctness: reduction at the source (fp32 accumulate) vs torch
    out_dev = ttnn.allocate_tensor_on_device(ttnn.Shape([T, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, mem)
    rpd = ep.reduce_program(meta_dev, out_dev, max_cores=c.get("rcores", 64))
    ep.reduce(meta_dev, out_dev, rpd)
    ttnn.synchronize_device(mesh)
    outs = [ttnn.to_torch(s_).float() for s_ in ttnn.get_device_tensors(out_dev)]
    yfw = yf.float().reshape(N, N, T, H)
    pccs, maxerr = [], 0.0
    for s in range(N):
        ref = torch.zeros(T, H)
        for d in range(N):
            ref += yfw[d, s] * hit[s, :, d].float()[:, None]
        pccs.append(float(torch.corrcoef(torch.stack([outs[s].flatten(), ref.flatten()]))[0, 1]))
        maxerr = max(maxerr, float((outs[s] - ref).abs().max()))
    rec["reduce_pcc_min"] = round(min(pccs), 6)
    rec["reduce_maxabs"] = round(maxerr, 4)

    n_ops, n_rep = c.get("n_ops", 20), c.get("n_replays", 10)
    rec["dispatch"] = time_trace(mesh, lambda: ep.dispatch(x_dev, meta_dev, dpd), n_ops, n_rep)
    rec["combine"] = time_trace(mesh, lambda: ep.combine(y_dev, cpd), n_ops, n_rep)

    def pair():
        ep.dispatch(x_dev, meta_dev, dpd)
        ep.combine(y_dev, cpd)

    rec["pair"] = time_trace(mesh, pair, n_ops // 2, n_rep)
    rec["reduce"] = time_trace(mesh, lambda: ep.reduce(meta_dev, out_dev, rpd), n_ops, n_rep)

    def full():
        ep.dispatch(x_dev, meta_dev, dpd)
        ep.combine(y_dev, cpd)
        ep.reduce(meta_dev, out_dev, rpd)

    rec["full"] = time_trace(mesh, full, n_ops // 2, n_rep)
    for t in (x_dev, meta_dev, y_dev, out_dev):
        ttnn.deallocate(t)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--payload", type=int, default=0, help="fabric max packet payload bytes (0 = default 4352)")
    ap.add_argument("--recv", default="l1", choices=["l1", "dram"])
    ap.add_argument("--fabric", default="1d", choices=["1d", "1d_ring"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = json.loads(a.cases) if a.cases.strip().startswith("[") else json.load(open(a.cases))
    params = {
        "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D if a.fabric == "1d" else ttnn.FabricConfig.FABRIC_1D_RING,
        "trace_region_size": 8_000_000,
    }
    if a.payload:
        rc = ttnn.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = a.payload
        params["fabric_router_config"] = rc
    with conftest.bh_2d_mesh_device_context(params) as full:
        mesh = full.create_submesh(ttnn.MeshShape(1, 8))
        payload = ttnn.get_tt_fabric_max_payload_size_bytes()
        logger.info(f"full {tuple(full.shape)} sub {tuple(mesh.shape)} payload {payload}")
        for c in cases:
            t0 = time.time()
            try:
                rec = run_case(mesh, c, payload, a.recv)
            except Exception as e:  # keep going, record
                rec = dict(c, error=f"{type(e).__name__}: {str(e)[:600]}")
            rec["wall_s"] = round(time.time() - t0, 1)
            rec["fabric"] = a.fabric
            logger.info(f"RESULT {json.dumps(rec)}")
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
