#!/usr/bin/env python3
"""Preprocess the AgentX corpus (semianalysisai/cc-traces-weka-062126, full 1M variant) into the compact
binary the simulator reads (sim_core.js: loadTraffic).

Per trace we build:
  * streams: stream 0 = main agent chain ('s' requests), stream k>=1 = subagent k (inner 'n' requests)
  * the AIPerf replay DAG: end-to-start delay per request, subagent spawn turn / join turn / dispatch offset
  * the prefix tree of 64-token blocks, compressed into "pieces" (see README.md, "KV residency"):
    hash_ids are prefix-chained and topologically increasing (child id > parent id), so a block id IS a trie
    node.  A piece is a maximal chain of nodes cut at branch points and at request ends, so every request
    touches whole pieces only -> exact LRU over pieces == exact LRU over 64-token pages.
  * lcp_prev (blocks shared with the previous request of the same stream) -> static-slot hit
  * lcp_best (blocks already seen anywhere in the trace, recorded order) -> infinite-cache hit

Output: <out>/traffic.bin (little-endian arrays, layout in <out>/traffic.json) + traffic.json (header + stats).
Usage (run on a cpu_only Slurm node, not the login node):
  python3 prep_traffic.py --traces /data/philei/agentx_data/062126/traces.jsonl --out /data/philei/m3_traffic_sim/data
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np


def lcp(a, b):
    m = min(len(a), len(b))
    if m == 0:
        return 0
    ne = np.nonzero(a[:m] != b[:m])[0]
    return int(ne[0]) if len(ne) else m


def process(args):
    ti, line = args
    d = json.loads(line)
    reqs = []  # (stream, idx_in_stream, t, api, blocks(np), out)
    streams = [dict(kind=0, t0=None)]
    mains = []
    groups = []
    for r in d["requests"]:
        if r["type"] == "subagent":
            inner = [q for q in r["requests"] if q.get("hash_ids")]
            if not inner:
                continue
            groups.append((r, inner))
        elif r.get("hash_ids"):
            mains.append(r)
    mains.sort(key=lambda r: r["t"])
    for k, r in enumerate(mains):
        reqs.append(
            (
                0,
                k,
                float(r["t"]),
                float(r.get("api_time") or 0.0),
                np.asarray(r["hash_ids"], dtype=np.int64),
                int(r["out"]),
            )
        )
    main_t = np.array([r["t"] for r in mains]) if mains else np.zeros(0)
    main_end = np.array([r["t"] + float(r.get("api_time") or 0.0) for r in mains]) if mains else np.zeros(0)
    sub_meta = []
    for gi, (g, inner) in enumerate(groups):
        inner.sort(key=lambda q: q["t"])
        sid = len(streams)
        streams.append(dict(kind=1))
        gt = float(g["t"])
        # spawn turn: last main turn with t <= group t (AIPerf: last retained parent turn before the marker)
        sp = int(np.searchsorted(main_t, gt + 1e-9, side="right")) - 1
        if sp < 0:
            continue  # AIPerf drops subagents with no spawning parent turn
        if g.get("duration_ms") is not None:
            g_end = gt + float(g["duration_ms"]) / 1000.0
        else:
            g_end = max(q["t"] + float(q.get("api_time") or 0.0) for q in inner)
        # join turn: first later main turn with t + 1e-6 >= g_end; none -> background (never blocks)
        jn = int(np.searchsorted(main_t, g_end - 1e-6, side="left"))
        if jn <= sp:
            jn = sp + 1
        if jn >= len(mains):
            jn = -1
        overlap = gt < main_end[sp]
        # dispatch offset of the first inner request: from parent issue (overlap path) or parent end
        first_t = float(inner[0]["t"])
        off = (first_t - main_t[sp]) if overlap else (first_t - gt)
        sub_meta.append((sid, sp, jn, max(0.0, off), int(overlap), gt, g_end))
        for k, q in enumerate(inner):
            reqs.append(
                (
                    sid,
                    k,
                    float(q["t"]),
                    float(q.get("api_time") or 0.0),
                    np.asarray(q["hash_ids"], dtype=np.int64),
                    int(q["out"]),
                )
            )
    # keep only streams that survived (drop subagents without a spawn turn)
    kept = {0} | {m[0] for m in sub_meta}
    remap = {}
    for s in sorted(kept):
        remap[s] = len(remap)
    reqs = [r for r in reqs if r[0] in kept]
    reqs.sort(key=lambda r: (remap[r[0]], r[1]))
    n = len(reqs)
    if n == 0:
        return None
    # ---- end-to-start delay within each stream
    delay = np.zeros(n)
    for i in range(1, n):
        if reqs[i][0] == reqs[i - 1][0]:
            delay[i] = max(0.0, reqs[i][2] - reqs[i - 1][2] - reqs[i - 1][3])
    # ---- prefix tree
    maxid = max(int(r[4].max()) for r in reqs)
    parent = np.full(maxid + 1, -2, dtype=np.int64)
    is_end = np.zeros(maxid + 1, dtype=bool)
    for r in reqs:
        h = r[4]
        parent[h[0]] = -1
        parent[h[1:]] = h[:-1]
        is_end[h[-1]] = True
    present = parent != -2
    nodes = np.nonzero(present)[0]  # increasing id == topological order
    par = parent[nodes]
    child_cnt = np.bincount(par[par >= 0], minlength=maxid + 1)
    start = (par < 0) | (child_cnt[np.maximum(par, 0)] > 1) | is_end[np.maximum(par, 0)]
    start[par < 0] = True
    piece_of = np.full(maxid + 1, -1, dtype=np.int64)
    np_ = 0
    piece_parent = []
    piece_len = []
    # sequential pass (ids are topologically sorted); python loop over unique nodes only
    so = start.tolist()
    pl = par.tolist()
    nl = nodes.tolist()
    pof = piece_of  # alias
    for x, s, p in zip(nl, so, pl):
        if s:
            pof[x] = np_
            piece_parent.append(int(pof[p]) if p >= 0 else -1)
            piece_len.append(1)
            np_ += 1
        else:
            q = pof[p]
            pof[x] = q
            piece_len[q] += 1
    # ---- per-request arrays
    leaf = np.array([piece_of[r[4][-1]] for r in reqs], dtype=np.int64)
    blocks = np.array([len(r[4]) for r in reqs], dtype=np.int64)
    lcp_prev = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        if reqs[i][0] == reqs[i - 1][0]:
            lcp_prev[i] = lcp(reqs[i][4], reqs[i - 1][4])
    # infinite-cache hit in recorded (start-time) order; "seen" is prefix-closed along a path
    order = sorted(range(n), key=lambda i: (reqs[i][2], reqs[i][0], reqs[i][1]))
    seen = np.zeros(maxid + 1, dtype=bool)
    lcp_best = np.zeros(n, dtype=np.int64)
    for i in order:
        h = reqs[i][4]
        lo, hi = 0, len(h)  # first unseen index via binary search
        while lo < hi:
            mid = (lo + hi) // 2
            if seen[h[mid]]:
                lo = mid + 1
            else:
                hi = mid
        lcp_best[i] = lo
        seen[h] = True
    # path depth in pieces (cost of one residency walk)
    pp = np.array(piece_parent, dtype=np.int64)
    pdepth = np.zeros(np_, dtype=np.int64)
    for q in range(np_):
        pdepth[q] = 0 if pp[q] < 0 else pdepth[pp[q]] + 1
    t0 = min(r[2] for r in reqs)
    t1 = max(r[2] + r[3] for r in reqs)
    # stream table
    st_kind = np.zeros(len(remap), dtype=np.int64)
    st_spawn = np.full(len(remap), -1, dtype=np.int64)
    st_join = np.full(len(remap), -1, dtype=np.int64)
    st_off = np.zeros(len(remap))
    st_ovl = np.zeros(len(remap), dtype=np.int64)
    st_t0 = np.zeros(len(remap))
    st_t1 = np.zeros(len(remap))
    for sid, sp, jn, off, ovl, gt, gend in sub_meta:
        s = remap[sid]
        st_kind[s] = 1
        st_spawn[s] = sp
        st_join[s] = jn
        st_off[s] = off
        st_ovl[s] = ovl
        st_t0[s] = gt - t0
        st_t1[s] = gend - t0
    stream_first = np.zeros(len(remap), dtype=np.int64)
    stream_cnt = np.zeros(len(remap), dtype=np.int64)
    for i, r in enumerate(reqs):
        s = remap[r[0]]
        if stream_cnt[s] == 0:
            stream_first[s] = i
        stream_cnt[s] += 1
    return dict(
        ti=ti,
        req=dict(
            stream=np.array([remap[r[0]] for r in reqs]),
            idx=np.array([r[1] for r in reqs]),
            t=np.array([r[2] - t0 for r in reqs]),
            api=np.array([r[3] for r in reqs]),
            delay=delay,
            blocks=blocks,
            out=np.array([r[5] for r in reqs]),
            leaf=leaf,
            lcp_prev=lcp_prev,
            lcp_best=lcp_best,
            pathlen=pdepth[leaf] + 1,
        ),
        stream=dict(
            kind=st_kind,
            spawn=st_spawn,
            join=st_join,
            off=st_off,
            ovl=st_ovl,
            t0=st_t0,
            t1=st_t1,
            first=stream_first,
            cnt=stream_cnt,
        ),
        piece=dict(parent=pp, len=np.array(piece_len, dtype=np.int64)),
        dur=t1 - t0,
    )


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument(
        "--traces",
        default=os.environ.get("AGENTX_TRACES", "/data/philei/agentx_data/062126/traces.jsonl"),
        help="traces.jsonl of semianalysisai/cc-traces-weka-062126 (HF dataset); env AGENTX_TRACES",
    )
    ap.add_argument("--out", default=os.environ.get("M3SIM_DATA", os.path.join(here, "data")))
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    lines = []
    with open(a.traces) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            if line.strip():
                lines.append((i, line))
    print(f"read {len(lines)} traces in {time.time() - t0:.0f}s", flush=True)
    with Pool(a.procs) as p:
        res = [r for r in p.imap(process, lines, chunksize=1) if r is not None]
    del lines
    print(f"processed in {time.time() - t0:.0f}s", flush=True)
    # ---- concatenate with global offsets
    R, S, P = [], [], []
    tr_req0, tr_st0, tr_pc0, tr_dur = [], [], [], []
    nr = ns = npc = 0
    for r in res:
        tr_req0.append(nr)
        tr_st0.append(ns)
        tr_pc0.append(npc)
        tr_dur.append(r["dur"])
        q = dict(r["req"])
        q["leaf"] = q["leaf"] + npc
        s = dict(r["stream"])
        s["spawn"] = np.where(s["spawn"] >= 0, s["spawn"] + nr, -1)  # spawn/join are main-stream request idx -> global
        s["join"] = np.where(s["join"] >= 0, s["join"] + nr, -1)
        s["first"] = s["first"] + nr
        q["trace"] = np.full(len(q["blocks"]), len(tr_req0) - 1)
        q["stream"] = q["stream"] + ns
        pc = dict(r["piece"])
        pc["parent"] = np.where(pc["parent"] >= 0, pc["parent"] + npc, -1)
        R.append(q)
        S.append(s)
        P.append(pc)
        nr += len(q["blocks"])
        ns += len(s["kind"])
        npc += len(pc["len"])
    cat = lambda L, k: np.concatenate([x[k] for x in L])
    arrays = [
        # name, dtype, data
        ("req_trace", "u2", cat(R, "trace")),
        ("req_stream", "u4", cat(R, "stream")),
        ("req_idx", "u2", cat(R, "idx")),
        ("req_t", "f4", cat(R, "t")),
        ("req_api", "f4", cat(R, "api")),
        ("req_delay", "f4", cat(R, "delay")),
        ("req_blocks", "u2", cat(R, "blocks")),
        ("req_out", "u4", cat(R, "out")),
        ("req_leaf", "u4", cat(R, "leaf")),
        ("req_lcp_prev", "u2", cat(R, "lcp_prev")),
        ("req_lcp_best", "u2", cat(R, "lcp_best")),
        ("st_kind", "u1", cat(S, "kind")),
        ("st_spawn", "i4", cat(S, "spawn")),
        ("st_join", "i4", cat(S, "join")),
        ("st_off", "f4", cat(S, "off")),
        ("st_ovl", "u1", cat(S, "ovl")),
        ("st_t0", "f4", cat(S, "t0")),
        ("st_t1", "f4", cat(S, "t1")),
        ("st_first", "u4", cat(S, "first")),
        ("st_cnt", "u2", cat(S, "cnt")),
        ("pc_parent", "i4", cat(P, "parent")),
        ("pc_len", "u2", cat(P, "len")),
        ("tr_req0", "u4", np.array(tr_req0)),
        ("tr_st0", "u4", np.array(tr_st0)),
        ("tr_pc0", "u4", np.array(tr_pc0)),
        ("tr_dur", "f4", np.array(tr_dur)),
    ]
    for name, dt, x in arrays:
        if dt[0] in "ui":
            info = np.iinfo(np.dtype("<" + dt))
            assert x.min() >= info.min and x.max() <= info.max, (name, x.min(), x.max())
    os.makedirs(a.out, exist_ok=True)
    layout = []
    off = 0
    with open(os.path.join(a.out, "traffic.bin"), "wb") as f:
        for name, dt, x in arrays:
            b = np.ascontiguousarray(x.astype("<" + dt)).tobytes()
            pad = (-off) % 8
            f.write(b"\0" * pad)
            off += pad
            layout.append(dict(name=name, dtype=dt, offset=off, n=int(len(x))))
            f.write(b)
            off += len(b)
    blocks = cat(R, "blocks")
    lb = cat(R, "lcp_best")
    lp = cat(R, "lcp_prev")
    pl = cat(R, "pathlen")
    stats = dict(
        traces=len(res),
        requests=int(nr),
        streams=int(ns),
        pieces=int(npc),
        input_tokens=int(blocks.sum() * 64),
        new_tokens_inf=int((blocks - lb).sum() * 64),
        inf_hit_rate=float(lb.sum() / blocks.sum()),
        prev_stream_hit_rate=float(lp.sum() / blocks.sum()),
        pathlen_mean=float(pl.mean()),
        pathlen_p99=float(np.percentile(pl, 99)),
        pathlen_max=int(pl.max()),
        pathlen_sum=int(pl.sum()),
        bytes=off,
    )
    json.dump(
        dict(layout=layout, stats=stats, source=a.traces, block=64),
        open(os.path.join(a.out, "traffic.json"), "w"),
        indent=1,
    )
    print(json.dumps(stats, indent=1), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
