"""verify-rows (S = k+1 query rows per user) decode attention block on a 1x8 row: split kernel + exchange/merge vs the torch full softmax over the selected ids (seqshard / dk_dev ref_sparse), SCALED_FP8 pool.
Run: program/bin/vr.sh <tag> tests.test_hy4_vrows_device TEST=pcc VAR=a1|stock|xm SS=1,2,3,4,5,6,7,8 [CASES=random_65536,...] [REPS=3]
Case tags: <kind>_<ctx>[_skew] kind random|real|hot; ctx 100 = columns 2..7 empty; skew = every selected key on column 0 (all 2048 on one column: 8 chunks of 256).
VAR=xm (XMerge, ONE object per process): use SS with a single S per process (or the object is freed between S)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
import test_hy4_decode_merge_device as DM
from dk_dev import pcc, timed_traced
from dm_block import close_row, open_row

import ttnn

emit = DM.emit
CASES = [
    "random_65536",
    "real_65536",
    "hot_65536",
    "random_100",
    "real_3000",
    "random_65536_skew",
    "real_65536_skew",
    "random_1048576",
]


def parse(tag):
    p = tag.split("_")
    return p[0], int(p[1]), len(p) > 2 and p[2] == "skew"


def free_xm(blk):
    xm = blk.xm
    if xm is not None:
        for t in (xm.land, xm.out):
            ttnn.deallocate(t)
        blk.xm = None


def cmp_run(md, blk, c, S):
    """compact path (PoolCtx.attend compact=True): split kernel with the 4 users per tile row packing -> a2a -> A1 merge on TR tile rows. -> host heads [64,S,512]"""
    import hy4_merge

    TR = (S + 3) // 4
    qg = blk.gather_q(c.q_dev)
    idx8 = blk.idx_rows(c.idx_dev)
    out = blk.mod.sparse_sdpa_split(
        qg, c.kv_dev, idx8, 512, DM.SCALE, blk.kc, blk.G, blk.ckc, None, emit_stats=3, owner_s=S, compact=True
    )
    x = ttnn.experimental.all_to_all_async_generic(
        out, in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
    )
    sk = torch.zeros(8, TR, 32, 1)
    for cc in range(8):
        for b in range(4):
            sk[cc, :, 8 * b : 8 * b + 8, 0] = c.sl[8 * cc : 8 * cc + 8][None, :]
    skd = DM.to_mesh(md, sk, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    res = hy4_merge.merge_fused(x, skd, TR, full=S > 2)
    r = (
        DM.from_mesh(md, res).float().reshape(8, TR * 4, 32 // 4, 4, 512)
        if False
        else DM.from_mesh(md, res).float().reshape(8, TR, 32, 512)
    )
    heads = torch.zeros(8, 8, S, 512)
    for s in range(S):
        u, b = s // 4, s % 4
        heads[:, :, s] = r[:, u, 8 * b : 8 * b + 8]
    return heads.reshape(64, S, 512)


_QPU = {}


def qp_obj(md, S):
    from hy4_qpush import QPush

    if S not in _QPU:
        _QPU[S] = QPush(md, S)
    return _QPU[S]


def qpp_run(md, blk, c, S):
    """B29: compact path with the q push program + flag-gated q read in the split reader (a2a + A1 merge as in cmp_run) -> host heads [64,S,512]"""
    import hy4_merge

    TR = (S + 3) // 4
    qp = qp_obj(md, S)
    land = qp(c.q_dev)
    idx8 = blk.idx_rows(c.idx_dev)
    out = blk.mod.sparse_sdpa_split(
        land,
        c.kv_dev,
        idx8,
        512,
        DM.SCALE,
        blk.kc,
        blk.G,
        blk.ckc,
        None,
        emit_stats=3,
        owner_s=S,
        compact=True,
        qland=(qp, S),
    )
    x = ttnn.experimental.all_to_all_async_generic(
        out, in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
    )
    sk = torch.zeros(8, TR, 32, 1)
    for cc in range(8):
        for b in range(4):
            sk[cc, :, 8 * b : 8 * b + 8, 0] = c.sl[8 * cc : 8 * cc + 8][None, :]
    skd = DM.to_mesh(md, sk, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    res = hy4_merge.merge_fused(x, skd, TR, full=S > 2)
    r = DM.from_mesh(md, res).float().reshape(8, TR, 32, 512)
    heads = torch.zeros(8, 8, S, 512)
    for s in range(S):
        u, b = s // 4, s % 4
        heads[:, :, s] = r[:, u, 8 * b : 8 * b + 8]
    return heads.reshape(64, S, 512)


def run_qplat(md):
    """B29: traced latency: gather_q + split vs q push + split (the chain with a2a + A1 merge too)"""
    import hy4_merge

    for S in [int(x) for x in os.environ.get("SS", "4").split(",")]:
        c = DM.Case(md, "random", 65536, S, "scaled", seed=5)
        blk = DM.mk_block(md, S, fuse=True, xchg=False)
        TR = (S + 3) // 4
        qp = qp_obj(md, S)
        idx8 = blk.idx_rows(c.idx_dev)
        sk = torch.zeros(8, TR, 32, 1)
        for cc in range(8):
            for b in range(4):
                sk[cc, :, 8 * b : 8 * b + 8, 0] = c.sl[8 * cc : 8 * cc + 8][None, :]
        skd = DM.to_mesh(md, sk, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        def split(push):
            if push:
                q = qp(c.q_dev)
                kw = dict(qland=(qp, S))
            else:
                q = blk.gather_q(c.q_dev)
                kw = {}
            return blk.mod.sparse_sdpa_split(
                q,
                c.kv_dev,
                idx8,
                512,
                DM.SCALE,
                blk.kc,
                blk.G,
                blk.ckc,
                None,
                emit_stats=3,
                owner_s=S,
                compact=True,
                **kw,
            )

        def chain(push):
            out = split(push)
            x = ttnn.experimental.all_to_all_async_generic(
                out, in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
            )
            return hy4_merge.merge_fused(x, skd, TR, full=S > 2, hm_users=S)

        res = {"test": "qp_lat", "S": S}
        for rep in range(2):
            res[f"gs_us_{rep}"] = timed_traced(md, lambda: split(False))
            res[f"ps_us_{rep}"] = timed_traced(md, lambda: split(True))
            res[f"gchain_us_{rep}"] = timed_traced(md, lambda: chain(False))
            res[f"pchain_us_{rep}"] = timed_traced(md, lambda: chain(True))
        emit(res)
        c.free()


_XPU = {}


def _xp_sink(md, c, TR):
    sk = torch.zeros(8, TR, 32, 1)
    for cc in range(8):
        for b in range(4):
            sk[cc, :, 8 * b : 8 * b + 8, 0] = c.sl[8 * cc : 8 * cc + 8][None, :]
    return DM.to_mesh(md, sk, ttnn.bfloat16, ttnn.TILE_LAYOUT)


def xpp_run(md, blk, c, S):
    """B47a: compact path with the exchange inside the split program (XPush) + flag-waiting A1 merge -> host heads [64,S,512] (same host unpacking as cmp_run)"""
    import hy4_merge
    from hy4_xpush import XPush

    TR = (S + 3) // 4
    if (id(md), S) not in _XPU:
        _XPU[(id(md), S)] = XPush(md, S)
    xp = _XPU[(id(md), S)]
    qg = blk.gather_q(c.q_dev)
    idx8 = blk.idx_rows(c.idx_dev)
    out = xp.split(qg, c.kv_dev, idx8, blk.mod, blk.kc, blk.G, blk.ckc, DM.SCALE)
    skd = _xp_sink(md, c, TR)
    res = hy4_merge.merge_fused(xp.land, skd, TR, full=S > 2, xp=xp)
    r = DM.from_mesh(md, res).float().reshape(8, TR, 32, 512)
    heads = torch.zeros(8, 8, S, 512)
    for s in range(S):
        u, b = s // 4, s % 4
        heads[:, :, s] = r[:, u, 8 * b : 8 * b + 8]
    return heads.reshape(64, S, 512)


def run_xplat(md):
    """B47a: traced latency of gather+split | + a2a + A1 merge (stock) | + XPush (exchange in the split program) + flag-waiting merge. DBG=1 env HY4_B47A_DBG skips the sends (hang bisect only)."""
    import hy4_merge
    from hy4_xpush import XPush

    for S in [int(x) for x in os.environ.get("SS", "4").split(",")]:
        c = DM.Case(md, "random", 65536, S, "scaled", seed=5)
        blk = DM.mk_block(md, S, fuse=True, xchg=False)
        TR = (S + 3) // 4
        xp = XPush(md, S)
        idx8 = blk.idx_rows(c.idx_dev)
        skd = _xp_sink(md, c, TR)

        def split():
            qg = blk.gather_q(c.q_dev)
            return blk.mod.sparse_sdpa_split(
                qg, c.kv_dev, idx8, 512, DM.SCALE, blk.kc, blk.G, blk.ckc, None, emit_stats=3, owner_s=S, compact=True
            )

        def a1():
            x = ttnn.experimental.all_to_all_async_generic(
                split(), in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
            )
            return hy4_merge.merge_fused(x, skd, TR, full=S > 2, hm_users=S)

        def xpush():
            qg = blk.gather_q(c.q_dev)
            xp.split(qg, c.kv_dev, idx8, blk.mod, blk.kc, blk.G, blk.ckc, DM.SCALE)
            return hy4_merge.merge_fused(xp.land, skd, TR, full=S > 2, hm_users=S, xp=xp)

        t0 = timed_traced(md, split)
        t1 = timed_traced(md, a1)
        t2 = timed_traced(md, xpush)
        emit(
            {
                "test": "xp_lat",
                "S": S,
                "gather_split_us": t0,
                "a1_total_us": t1,
                "xpush_total_us": t2,
                "a1_stage_us": round(t1 - t0, 1),
                "xpush_stage_us": round(t2 - t0, 1),
            }
        )
        c.free()


_XMC = {}


def xmc_run(md, blk, c, S):
    """B26: compact path with the fabric exchange + merge program (XMerge users=S) instead of a2a + A1 merge -> host heads [64,S,512]"""
    from hy4_xmerge import XMerge

    TR = (S + 3) // 4
    if S not in _XMC:
        _XMC[S] = XMerge(md, TR, nsplit=int(os.environ.get("HY4_XM_NSPLIT", "8")), users=S)
    qg = blk.gather_q(c.q_dev)
    idx8 = blk.idx_rows(c.idx_dev)
    out = blk.mod.sparse_sdpa_split(
        qg, c.kv_dev, idx8, 512, DM.SCALE, blk.kc, blk.G, blk.ckc, None, emit_stats=3, owner_s=S, compact=True
    )
    sk = torch.zeros(8, TR, 32, 1)
    for cc in range(8):
        for b in range(4):
            sk[cc, :, 8 * b : 8 * b + 8, 0] = c.sl[8 * cc : 8 * cc + 8][None, :]
    skd = DM.to_mesh(md, sk, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    res = _XMC[S](out, skd)
    return DM.from_mesh(md, res).float().reshape(64, S, 512)


def run_xmclat(md):
    """B26: traced latency of the compact S>1 block: gather+split | + a2a + A1 merge | + XMerge(users) exchange (stage us = diff)"""
    import hy4_merge
    from hy4_xmerge import XMerge

    for S in [int(x) for x in os.environ.get("SS", "4").split(",")]:
        c = DM.Case(md, "random", 65536, S, "scaled", seed=5)
        blk = DM.mk_block(md, S, fuse=True, xchg=False)
        TR = (S + 3) // 4
        xmc = XMerge(md, TR, nsplit=int(os.environ.get("HY4_XM_NSPLIT", "8")), users=S)
        idx8 = blk.idx_rows(c.idx_dev)
        sk = torch.zeros(8, TR, 32, 1)
        for cc in range(8):
            for b in range(4):
                sk[cc, :, 8 * b : 8 * b + 8, 0] = c.sl[8 * cc : 8 * cc + 8][None, :]
        skd = DM.to_mesh(md, sk, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        def split():
            qg = blk.gather_q(c.q_dev)
            return blk.mod.sparse_sdpa_split(
                qg, c.kv_dev, idx8, 512, DM.SCALE, blk.kc, blk.G, blk.ckc, None, emit_stats=3, owner_s=S, compact=True
            )

        a27_links = int(os.environ.get("A27_LINKS", "2"))  # B27: a2a link count / output memory config probes
        a27_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("A27_L1", "0") == "1" else None

        def a2a():
            out = split()
            return ttnn.experimental.all_to_all_async_generic(
                out,
                in_dim=1,
                out_dim=0,
                num_links=a27_links,
                memory_config=a27_mc,
                topology=ttnn.Topology.Ring,
                cluster_axis=1,
            )

        def a1():
            x = a2a()
            return hy4_merge.merge_fused(x, skd, TR, full=S > 2, hm_users=S)

        t0 = timed_traced(md, split)
        ta = timed_traced(md, a2a)
        emit(
            {
                "test": "xmc_lat_a2a",
                "S": S,
                "links": a27_links,
                "l1": a27_mc is not None,
                "a2a_stage_us": round(ta - t0, 1),
            }
        )
        t1 = timed_traced(md, a1)
        t2 = timed_traced(md, lambda: xmc(split(), skd))
        emit(
            {
                "test": "xmc_lat",
                "S": S,
                "gather_split_us": t0,
                "a1_total_us": t1,
                "xmc_total_us": t2,
                "a1_stage_us": round(t1 - t0, 1),
                "xmc_stage_us": round(t2 - t0, 1),
            }
        )
        c.free()


def run_pcc(md):
    var = os.environ.get("VAR", "a1")
    SS = [int(x) for x in os.environ.get("SS", "1,2,3,4,5,6,7,8").split(",")]
    sel = os.environ.get("CASES")
    reps = int(os.environ.get("REPS", "3"))
    for tag in CASES:
        if sel and tag not in sel.split(","):
            continue
        kind, ctx, skew = parse(tag)
        for S in SS:
            name = f"{tag}_S{S}_{var}"
            try:
                c = DM.Case(md, kind, ctx, S, "scaled", seed=3, skew=skew)
                if var in ("cmp", "xmc", "qpp", "xpp"):
                    blk = DM.mk_block(md, S, fuse=True, xchg=False)
                    got = [
                        {"cmp": cmp_run, "xmc": xmc_run, "qpp": qpp_run, "xpp": xpp_run}[var](md, blk, c, S)
                        for _ in range(reps)
                    ]
                    ps = [round(pcc(g, c.ref), 6) for g in got]
                    rows = [round(min(pcc(g[:, s], c.ref[:, s]) for s in range(S)), 6) for g in got]
                    emit(
                        {
                            "test": "vrows_pcc",
                            "case": name,
                            "G": blk.G,
                            "kc": blk.kc,
                            "pcc_per_call": ps,
                            "min_row": rows,
                            "finite": bool(all(torch.isfinite(g).all() for g in got)),
                            "pass": bool(
                                min(ps) >= DM.GATE
                                and min(rows) >= DM.GATE
                                and all(torch.isfinite(g).all() for g in got)
                            ),
                            "empty_col_rows": int((torch.from_numpy(c.cnt) == 0).sum()),
                            "max_logit": c.max_logit,
                        }
                    )
                    c.free()
                    continue
                if var == "stock":
                    blk = DM.mk_block(md, S, fuse=False, xchg=False)
                elif var == "a1":
                    blk = DM.mk_block(md, S, fuse=True, xchg=False)
                else:
                    blk = DM.mk_block(md, S, fuse=True, xchg=True)
                idx8 = blk.idx_rows(c.idx_dev)
                outs = [
                    blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev if var != "stock" else c.sinkrow3_dev)
                    for _ in range(reps)
                ]
                got = [DM.result_to_heads(md, o, S, True) for o in outs]
                ps = [round(pcc(g, c.ref), 6) for g in got]
                rows = [round(min(pcc(g[:, s], c.ref[:, s]) for s in range(S)), 6) for g in got]
                emit(
                    {
                        "test": "vrows_pcc",
                        "case": name,
                        "G": blk.G,
                        "kc": blk.kc,
                        "pcc_per_call": ps,
                        "min_row": rows,
                        "finite": bool(all(torch.isfinite(g).all() for g in got)),
                        "pass": bool(
                            min(ps) >= DM.GATE and min(rows) >= DM.GATE and all(torch.isfinite(g).all() for g in got)
                        ),
                        "empty_col_rows": int((torch.from_numpy(c.cnt) == 0).sum()),
                        "max_logit": c.max_logit,
                    }
                )
                if var == "xm":
                    free_xm(blk)
                c.free()
            except Exception as e:
                import traceback

                traceback.print_exc()
                emit({"test": "vrows_pcc", "case": name, "error": str(e).strip().splitlines()[0][:300]})


def run_lat(md):
    """block latency (traced) per S: split-kernel stage only (upto=2), full block; GK="G:kc,G:kc" sweeps the split config (default: pick_split)"""
    var = os.environ.get("VAR", "a1")
    SS = [int(x) for x in os.environ.get("SS", "1,2,3,4,5,6,7,8").split(",")]
    gks = [tuple(int(v) for v in x.split(":")) for x in os.environ["GK"].split(",")] if os.environ.get("GK") else [None]
    for tag in os.environ.get("CASES", "random_65536,random_65536_skew").split(","):
        kind, ctx, skew = parse(tag)
        for S in SS:
            try:
                c = DM.Case(md, kind, ctx, S, "scaled", seed=5, skew=skew)
                for gk in gks:
                    try:
                        kw = dict(G=gk[0], kc=gk[1]) if gk else {}
                        if gk and S * 8 * gk[0] > 120:
                            continue
                        blk = DM.mk_block(md, S, fuse=var != "stock", xchg=var == "xm", **kw)
                        idx8 = blk.idx_rows(c.idx_dev)
                        full = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev))
                        upto2 = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev, upto=2))
                        upto1 = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev, upto=1))
                        emit(
                            {
                                "test": "vrows_lat",
                                "case": f"{tag}_S{S}_{var}",
                                "G": blk.G,
                                "kc": blk.kc,
                                "block_us": full,
                                "qg_plus_kernel_us": upto2,
                                "qg_us": upto1,
                            }
                        )
                        if var == "xm":
                            free_xm(blk)
                    except Exception as e:
                        emit(
                            {
                                "test": "vrows_lat",
                                "case": f"{tag}_S{S}_{var}_{gk}",
                                "error": str(e).strip().splitlines()[0][:300],
                            }
                        )
                c.free()
            except Exception as e:
                import traceback

                traceback.print_exc()
                emit({"test": "vrows_lat", "case": f"{tag}_S{S}_{var}", "error": str(e).strip().splitlines()[0][:300]})


if __name__ == "__main__":
    full, md = open_row()
    try:
        for name, fn in [
            ("pcc", run_pcc),
            ("lat", run_lat),
            ("xmclat", run_xmclat),
            ("qplat", run_qplat),
            ("xplat", run_xplat),
        ]:
            if os.environ.get("TEST", "pcc") in (name, "all"):
                fn(md)
    finally:
        close_row(full, md)
