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
    """block latency (traced) per S: split-kernel stage only (upto=2), full block, for the balanced and skew cases"""
    var = os.environ.get("VAR", "a1")
    SS = [int(x) for x in os.environ.get("SS", "1,2,3,4,5,6,7,8").split(",")]
    for tag in os.environ.get("CASES", "random_65536,random_65536_skew").split(","):
        kind, ctx, skew = parse(tag)
        for S in SS:
            try:
                c = DM.Case(md, kind, ctx, S, "scaled", seed=5, skew=skew)
                blk = DM.mk_block(md, S, fuse=var != "stock", xchg=var == "xm")
                idx8 = blk.idx_rows(c.idx_dev)
                full = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev))
                upto2 = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev, upto=2))
                emit(
                    {
                        "test": "vrows_lat",
                        "case": f"{tag}_S{S}_{var}",
                        "G": blk.G,
                        "kc": blk.kc,
                        "block_us": full,
                        "qg_plus_kernel_us": upto2,
                    }
                )
                if var == "xm":
                    free_xm(blk)
                c.free()
            except Exception as e:
                import traceback

                traceback.print_exc()
                emit({"test": "vrows_lat", "case": f"{tag}_S{S}_{var}", "error": str(e).strip().splitlines()[0][:300]})


if __name__ == "__main__":
    full, md = open_row()
    try:
        for name, fn in [("pcc", run_pcc), ("lat", run_lat)]:
            if os.environ.get("TEST", "pcc") in (name, "all"):
                fn(md)
    finally:
        close_row(full, md)
