"""fuse-sparse-merge stage A1: fused JIT merge kernel vs the 8-op merge3 and the torch full-softmax reference, 1x8 row. Run: run_fsm.sh <tag> tests.test_hy4_fused_merge_device TEST=pcc|lat [CASES=tag,..]
(module form: dm_block/dk_dev resolve through sys.path inserts below)"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
import test_hy4_decode_merge_device as DM
from dk_dev import pcc, timed_traced
from dm_block import NC, Block, close_row, from_mesh, open_row

import ttnn

emit = DM.emit
CASES = [
    ("random", 65536, 1, "bf16"),
    ("real", 65536, 1, "bf16"),
    ("random", 65536, 2, "bf16"),
    ("real", 65536, 2, "bf16"),
    ("random", 3000, 1, "bf16"),
    ("random", 100, 1, "bf16"),
    ("real", 100, 2, "bf16"),
    ("random", 65536, 1, "scaled"),
    ("real", 65536, 2, "scaled"),
    ("real", 100, 1, "scaled"),
    ("random", 1048576, 1, "scaled"),
    ("hot", 65536, 1, "bf16"),
    ("hot", 65536, 2, "scaled"),
]


def run_pcc(md):
    sel = os.environ.get("CASES")
    for kind, ctx, S, fmt in CASES:
        tag = f"{kind}_{ctx}_S{S}_{fmt}"
        if sel and tag not in sel.split(","):
            continue
        try:
            c = DM.Case(md, kind, ctx, S, fmt, seed=3)
            base = DM.mk_block(md, S, fuse=False)
            fus = DM.mk_block(md, S, fuse=True)
            idx8 = base.idx_rows(c.idx_dev)
            qg = base.gather_q(c.q_dev)
            out = base.kernel(qg, c.kv_dev, idx8)
            x = base.a2a(out)
            a = base.merge(x, c.sinkrow3_dev)
            b = fus.merge(x, c.sinkrow3_dev)
            ga, gb = DM.result_to_heads(md, a, S, True), DM.result_to_heads(md, b, S, True)
            p, pb = pcc(gb, c.ref), pcc(ga, c.ref)
            per_row = min(pcc(gb[:, s], c.ref[:, s]) for s in range(S))
            emit(
                {
                    "test": "fused_merge_pcc",
                    "case": tag,
                    "pcc_fused_vs_ref": round(p, 6),
                    "pcc_stock_vs_ref": round(pb, 6),
                    "pcc_fused_vs_stock": round(pcc(gb, ga), 6),
                    "min_row": round(per_row, 6),
                    "finite": bool(torch.isfinite(gb).all()),
                    "pass": bool(p >= DM.GATE and per_row >= DM.GATE and torch.isfinite(gb).all()),
                    "empty_col_rows": int((torch.from_numpy(c.cnt) == 0).sum()),
                    "max_logit": c.max_logit,
                }
            )
            c.free()
        except Exception as e:
            import traceback

            traceback.print_exc()
            emit({"test": "fused_merge_pcc", "case": tag, "error": str(e).strip().splitlines()[0][:300]})


def run_xchg_pcc(md):
    """A2: fabric exchange + merge in one program (HY4_XM_NSPLIT) vs torch full softmax"""
    sel = os.environ.get("CASES")
    for kind, ctx, S, fmt in CASES:
        tag = f"{kind}_{ctx}_S{S}_{fmt}"
        if sel and tag not in sel.split(","):
            continue
        try:
            c = DM.Case(md, kind, ctx, S, fmt, seed=3)
            blk = DM.mk_block(md, S, fuse=True, xchg=True)
            idx8 = blk.idx_rows(c.idx_dev)
            outs = []
            for rep in range(3):  # repeated calls: semaphore / landing buffer reuse
                outs.append(blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev))
            got = [DM.result_to_heads(md, o, S, True) for o in outs]
            ps = [round(pcc(g, c.ref), 6) for g in got]
            emit({"test": "xchg_pcc", "case": tag, "pcc_per_call": ps, "finite": bool(all(torch.isfinite(g).all() for g in got)), "pass": bool(min(ps) >= DM.GATE)})
            c.free()
        except Exception as e:
            import traceback

            traceback.print_exc()
            emit({"test": "xchg_pcc", "case": tag, "error": str(e).strip().splitlines()[0][:300]})
            raise


def run_x2_pcc(md):
    sel = os.environ.get("CASES")
    for kind, ctx, S, fmt in CASES:
        tag = f"{kind}_{ctx}_S{S}_{fmt}"
        if sel and tag not in sel.split(","):
            continue
        c = DM.Case(md, kind, ctx, S, fmt, seed=3)
        blk = DM.mk_block(md, S, fuse=True, xchg=False)
        blk.xchg2 = True
        idx8 = blk.idx_rows(c.idx_dev)
        outs = [blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev) for _ in range(3)]
        got = [DM.result_to_heads(md, o, S, True) for o in outs]
        ps = [round(pcc(g, c.ref), 6) for g in got]
        emit({"test": "xchg2_pcc", "case": tag, "pcc_per_call": ps, "finite": bool(all(torch.isfinite(g).all() for g in got)), "pass": bool(min(ps) >= DM.GATE)})
        c.free()


def run_x2_lat(md):
    v = os.environ.get("X2V", "a2b")  # one variant per process: the persistent L1 buffers of the variants stack at the top of L1
    for fmt in os.environ.get("X2F", "bf16,scaled").split(","):
        for S in (1, 2):
            c = DM.Case(md, "random", 65536, S, fmt, seed=5)
            blk = DM.mk_block(md, S, fuse=True, xchg=(v == "a2"))
            blk.xchg2 = v == "a2b"
            if v == "a1":
                blk.xchg = False
            idx8 = blk.idx_rows(c.idx_dev)
            us = timed_traced(md, lambda: blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev))
            emit({"test": "xchg2_lat", "variant": v, "fmt": fmt, "S": S, "us": us})
            c.free()
            blk.xm = blk.xm2 = None


def run_xchg_lat(md):
    for fmt in ("bf16", "scaled"):
        for S in (1, 2):
            c = DM.Case(md, "random", 65536, S, fmt, seed=5)
            base = DM.mk_block(md, S, fuse=True, xchg=False)
            xb = DM.mk_block(md, S, fuse=True, xchg=True)
            idx8 = base.idx_rows(c.idx_dev)
            a = timed_traced(md, lambda: base.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev))
            b = timed_traced(md, lambda: xb.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev))
            q2 = timed_traced(md, lambda: xb.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev, upto=2))
            emit({"test": "xchg_lat", "fmt": fmt, "S": S, "block_a1_us": a, "block_a2_us": b, "qg_plus_kernel_us": q2, "xm_stage_us": round(b - q2, 1), "dbg": os.environ.get("HY4_XM_DBG", "0")})
            c.free()


def run_lat(md):
    for fmt in ("bf16", "scaled"):
        for S in (1, 2):
            c = DM.Case(md, "random", 65536, S, fmt, seed=5)
            base = DM.mk_block(md, S, fuse=False)
            idx8 = base.idx_rows(c.idx_dev)
            x = base.a2a(base.kernel(base.gather_q(c.q_dev), c.kv_dev, idx8))
            us0 = timed_traced(md, lambda: base.merge(x, c.sinkrow3_dev))
            emit({"test": "fused_merge_lat", "fmt": fmt, "S": S, "variant": "stock_merge3", "us": us0})
            for ns in (2, 4, 8, 16):
                if S * ns > 120:
                    continue
                import hy4_merge

                us = timed_traced(md, lambda: hy4_merge.merge_fused(x, c.sinkrow3_dev, S, nsplit=ns))
                emit({"test": "fused_merge_lat", "fmt": fmt, "S": S, "variant": f"fused_nsplit{ns}", "us": us})
            fb = DM.mk_block(md, S, fuse=True)
            tot = timed_traced(md, lambda: fb.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev))
            sb = timed_traced(md, lambda: base.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev))
            emit({"test": "fused_merge_lat", "fmt": fmt, "S": S, "variant": "block_total", "stock_us": sb, "fused_us": tot})
            c.free()


if __name__ == "__main__":
    full, md = open_row()
    try:
        for name, fn in [("pcc", run_pcc), ("lat", run_lat), ("xpcc", run_xchg_pcc), ("xlat", run_xchg_lat), ("x2pcc", run_x2_pcc), ("x2lat", run_x2_lat)]:
            if os.environ.get("TEST", "pcc") in (name, "all"):
                fn(md)
    finally:
        close_row(full, md)
