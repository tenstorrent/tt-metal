"""fuse-sparse-merge: where do the 19 us of the fused merge go? (HY4_MRG_DBG variants, isolated stage, traced)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
import test_hy4_decode_merge_device as DM
from dk_dev import timed_traced
from dm_block import close_row, open_row
import ttnn
import hy4_merge

full, md = open_row()
try:
    S = 1
    c = DM.Case(md, "random", 65536, S, "bf16", seed=5)
    base = DM.mk_block(md, S, fuse=False)
    x = base.a2a(base.kernel(base.gather_q(c.q_dev), c.kv_dev, base.idx_rows(c.idx_dev)))
    ttnn.synchronize_device(md)
    for ns in (4, 16):
        us = timed_traced(md, lambda: hy4_merge.merge_fused(x, c.sinkrow3_dev, S, nsplit=ns))
        print("RES", {"dbg": os.environ.get("HY4_MRG_DBG", "0"), "nsplit": ns, "us": us}, flush=True)
    # trivial op for the launch-gap floor
    us = timed_traced(md, lambda: ttnn.add(c.sinkrow3_dev, c.sinkrow3_dev))
    print("RES", {"floor_add_us": us}, flush=True)
finally:
    close_row(full, md)
