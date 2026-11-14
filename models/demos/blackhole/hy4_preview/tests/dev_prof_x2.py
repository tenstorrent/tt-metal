"""fuse-sparse-merge: device-profiler run of the fused (A2b) vs unfused scaled block, eager calls, then ReadDeviceProfiler. env X2V=a2b|a1 X2F=scaled|bf16"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
import test_hy4_decode_merge_device as DM
from dm_block import close_row, open_row
import ttnn
full, md = open_row()
try:
    v, fmt = os.environ.get("X2V", "a2b"), os.environ.get("X2F", "scaled")
    S = 1
    c = DM.Case(md, "random", 65536, S, fmt, seed=5)
    blk = DM.mk_block(md, S, fuse=True, xchg=(v == 'a2'))
    blk.xchg2 = v == "a2b"
    idx8 = blk.idx_rows(c.idx_dev)
    for _ in range(3):
        blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev)
    ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(4):
        blk.run(c.q_dev, c.kv_dev, idx8, c.sinkrow3_dev)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    ttnn.ReadDeviceProfiler(md)
    ttnn.release_trace(md, tid)
    print("PROF_DONE", flush=True)
finally:
    close_row(full, md)
