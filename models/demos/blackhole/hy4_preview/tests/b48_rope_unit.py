"""B48 unit test (1x8 slot, MESH_DEVICE=1x8, HY4_B40_DERIVE=1): SpecDerive.derive with the precomputed rope table (HY4_B48_ROPETAB=1) vs the stock two-level path, bit for bit
(cos, sin, c128, s128 and the B40 geometry tensors), S = 1, 2, 4, boundary + random positions up to 1M; then trace timing of 20 derives per variant. RES lines."""
import json
import os
import random
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_common import RopeDev, upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import SpecDerive
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def snap(md, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).clone()


def main():
    SS = [int(x) for x in os.environ.get("SS", "1,2,4").split(",")]
    NPOS = int(os.environ.get("N", "300"))
    first = os.environ.get("FIRST", "0") == "1"
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        rope = RopeDev(md, 64, 1.0e7, True)
        rnd = random.Random(7)
        for S in SS:
            ctx = PoolCtx(md, mc, ccl, 2048, rows=1, S=S, mode="dense", bucket_tokens=4096)
            os.environ["HY4_B48_ROPETAB"] = "0"
            sd0 = SpecDerive(md, ctx, rope, first, S)
            t0 = time.time()
            os.environ["HY4_B48_ROPETAB"] = "1"
            sd1 = SpecDerive(md, ctx, rope, first, S)
            res(event="table_built", S=S, secs=round(time.time() - t0, 1))
            assert sd0._rtab is None and sd1._rtab is not None
            pos = [0, 1, 1023, 1024, 1025, 65535, 65536, 1048575 - S, 1048576 - S] + [
                rnd.randrange(0, 1048576 - S) for _ in range(NPOS)
            ]
            bad = []
            for P in pos:
                p = upload(md, torch.tensor([float(P)]).reshape(1, 1, 1, 1), ttnn.float32)
                outs = []
                for sd in (sd0, sd1):
                    sd.derive(p)
                    ttnn.synchronize_device(md)
                    d = {n: snap(md, getattr(sd, n)) for n in ("cos", "sin", "c128", "s128")}
                    for n in ("wr", "kvcnt", "colinfo"):
                        d[n] = snap(md, getattr(ctx, n))
                    outs.append(d)
                for n in outs[0]:
                    if not torch.equal(
                        outs[0][n].view(torch.int16) if outs[0][n].dtype == torch.bfloat16 else outs[0][n],
                        outs[1][n].view(torch.int16) if outs[1][n].dtype == torch.bfloat16 else outs[1][n],
                    ):
                        bad.append((P, n))
                ttnn.deallocate(p)
            res(event="equal", S=S, positions=len(pos), mismatches=bad[:10], ok=not bad)
            p = upload(md, torch.tensor([float(123456)]).reshape(1, 1, 1, 1), ttnn.float32)
            for name, sd in (("stock", sd0), ("table", sd1)):
                sd.derive(p)
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                for _ in range(20):
                    sd.derive(p)
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ts = []
                for _ in range(5):
                    t0 = time.time()
                    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(md)
                    ts.append((time.time() - t0) / 20 * 1e6)
                ttnn.release_trace(md, tid)
                res(event="time", S=S, variant=name, us_per_derive=min(ts))
            del sd1._rtab
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
