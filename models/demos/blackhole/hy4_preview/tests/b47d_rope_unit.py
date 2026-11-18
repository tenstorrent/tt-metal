"""B47d unit (1x8 slot, MESH_DEVICE=1x8): HY4_B47D_ROPE=1 SpecDerive.derive (rope rows gathered inside the one-core derive_geom program from final bf16 tables)
vs the stock B40 derive (two-level fp32 ops): cos / sin / c128 / s128 (rows < S, all 8 devices) and wr / kvcnt / colinfo / upd bit-exact.
Env: SS=1,2,3,4  N=300 random positions  FIRST=0,1  SOAK=1 (every position repeated ITER times)  TIME=1 (trace timing stock vs new)."""
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
    SS = [int(x) for x in os.environ.get("SS", "1,2,3,4").split(",")]
    FIRSTS = [bool(int(x)) for x in os.environ.get("FIRST", "0,1").split(",")]
    NPOS = int(os.environ.get("N", "300"))
    ITER = int(os.environ.get("ITER", "1")) if os.environ.get("SOAK") == "1" else 1
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        rope = RopeDev(md, 64, 1.0e7, True)
        rnd = random.Random(7)
        os.environ["HY4_B40_DERIVE"] = "1"
        for S in SS:
            ctx = PoolCtx(md, mc, ccl, 2048, rows=1, S=S, mode="topk", bucket_tokens=4096)
            for first in FIRSTS:
                os.environ["HY4_B47D_ROPE"] = "0"
                t0 = time.time()
                ref = SpecDerive(md, ctx, rope, first, S)
                os.environ["HY4_B47D_ROPE"] = "1"
                new = SpecDerive(md, ctx, rope, first, S)
                res(event="built", S=S, first=first, secs=round(time.time() - t0, 1))
                edge = [
                    0,
                    1,
                    2,
                    62,
                    63,
                    64,
                    1022,
                    1023,
                    1024,
                    1025,
                    2047,
                    4095,
                    65535,
                    65536,
                    524287,
                    524288,
                    1048570,
                    1048575 - S,
                    1048576 - S,
                ]
                pos_list = edge + [rnd.randrange(0, 1048576 - S) for _ in range(NPOS)]
                bad, n = {}, 0
                for P in pos_list:
                    p = upload(md, torch.tensor([float(P)]).reshape(1, 1, 1, 1), ttnn.float32)
                    ref.derive(p)
                    ttnn.synchronize_device(md)
                    want = {k: snap(md, getattr(ref, k)) for k in ("cos", "sin", "c128", "s128")}
                    for k in ("wr", "kvcnt", "colinfo"):
                        want[k] = snap(md, getattr(ctx, k))
                    for j in range(S):
                        want[f"upd{j}"] = snap(md, ctx.upd_rows[j])
                    for it in range(ITER):
                        new.derive(p)
                        ttnn.synchronize_device(md)
                        got = {k: snap(md, getattr(new, k)) for k in ("cos", "sin", "c128", "s128")}
                        for k in ("wr", "kvcnt", "colinfo"):
                            got[k] = snap(md, getattr(ctx, k))
                        for j in range(S):
                            got[f"upd{j}"] = snap(md, ctx.upd_rows[j])
                        for k in want:
                            a, b = want[k], got[k]
                            if k in ("cos", "sin", "c128", "s128"):
                                a, b = a[:, :, :S], b[:, :, :S]  # valid rows (all 8 devices)
                            if not torch.equal(
                                a.reshape(-1).view(torch.int16) if a.dtype == torch.bfloat16 else a,
                                b.reshape(-1).view(torch.int16) if b.dtype == torch.bfloat16 else b,
                            ):
                                bad.setdefault(k, []).append((P, it))
                        n += 1
                    ttnn.deallocate(p)
                res(
                    event="bitexact",
                    S=S,
                    first=first,
                    cases=n,
                    mismatches={k: v[:5] for k, v in bad.items()},
                    ok=not bad,
                )
                if os.environ.get("TIME") == "1":
                    p = upload(md, torch.tensor([65536.0]).reshape(1, 1, 1, 1), ttnn.float32)
                    out = {}
                    for name, sd in (("stock", ref), ("b47d", new)):
                        sd.derive(p)
                        ttnn.synchronize_device(md)
                        tid = ttnn.begin_trace_capture(md, cq_id=0)
                        for _ in range(20):
                            sd.derive(p)
                        ttnn.end_trace_capture(md, tid, cq_id=0)
                        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                        t0 = time.time()
                        for _ in range(10):
                            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                        ttnn.synchronize_device(md)
                        out[name] = round((time.time() - t0) / 10 / 20 * 1e6, 1)
                        ttnn.release_trace(md, tid)
                    res(event="time_us_per_derive", S=S, first=first, **out)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
