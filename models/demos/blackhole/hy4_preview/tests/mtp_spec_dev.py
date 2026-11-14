"""mtp-device S3a: device-side round bookkeeping on a 1x8 slot: SpecDerive.derive(p) vs the host helpers (PoolCtx.set_step) for S = 1, 2, 4 at random positions incl. page / column boundaries up to 1M,
and the in-trace greedy accept vs spec/accept.finish_round (random matches, caps). Prints RES lines. Env: SS=1,4  N=40 (positions per S)."""
import json
import os
import random

import torch

import ttnn
from models.demos.blackhole.hy4_preview.spec.accept import finish_round
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_common import RopeDev, upload
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import SpecDerive, accept
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def snap(md, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).clone()


def main():
    SS = [int(x) for x in os.environ.get("SS", "1,4").split(",")]
    NPOS = int(os.environ.get("N", "40"))
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        rope = RopeDev(md, 64, 1.0e7, True)
        rnd = random.Random(5)
        for S in SS:
            pages = 2048
            ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=S, mode="dense", bucket_tokens=4096)
            sd = SpecDerive(md, ctx, rope, False, S)
            pos_list = [0, 1, 62, 63, 64, 511, 512, 513, 2047, 2048, 2049, 4095, 65535, 65536, 1048570, 1048575 - S] + [
                rnd.randrange(0, 1048576 - S) for _ in range(NPOS)
            ]
            bad = {}
            for P in pos_list:
                ctx.set_step([P])
                ref = dict(
                    wr=snap(md, ctx.wr),
                    kvcnt=snap(md, ctx.kvcnt),
                    colinfo=snap(md, ctx.colinfo),
                    ids=snap(md, ctx.ids_g),
                )
                for j in range(S):
                    ref[f"upd{j}"] = snap(md, ctx.upd_rows[j])
                c, s_ = rope.tables(list(range(P, P + S)))
                ref["cos"], ref["sin"] = snap(md, c)[:1], snap(md, s_)[:1]
                p = upload(md, torch.tensor([float(P)]).reshape(1, 1, 1, 1), ttnn.float32)
                sd.derive(p)
                ttnn.synchronize_device(md)
                got = dict(
                    wr=snap(md, ctx.wr),
                    kvcnt=snap(md, ctx.kvcnt),
                    colinfo=snap(md, ctx.colinfo),
                    ids=snap(md, ctx.ids_g),
                )
                for j in range(S):
                    got[f"upd{j}"] = snap(md, ctx.upd_rows[j])
                got["cos"], got["sin"] = snap(md, sd.cos)[:1], snap(md, sd.sin)[:1]
                for k in ref:
                    a, b = ref[k], got[k]
                    if k in ("cos", "sin"):
                        # host tables are fp32-computed (rounded at 1M); compare against the accurate device tables loosely: max abs diff
                        d = float((a.float() - b.float()).abs().max())
                        if d > 0.05:
                            bad.setdefault(k, []).append((P, d))
                    elif not torch.equal(a.to(torch.int64) & 0xFFFFFFFF, b.to(torch.int64) & 0xFFFFFFFF):
                        bad.setdefault(k, []).append(P)
                ttnn.deallocate(p)
            res(event="derive", S=S, positions=len(pos_list), mismatches={k: v[:5] for k, v in bad.items()}, ok=not bad)
            # ---- accept
            k = S - 1
            if k >= 1:
                ar16 = upload(md, torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 16), ttnn.float32)
                oh0 = ar16
                nbad = 0
                for t in range(100):
                    drafts = [rnd.randrange(5) for _ in range(k)]
                    tgt = [d if rnd.random() < 0.6 else rnd.randrange(5) for d in drafts] + [rnd.randrange(5)]
                    cap = rnd.choice([0, 1, 2, 5])
                    a = torch.zeros(16)
                    a[: k + 1] = torch.tensor(tgt, dtype=torch.float32)
                    d = torch.full((16,), -1.0)
                    d[:k] = torch.tensor(drafts, dtype=torch.float32)
                    mt, oh, nx = accept(
                        upload(md, a.reshape(1, 1, 1, 16), ttnn.float32),
                        upload(md, d.reshape(1, 1, 1, 16), ttnn.float32),
                        upload(md, torch.tensor([float(cap)]).reshape(1, 1, 1, 1), ttnn.float32),
                        k,
                        ar16,
                        oh0,
                        None,
                    )
                    m_dev = int(round(float(ttnn.to_torch(ttnn.get_device_tensors(mt)[0]).reshape(-1)[0])))
                    nx_dev = int(round(float(ttnn.to_torch(ttnn.get_device_tensors(nx)[0]).reshape(-1)[0])))
                    r = finish_round(100, drafts, tgt, cap=cap)
                    if m_dev != r.m or nx_dev != r.next_pending:
                        nbad += 1
                res(event="accept", k=k, cases=100, mismatches=nbad, ok=nbad == 0)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
