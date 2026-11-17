# SPDX-License-Identifier: Apache-2.0
"""B32 A2: real-routing census of the tp24 routed-expert union in verify rounds (Tt = 16 rows = 4 users x S=4) on the b24 prompt classes.
Per class: batch prefill, NC_WARM traced rounds, then NC_EAGER eager rounds (runner.body) with Tp24Experts.__call__ wrapped to dump the packed route words (device 0) per layer.
Writes /rout/<TAG>_census.json: list of rounds, each a list of per-layer dicts {n: distinct, g: [count e%3==0,1,2], u: [distinct per user], ids: [[8 x 4] x 4 users]}."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

K = 3
MAXLEN = int(os.environ.get("MAXLEN", "32768"))
NWARM = int(os.environ.get("NC_WARM", "20"))
NEAGER = int(os.environ.get("NC_EAGER", "4"))
R = 4
TAG = os.environ.get("QTAG", "census")


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    from models.demos.blackhole.hy4_preview.tt import hy4_moe_tp24 as M24
    from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

    d = torch.load(os.environ.get("B24_PROMPTS", "/b24/b24_prompts.pt"))
    nd = (
        torch.load(os.environ.get("NEEDLE_PROMPTS", "/b32/needle_prompts.pt"))
        if "needle" in os.environ.get("CLASSES", "")
        else None
    )
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "1900e6"))))
    cur = {"on": False, "layers": []}
    orig = M24.Tp24Experts.__call__

    def wrapped(self, h, route, T, out=None, shared=None):
        if cur["on"]:
            ttnn.synchronize_device(md)
            t = ttnn.to_torch(ttnn.get_device_tensors(route)[0]).reshape(-1).to(torch.int64)
            tt = t.shape[0] // 16
            w = t.reshape(tt, 16)[:, :8] & 0xFFFF
            cur["layers"].append(dict(T=int(T), ids=w.tolist()))
        return orig(self, h, route, T, out=out, shared=shared)

    M24.Tp24Experts.__call__ = wrapped
    from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as PD

    orig_sg = PD.select_global
    cur["idx"] = []

    def sg(ctx, qh, w32, key_view):
        g = orig_sg(ctx, qh, w32, key_view)
        if cur["on"]:
            ttnn.synchronize_device(md)
            dts = ttnn.get_device_tensors(g)
            rec = []
            for u in range(4):
                x = ttnn.to_torch(dts[8 * u]).reshape(-1, 2048).to(torch.int64) & 0xFFFFFFFF
                sets = [set(v for v in x[j].tolist() if v != 0xFFFFFFFF) for j in range(x.shape[0])]
                nv = [len(a) for a in sets]
                ov = [len(sets[0] & a) / max(1, len(a)) for a in sets]  # row j vs row 0
                ovn = [len(sets[j] & sets[j + 1]) / max(1, len(sets[j])) for j in range(len(sets) - 1)]
                un = len(set().union(*sets))
                rec.append(dict(nv=nv, ov0=ov, ovn=ovn, union=un))
            cur["idx"].append(rec)
        return g

    PD.select_global = sg
    out = {}
    try:
        eng = Hy4Engine(md, R, MAXLEN, k=K)
        eng.warm()
        res(event="warm")
        for cls in os.environ.get("CLASSES", "gsm,code,story,chat").split(","):
            ps = d[cls][:R] if cls != "needle" else [list(x) for x in nd["ids"]][:R]
            eng.drain()
            for r in range(R):
                eng.release(r)
            t0 = time.time()
            first = eng.prefill({r: ps[r] for r in range(len(ps))})
            res(event="prefill", cls=cls, lens=[len(x) for x in ps], sec=round(time.time() - t0, 1))
            blk = None
            for i in range(NWARM):
                o = eng.round([K] * R, False, plugin_block=blk)
                blk = {r: (o[r]["tok"], o[r]["pos"]) for r in range(R) if eng.live[r]}
            rounds = []
            for e in range(NEAGER):
                cur["layers"] = []
                cur["idx"] = []
                cur["on"] = True
                eng.runner.body("G")
                ttnn.synchronize_device(md)
                cur["on"] = False
                rounds.append(dict(moe=cur["layers"], idx=cur["idx"]))
                res(event="eager", cls=cls, i=e, ncalls=len(cur["layers"]))
            out[cls] = rounds
            json.dump(out, open(f"/rout/{TAG}_census.json", "w"))
        res(event="finished")
    finally:
        M24.Tp24Experts.__call__ = orig
        PD.select_global = orig_sg
        close_mesh(md)


if __name__ == "__main__":
    main()
