# SPDX-License-Identifier: Apache-2.0
"""B36 unit: sel2 (stage-2 select: the kl best of 16 sorted candidate lists + id gather, one program) vs
 (a) a host model (value desc, list asc, position asc; -inf dropped; exhausted -> (-inf, 0xFFFFFFFF)), bit exact on values AND ids,
 (b) the stock chain (topk_large_indices over the B*kb row + cs_gather): both sorted descending, so the bf16 VALUE sequences must be bit identical (ids differ only among ties).
Every device holds different rows (heavy ties, -inf tails incl. fully invalid lists, valid counts below and above kl), several launches in a row."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt import hy4_idx_select as IS
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP

S = int(os.environ.get("S", "4"))
NB, KB, KL = 16, 512, 512
BLOCKED = os.environ.get("BLOCKED", "1") == "1"
REPS = int(os.environ.get("REPS", "8"))
R, C = 4, 8
LEVELS = [30, 300, 8, 3000]


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def host_ref(vals, ids):
    """vals [S][NB][KB] float, ids [S][NB][KB] int -> (v [S,KL] bf16 bits int, g [S,KL] int)"""
    ov = torch.full((S, KL), 0xFF80, dtype=torch.int64)
    og = torch.full((S, KL), 0xFFFFFFFF, dtype=torch.int64)
    for s in range(S):
        ent = []
        for b in range(NB):
            for p in range(KB):
                v = float(vals[s][b][p])
                if v != float("-inf"):
                    ent.append((-v, b, p))
        ent.sort()
        for i, (_, b, p) in enumerate(ent[:KL]):
            ov[s, i] = int(torch.tensor(vals[s][b][p]).to(torch.bfloat16).view(torch.int16)) & 0xFFFF
            og[s, i] = int(ids[s][b][p])
    return ov, og


def main():
    md = open_mesh(trace=int(100e6))
    try:
        g = torch.Generator().manual_seed(33)
        m2 = lambda dims: ttnn.ShardTensor2dMesh(md, mesh_shape=(R, C), dims=dims)
        up = lambda t, dt, dims: ttnn.from_torch(
            t,
            dtype=dt,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            mesh_mapper=m2(dims),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ok_host = ok_stock = True
        for it in range(REPS):
            levels = LEVELS[it % len(LEVELS)]
            ninf = [0, 0, 6000, 7600][(it // 2) % 4] if it % 3 else 0
            V = torch.zeros(R, C, S, NB, KB)
            I = torch.zeros(R, C, S, NB, KB, dtype=torch.int64)
            for r in range(R):
                for c in range(C):
                    x = torch.randint(0, levels, (S, NB, KB), generator=g).float() / 16 - 2
                    x = x.to(torch.bfloat16).float()
                    x, _ = torch.sort(x, dim=2, descending=True)
                    if ninf:
                        for s in range(S):
                            for b in range(NB):
                                nin = max(0, min(KB, ninf // NB + int(torch.randint(-200, 200, (1,), generator=g))))
                                if int(torch.randint(0, 8, (1,), generator=g)) == 0:
                                    nin = KB
                                if nin:
                                    x[s, b, KB - nin :] = float("-inf")
                    V[r, c] = x
                    I[r, c] = torch.randint(0, 1 << 30, (S, NB, KB), generator=g)
            assert BLOCKED  # production layout (HY4_IDX_BLK=1)
            vh = V.permute(0, 2, 1, 3, 4).reshape(R * S, 1, 1, C * NB * KB)
            ih = I.permute(0, 2, 1, 3, 4).reshape(R * S, 1, 1, C * NB * KB)
            vt = up(vh.to(torch.bfloat16), ttnn.bfloat16, (0, 3))
            it_ = up(ih.to(torch.int32), ttnn.uint32, (0, 3))
            li, lv = HP.sel2(vt, it_, S, NB, KB, KL, BLOCKED)
            # stock chain on the same data
            cvals = ttnn.reshape(vt, [1, 1, S, NB * KB])
            pos = IS.topk_ids(cvals, KL)
            sid, sval = IS.cs_gather(
                ttnn.reshape(pos, [S, 1, 1, KL]),
                ttnn.reshape(it_, [S, 1, 1, NB * KB]),
                ttnn.reshape(vt, [S, 1, 1, NB * KB]),
                4,
            )
            ttnn.synchronize_device(md)
            dli, dlv = ttnn.get_device_tensors(li), ttnn.get_device_tensors(lv)
            dsi, dsv = ttnn.get_device_tensors(sid), ttnn.get_device_tensors(sval)
            nbad_host = nbad_stock = 0
            for r in range(R):
                for c in range(C):
                    gv = ttnn.to_torch(dlv[r * C + c]).reshape(S, KL).view(torch.int16).long() & 0xFFFF
                    gi = ttnn.to_torch(dli[r * C + c]).reshape(S, KL).long() & 0xFFFFFFFF
                    ev, eg = host_ref(V[r, c], I[r, c])
                    if not (torch.equal(gv, ev) and torch.equal(gi, eg)):
                        nbad_host += 1
                        if nbad_host == 1:
                            res(
                                event="host_mismatch",
                                it=it,
                                r=r,
                                c=c,
                                nbad_v=int((gv != ev).sum()),
                                nbad_i=int((gi != eg).sum()),
                            )
                    sv = ttnn.to_torch(dsv[r * C + c]).reshape(S, KL).view(torch.int16).long() & 0xFFFF
                    nvld = (ev != 0xFF80).sum(dim=1)
                    for s in range(S):
                        n = int(nvld[s])
                        if not torch.equal(gv[s, :n], sv[s, :n]):
                            nbad_stock += 1
                            res(event="stock_mismatch", it=it, r=r, c=c, s=s, n=n)
                            break
            ok_host &= nbad_host == 0
            ok_stock &= nbad_stock == 0
            res(event="case", it=it, levels=levels, ninf=ninf, devs_bad_vs_host=nbad_host, bad_vs_stock=nbad_stock)
        res(event="done", ok_vs_host=ok_host, ok_vs_stock_values=ok_stock)
    finally:
        close_mesh(md)


main()
