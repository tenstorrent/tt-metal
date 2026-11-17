# SPDX-License-Identifier: Apache-2.0
"""B36 unit: sel3_own (flag wait + stage-3 top-2048 over the 8 gathered sorted candidate rows + own filter, one program) vs
 (a) a host model: global top-K by (value desc, source column asc, list position asc), -inf dropped; out = own selected ids in list order (bit exact),
 (b) the stock chain on the same pushed data (push_wait + topk_large_indices + gather_filter_own): the stock xl top-k resolves ties arbitrarily, so
     the comparison is on the multiset of selected VALUES (summed over the 8 columns of a mesh row) and the selected count.
Every device holds different rows (heavy ties, -inf tails, valid counts below and above K), several launches in a row.
Ids encode the list position (unique per column) so the output ids can be mapped back to list positions."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt import hy4_idx_select as IS
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt import hy4_qpush

S = int(os.environ.get("S", "4"))
KL = int(os.environ.get("KLOC", "512"))
K = 2048
W = 8 * KL
REPS = int(os.environ.get("REPS", "18"))
R, C = 4, 8
LEVELS = [30, 300, 8, 2000, 1, 3]
EXACT = (
    os.environ.get("HY4_B36_EXACT", "0") == "1"
)  # stock tie order emulated: device output must equal the stock chain bit for bit (incl. order)


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def tok(cc, j):
    """token id of local row j on column cc (layout invariant): page j // 64, column cc, slot j % 64"""
    return ((j >> 6) << 9) | (cc << 6) | (j & 63)


def local_row(t):
    return ((t >> 9) << 6) | (t & 63)


def host_ref(vals, ci, r, c):
    """vals [R][C] [S,KL] bf16 lists; device (r, c) -> out [8*S, K] int64 (own selected local rows in list order, sentinel tail)"""
    out = torch.full((8 * S, K), 0xFFFFFFFF, dtype=torch.int64)
    for s in range(S):
        v = torch.cat([vals[r][cc][s] for cc in range(C)]).float().tolist()
        order = [i for i in range(W) if v[i] != float("-inf")]
        order.sort(key=lambda i: (-v[i], i))
        sel = set(order[:K])
        nvalid = int(ci[r, c, 0, 2 + s])
        row = []
        for j in range(KL):
            if c * KL + j in sel:
                t = tok(c, j)
                if t < nvalid:
                    row.append(local_row(t))
        row = row + [0xFFFFFFFF] * (K - len(row))
        for jj in range(8):
            out[jj * S + s] = torch.tensor(row, dtype=torch.int64)
    return out


def main():
    md = open_mesh(trace=int(100e6))
    try:
        g = torch.Generator().manual_seed(21)
        m2 = lambda dims: ttnn.ShardTensor2dMesh(md, mesh_shape=(R, C), dims=dims)
        up = lambda t, dt, dims: ttnn.from_torch(
            t,
            dtype=dt,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            mesh_mapper=m2(dims),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        push = hy4_qpush.Push(md, S, KL * 2, max(1, min(4352 // (KL * 2), 4)))
        out_a = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 8 * S, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        out_b = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 8 * S, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        ok_host = ok_stock = True
        for it in range(REPS):
            levels = LEVELS[it % len(LEVELS)]
            ninf = [0, 0, 3000, 1500][(it // 2) % 4] if it % 3 else 0
            small_nv = it % 2 == 0 and it % 4 == 2  # some cases with per-user valid counts that drop selected ids
            nv = torch.randint(1 << 10, 1 << 12, (R,), generator=g) if small_nv else torch.full((R,), 1 << 30)
            ci = HP.colid_host(R, nv, S).to(torch.int32)
            vals = [[None] * C for _ in range(R)]
            lids = [[None] * C for _ in range(R)]
            for r in range(R):
                for cc in range(C):
                    x = torch.randint(0, levels, (S, KL), generator=g).float() / 16 - (
                        2 if it < 12 else 0
                    )  # it >= 12: non-negative values (positive tie groups)
                    x = x.to(torch.bfloat16).float()
                    x, _ = torch.sort(x, dim=1, descending=True)
                    for s in range(S):
                        nin = max(0, min(KL, ninf // 8 + int(torch.randint(-40, 40, (1,), generator=g)))) if ninf else 0
                        if nin:
                            x[s, KL - nin :] = float("-inf")
                    vals[r][cc] = x.to(torch.bfloat16)
                    lids[r][cc] = torch.tensor([[tok(cc, j) for j in range(KL)]] * S)
            vh = torch.cat(
                [torch.cat([vals[r][cc].reshape(1, S, 1, KL) for cc in range(C)], dim=3) for r in range(R)], 0
            )
            vh = vh.reshape(R * S, 1, 1, C * KL)
            lh = torch.cat(
                [torch.cat([lids[r][cc].reshape(1, S, 1, KL) for cc in range(C)], dim=3) for r in range(R)], 0
            )
            lh = lh.reshape(R * S, 1, 1, C * KL)
            vt = up(vh, ttnn.bfloat16, (0, 3))
            lt = up(lh.to(torch.int32), ttnn.uint32, (0, 3))
            cit = up(ci.reshape(R, C, 1, 16), ttnn.uint32, (0, 1))
            # (a) stock chain
            push(vt)
            av = HP.push_wait(push, S)
            pos2 = ttnn.reshape(IS.topk_ids(ttnn.reshape(av, [1, 1, S, W]), K), [S, 1, 1, K])
            HP.gather_filter_own(pos2, lt, cit, S, out_a, True)
            ttnn.synchronize_device(
                md
            )  # all consumers of round (a) are done before any peer re-pushes into the landing tensor
            # (b) sel3
            push(vt)
            HP.sel3_own(push, lt, cit, S, out_b, True, K=K)
            ttnn.synchronize_device(md)
            da, db = ttnn.get_device_tensors(out_a), ttnn.get_device_tensors(out_b)
            nbad_host = nbad_stock = 0
            for r in range(R):
                tot_a = [[] for _ in range(S)]
                tot_b = [[] for _ in range(S)]
                for c in range(C):
                    a = ttnn.to_torch(da[r * C + c]).reshape(8 * S, K).long() & 0xFFFFFFFF
                    b = ttnn.to_torch(db[r * C + c]).reshape(8 * S, K).long() & 0xFFFFFFFF
                    h = host_ref(vals, ci, r, c)
                    if EXACT:
                        if not torch.equal(a, b):
                            nbad_stock += 1
                            if nbad_stock == 1:
                                res(
                                    event="exact_mismatch",
                                    it=it,
                                    r=r,
                                    c=c,
                                    nbad=int((a != b).sum()),
                                    na=int((a[0] != 0xFFFFFFFF).sum()),
                                    nb=int((b[0] != 0xFFFFFFFF).sum()),
                                )
                        continue
                    if not torch.equal(b, h):
                        nbad_host += 1
                        if nbad_host == 1:
                            res(
                                event="host_mismatch",
                                it=it,
                                r=r,
                                c=c,
                                nbad=int((b != h).sum()),
                                nb=int((b[0] != 0xFFFFFFFF).sum()),
                                nh=int((h[0] != 0xFFFFFFFF).sum()),
                            )
                    for s in range(S):
                        ra = a[s][a[s] != 0xFFFFFFFF].tolist()
                        rb = b[s][b[s] != 0xFFFFFFFF].tolist()
                        tot_a[s] += [float(vals[r][c][s][j]) for j in ra]
                        tot_b[s] += [float(vals[r][c][s][j]) for j in rb]
                if not small_nv and not EXACT:
                    for s in range(S):
                        if sorted(tot_a[s]) != sorted(tot_b[s]):
                            nbad_stock += 1
                            res(event="stock_mismatch", it=it, r=r, s=s, na=len(tot_a[s]), nb=len(tot_b[s]))
            ok_host &= nbad_host == 0
            ok_stock &= nbad_stock == 0
            res(
                event="case",
                it=it,
                levels=levels,
                ninf=ninf,
                small_nv=small_nv,
                devs_bad_vs_host=nbad_host,
                rows_bad_vs_stock=nbad_stock,
            )
        res(event="done", ok_vs_host=ok_host, ok_vs_stock_values=ok_stock)
    finally:
        close_mesh(md)


main()
