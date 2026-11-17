# SPDX-License-Identifier: Apache-2.0
"""B35 unit: gather_filter (cs_gather + id_filter fused; optionally with the id all-gather replaced by the push + flag wait) vs a host model of the stock chain (ids = cand[pos] / sentinel, id_filter semantics).
Every device holds different local ids; several launches in a row (monotonic flag counters), all bit-exact."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt import hy4_qpush

S = int(os.environ.get("S", "4"))
KL = int(os.environ.get("KLOC", "512"))
K = 2048
W = 8 * KL
REPS = int(os.environ.get("REPS", "6"))
R, C = 4, 8


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def ref(pos, lids, colinfo, r, c):
    """host model for device (r, c): pos [S,K] int, lids [R][C] -> [S,KL] int64, colinfo [R,C,16] -> out [OWN*S, K] int64"""
    out = torch.full((8 * S, K), 0xFFFFFFFF, dtype=torch.int64)
    for s in range(S):
        cand = torch.cat([lids[r][cc][s] for cc in range(C)]).long()
        ids = [(int(cand[p]) if p < W else 0xFFFFFFFF) for p in pos[s].tolist()]
        nvalid = int(colinfo[r, c, 0, 2 + s])
        row = []
        for t in ids:
            if t < nvalid and ((t >> 6) & 7) == c:
                row.append(((t >> 9) << 6) | (t & 63))
        row = row + [0xFFFFFFFF] * (K - len(row))
        for j in range(8):
            out[j * S + s] = torch.tensor(row, dtype=torch.int64)
    return out


def main():
    md = open_mesh(trace=int(100e6))
    try:
        g = torch.Generator().manual_seed(11)
        m2 = lambda dims: ttnn.ShardTensor2dMesh(md, mesh_shape=(R, C), dims=dims)
        up = lambda t, dims: ttnn.from_torch(
            t,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            mesh_mapper=m2(dims),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        push = (
            hy4_qpush.Push(md, S, KL * 4, max(1, min(4352 // (KL * 4), 2)))
            if os.environ.get("PUSH", "1") == "1" and os.environ.get("OWN") != "1"
            else None
        )
        out_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 8 * S, K]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        ok_all = True
        for it in range(REPS):
            nv = torch.randint(1 << 10, 1 << 16, (R,), generator=g) if it % 2 == 0 else torch.full((R,), 1 << 30)
            ci = HP.colid_host(R, nv, S).to(torch.int32)  # [R, C, 1, 16]
            # local ids: global token ids spread over [0, 2^17) with some invalid
            lids = [
                [
                    (torch.randint(0, 1 << 8, (S, KL), generator=g) << 9)
                    | (cc << 6)
                    | torch.randint(0, 64, (S, KL), generator=g)
                    for cc in range(C)
                ]
                for _ in range(R)
            ]  # layout invariant: column cc candidates are tokens of column cc
            pos = torch.randint(0, W + 64, (R, C, S, K), generator=g)  # some positions >= W
            lids_h = torch.cat(
                [torch.cat([lids[r][c].reshape(1, S, 1, KL) for c in range(C)], dim=3) for r in range(R)], dim=0
            )
            lids_h = lids_h.reshape(R * S, 1, 1, C * KL)
            pos_h = pos.permute(0, 2, 1, 3).reshape(R * S, 1, 1, C * K)
            lt = up(lids_h.to(torch.int32), (0, 3))
            pt = up(pos_h.to(torch.int32), (0, 3))
            cit = up(ci.reshape(R, C, 1, 16), (0, 1))
            if os.environ.get("OWN") == "1":
                if os.environ.get("NQ"):
                    HP.gather_filter_own(pt, lt, cit, S, out_t, True, nq=int(os.environ["NQ"]))
                else:
                    HP.gather_filter(pt, lt, cit, S, out_t, True, own=True)
                src = "own"
            elif push is not None:
                land = push(lt)
                HP.gather_filter(pt, land, cit, S, out_t, True, push=push)
                src = "push"
            else:
                # gathered candidates built on host
                ai = torch.cat(
                    [torch.cat([lids[r][c] for c in range(C)], dim=1).reshape(1, S, 1, W) for r in range(R)], dim=0
                )
                ai = ai.reshape(R * S, 1, 1, W)
                ait = up(torch.cat([ai] * C, dim=3).to(torch.int32), (0, 3))
                HP.gather_filter(pt, ait, cit, S, out_t, True)
                src = "gathered"
            ttnn.synchronize_device(md)
            dts = ttnn.get_device_tensors(out_t)
            eq = True
            for r in range(R):
                for c in range(C):
                    got = ttnn.to_torch(dts[r * C + c]).reshape(8 * S, K).long() & 0xFFFFFFFF
                    exp = ref(pos[r, c], lids, ci, r, c)
                    if not torch.equal(got, exp):
                        eq = False
                        res(event="mismatch", it=it, r=r, c=c, nbad=int((got != exp).sum()))
                        break
                if not eq:
                    break
            ok_all &= eq
            res(event="case", it=it, src=src, equal=eq)
        res(event="done", ok=ok_all)
    finally:
        close_mesh(md)


main()
