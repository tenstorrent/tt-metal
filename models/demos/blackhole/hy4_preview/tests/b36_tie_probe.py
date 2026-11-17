# SPDX-License-Identifier: Apache-2.0
"""B36 probe: semantics of the stock topk_large_indices as used by the indexer select (k=512 / 2048 over [S,1,1,W] bf16 rows): output order, tie rule, -inf handling.
Rows = 8 sorted-desc lists of 512 (as the gathered local candidates), heavy ties. Prints the match of the device output against the host models (value desc, pos asc) / (value desc, pos desc)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

S = 4
KL = int(os.environ.get("KLOC", "512"))
K = int(os.environ.get("K", "2048"))
W = 8 * KL


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def model(vals, k, hi_first):
    """vals bf16 [W] -> top-k positions: value desc (-inf dropped), ties by position asc (hi_first False) / desc"""
    v = vals.float().tolist()
    idx = [i for i in range(len(v)) if v[i] != float("-inf")]
    idx.sort(key=lambda i: (-v[i], -i if hi_first else i))
    return idx[:k]


def main():
    md = open_mesh(trace=int(100e6))
    try:
        g = torch.Generator().manual_seed(5)
        for case, (levels, ninf) in enumerate([(40, 0), (400, 0), (40, 3000), (10, 0), (40, 2000)]):
            lists = []
            for s in range(S):
                row = []
                for c in range(8):
                    x = torch.randint(0, levels, (KL,), generator=g).float() / 8 - 2
                    x = x.to(torch.bfloat16).float()
                    x, _ = torch.sort(x, descending=True)
                    ninfc = max(0, min(KL, ninf // 8 + int(torch.randint(-30, 30, (1,), generator=g))))
                    if ninfc:
                        x[KL - ninfc :] = float("-inf")
                    row.append(x.to(torch.bfloat16))
                lists.append(torch.cat(row))
            vals = torch.stack(lists).reshape(S, 1, 1, W)
            t = ttnn.from_torch(
                vals,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ids = ttnn.experimental.topk_large_indices(ttnn.reshape(t, [1, 1, S, W]), k=K)
            ttnn.synchronize_device(md)
            got = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(S, K).long() & 0xFFFFFFFF
            torch.save(
                dict(vals=vals, got=got),
                os.path.join(os.environ.get("OUT", "/tmp"), f"tie{os.environ.get('PTAG', '')}_case{case}.pt"),
            )
            for s in range(S):
                nval = int((vals[s].flatten() != float("-inf")).sum())
                ma, mb = model(vals[s].flatten(), K, False), model(vals[s].flatten(), K, True)
                gl = got[s].tolist()
                n_inv = sum(1 for x in gl if x >= W)
                ga = gl[: len(ma)]
                eqa, eqb = ga == ma, ga == mb
                seta = set(ga) == set(ma)
                setb = set(ga) == set(mb)
                vv = vals[s].flatten().float()
                sel_idx = [x for x in gl if x < W]
                sv = vv[torch.tensor(sel_idx)] if sel_idx else vv[:0]
                mono = bool((sv[1:] <= sv[:-1]).all()) if len(sel_idx) > 1 else True
                mset = sorted(sv.tolist()) == sorted(vv[torch.tensor(ma)].tolist()) if ma else True
                res(
                    event="row",
                    monotone_desc=mono,
                    value_multiset_eq_model=mset,
                    nsel=len(sel_idx),
                    case=case,
                    s=s,
                    nvalid=nval,
                    n_invalid_out=n_inv,
                    order_pos_asc=eqa,
                    order_pos_desc=eqb,
                    set_pos_asc=seta,
                    set_pos_desc=setb,
                    first_bad=next((i for i in range(len(ma)) if ga[i] != ma[i]), -1),
                    tail=gl[len(ma) : len(ma) + 3],
                )
        res(event="done")
    finally:
        close_mesh(md)


main()
