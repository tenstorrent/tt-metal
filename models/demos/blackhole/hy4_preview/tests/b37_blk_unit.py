# SPDX-License-Identifier: Apache-2.0
"""B37: HY4_IDX_BLK blocked layout vs plain: idx_score output bytes, stage-1 top-k + cs_cand2 outputs, at the 1M pool shape (n_local 131072, B=16, S=4, per-row kv lengths ~770)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, dev0, open_mesh, rep
from models.demos.blackhole.hy4_preview.tt import hy4_idx_score as IS
from models.demos.blackhole.hy4_preview.tt import hy4_idx_select as SEL


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    md = open_mesh()
    try:
        S, B, n, kb = 4, int(os.environ.get("B", "16")), int(os.environ.get("N", "131072")), 512
        m = n // B
        g = torch.Generator().manual_seed(1)
        nk = int(os.environ.get("NKEYS", "1024"))
        keys = torch.zeros(1, 1, n, 128)
        keys[:, :, :nk] = torch.randn(1, 1, nk, 128, generator=g)
        q = torch.randn(S, 1, 32, 128, generator=g)
        w = torch.randn(1, 1, 32, 32, generator=g)
        kvl = [770 + s for s in range(S)]
        kvt = rep(
            md, torch.tensor(kvl + [0] * (16 - S), dtype=torch.int32).reshape(1, 16), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        )
        kt = rep(md, keys, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)
        qt = rep(md, q, ttnn.TILE_LAYOUT, ttnn.bfloat16)
        wt = rep(md, w, ttnn.TILE_LAYOUT, ttnn.bfloat16)
        colid = rep(md, torch.zeros(1, 1, 1, 16, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        outs = {}
        for blk in (0, 1):
            sc = IS.idx_score(qt, kt, wt, 0, S, kvt=kvt, n_rows=n, kvt_rows=True, blocks=B if blk else 1)
            scb = dev0(sc).reshape(S, n).float()
            outs[blk] = {"sc": scb}
            rows = ttnn.reshape(sc, [1, 1, S * B, m])
            bids = SEL.topk_ids(rows, kb, m)
            bvals, bgids = SEL.cs_cand2(
                ttnn.reshape(sc, [S * B, 1, 1, m]),
                ttnn.reshape(bids, [S * B, 1, 1, kb]),
                colid,
                1,
                m,
                B,
                4,
                blocked_out=bool(blk),
            )
            outs[blk]["bids"] = dev0(bids).reshape(S * B, kb)
            outs[blk]["bvals"] = dev0(bvals).reshape(S, B * kb).float()
            outs[blk]["bgids"] = dev0(bgids).reshape(S, B * kb).to(torch.int64)
        for k in ("sc", "bids", "bvals", "bgids"):
            a, b = outs[0][k], outs[1][k]
            res(event="cmp", what=k, equal=bool((a == b).all()), ndiff=int((a != b).sum()), shape=list(a.shape))
        # valid-count per row among the stage-1 candidates (score > -inf)
        for blk in (0, 1):
            v = outs[blk]["bvals"]
            res(event="valid", blk=blk, per_row=[int((v[s] > -1e30).sum()) for s in range(S)], expect=kvl)
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
