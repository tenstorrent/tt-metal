# SPDX-License-Identifier: Apache-2.0
"""B37: select_v2 end to end, blocked (HY4_IDX_BLK layout) vs plain, at the 1M shape (n_local 131072, B=16, kb=512, kloc=512, S=4 verify rows, per-row kv ~ NKEYS+s), ag = 8x concat of the own data (all columns identical)."""
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
        S, B, n, kb, kloc = 4, int(os.environ.get("B", "16")), int(os.environ.get("N", "131072")), 512, 512
        g = torch.Generator().manual_seed(2)
        nk = int(os.environ.get("NKEYS", "100"))
        keys = torch.zeros(1, 1, n, 128)
        keys[:, :, :nk] = torch.randn(1, 1, nk, 128, generator=g)
        q = torch.randn(S, 1, 32, 128, generator=g)
        w = torch.randn(1, 1, 32, 32, generator=g)
        kvl = [nk + s for s in range(S)]
        kvt = rep(
            md, torch.tensor(kvl + [0] * (16 - S), dtype=torch.int32).reshape(1, 16), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32
        )
        kt = rep(md, keys, ttnn.TILE_LAYOUT, ttnn.bfloat8_b)
        qt = rep(md, q, ttnn.TILE_LAYOUT, ttnn.bfloat16)
        wt = rep(md, w, ttnn.TILE_LAYOUT, ttnn.bfloat16)
        colid = rep(md, torch.zeros(1, 1, 1, 16, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        ag = lambda t: ttnn.concat([t] * 8, dim=3)
        outs = {}
        for blk in (0, 1):
            sc = IS.idx_score(qt, kt, wt, 0, S, kvt=kvt, n_rows=n, kvt_rows=True, blocks=B if blk else 1)
            ids = SEL.select_v2(ag, colid, sc, n, B, kb, kloc, 2048, 4, False, None, blocked=bool(blk))
            outs[blk] = dev0(ids).reshape(S, 2048).to(torch.int64)
        for s in range(S):
            a, b = outs[0][s], outs[1][s]
            va = a[a < 2**31]
            vb = b[b < 2**31]
            res(
                event="row",
                s=s,
                n_plain=int(va.numel()),
                n_blk=int(vb.numel()),
                equal=bool((a == b).all()),
                expect_keys=kvl[s],
            )
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
