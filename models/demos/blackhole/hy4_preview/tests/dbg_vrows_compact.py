"""debug: compact vs plain owner-layout tile output of the split kernel (same inputs): the 8 real head rows of user s must sit at rows 8*(s%4).. of tile row s//4"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
import test_hy4_decode_merge_device as DM
from dm_block import NC, close_row, from_mesh, open_row

import ttnn


def main():
    full, md = open_row()
    try:
        for S in [int(x) for x in os.environ.get("SS", "3,4").split(",")]:
            c = DM.Case(md, "real", 65536, S, "scaled", seed=3)
            blk = DM.mk_block(md, S, fuse=True, xchg=False)
            idx8 = blk.idx_rows(c.idx_dev)
            qg = blk.gather_q(c.q_dev)
            on = blk.mod.sparse_sdpa_split(
                qg, c.kv_dev, idx8, 512, DM.SCALE, blk.kc, blk.G, blk.ckc, None, emit_stats=3, owner_s=S
            )
            oc = blk.mod.sparse_sdpa_split(
                qg, c.kv_dev, idx8, 512, DM.SCALE, blk.kc, blk.G, blk.ckc, None, emit_stats=3, owner_s=S, compact=True
            )
            ttnn.synchronize_device(md)
            TR = (S + 3) // 4
            n = from_mesh(md, on).float().reshape(NC, NC, S, 32, 576)  # [dev, j, s, 32, 576]
            m = from_mesh(md, oc).float().reshape(NC, NC, TR, 32, 576)
            for s in range(S):
                u, b = s // 4, s % 4
                a = n[:, :, s, :8, :]
                bb = m[:, :, u, 8 * b : 8 * b + 8, :]
                # per column tile error
                err = (a - bb).abs()
                bad_cols = [ct for ct in range(16) if err[..., 32 * ct : 32 * ct + 32].max() > 0] + [
                    ct for ct in (16, 17) if err[..., 32 * ct].max() > 0
                ]
                # stats tiles: only column 0 is meaningful
                print(
                    "RES",
                    dict(
                        S=S,
                        s=s,
                        max_err=float(max(err[..., :512].max(), err[..., 512].max(), err[..., 544].max())),
                        bad_col_tiles=bad_cols,
                        finite=bool(torch.isfinite(a).all()),
                    ),
                    flush=True,
                )
            # a2a + merge: plain (S tile rows) vs compact (TR tile rows)
            import hy4_merge

            def a2a(x):
                return ttnn.experimental.all_to_all_async_generic(
                    x, in_dim=1, out_dim=0, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1
                )

            xn, xc = a2a(on), a2a(oc)
            rn = hy4_merge.merge_fused(xn, c.sinkrow3_dev, S)
            sink_c = torch.zeros(NC, TR, 32, 1)
            for cc in range(NC):
                for b in range(4):
                    sink_c[cc, :, 8 * b : 8 * b + 8, 0] = c.sl[8 * cc : 8 * cc + 8][None, :]
            sink_cd = DM.to_mesh(md, sink_c, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            rc = hy4_merge.merge_fused(xc, sink_cd, TR, full=S > 2)
            ttnn.synchronize_device(md)
            Rn = from_mesh(md, rn).float().reshape(NC, S, 32, 512)
            Rc = from_mesh(md, rc).float().reshape(NC, TR, 32, 512)
            xs_n = from_mesh(md, xn).float().reshape(NC, NC, S, 32, 576)
            xs_c = from_mesh(md, xc).float().reshape(NC, NC, TR, 32, 576)
            for s in range(S):
                u, b = s // 4, s % 4
                e = (Rn[:, s, :8] - Rc[:, u, 8 * b : 8 * b + 8]).abs().max()
                ex = (xs_n[:, :, s, :8, :512] - xs_c[:, :, u, 8 * b : 8 * b + 8, :512]).abs().max()
                es = (xs_n[:, :, s, :8, 512] - xs_c[:, :, u, 8 * b : 8 * b + 8, 512]).abs().max()
                print(
                    "RES",
                    dict(stage="a2a_merge", S=S, s=s, merge_err=float(e), x_o_err=float(ex), x_ms_err=float(es)),
                    flush=True,
                )
            c.free()
    finally:
        close_row(full, md)


main()
