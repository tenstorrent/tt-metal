import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_hy4_decode_merge_device as T
from dm_block import *

full, md = open_row()
try:
    S = int(os.environ.get("S", "1"))
    c = T.Case(md, "random", 65536, S, "bf16", seed=3)
    outs = {}
    for tile in (False, True):
        blk = Block(md, S, v2=True, tile=tile)
        idx8 = blk.idx_rows(c.idx_dev)
        qg = blk.gather_q(c.q_dev)
        out = blk.kernel(qg, c.kv_dev, idx8)
        t = blk.to_owner(out)
        outs[tile] = from_mesh(md, t).float()
        print("shape", tile, tuple(t.shape), tuple(outs[tile].shape), flush=True)
    a, b = outs[False], outs[True]  # [8c, 8j, ?, 544]
    print("A", a.shape, "B", b.shape)
    a = a.reshape(8, 8, 1 if S == 1 else S, -1, 544) if False else a
    # normalise to [c, j, s, h, 544]
    A = a.reshape(8, 8, 8, S, 544)  # RM owner: [8c][8j][8h*S rows (h,s)] -> j,(h,s)
    A = A.reshape(8, 8, 8, S, 544).permute(0, 1, 3, 2, 4)  # c,j,s,h
    B = b.reshape(8, 8, S, -1, 544)[:, :, :, :8]  # c,j,s,h
    print("max abs diff o", (A[..., :512] - B[..., :512]).abs().max().item())
    print(
        "o A",
        A[0, 0, 0, 0, :6].tolist(),
        "o B",
        B[0, 0, 0, 0, :6].tolist(),
        "nonzero B",
        (B != 0).float().mean().item(),
        "A",
        (A != 0).float().mean().item(),
    )
    d = (A[..., :512] - B[..., :512]).abs().amax(-1)  # c,j,s,h
    print("mismatch map (c,j,h) max over cols, s=0:")
    for cc in range(2):
        print(" c", cc, [[round(v, 3) for v in row] for row in d[cc, :, 0, :].tolist()])
    dd = (A[..., :512] - B[..., :512]).abs()[0, 0, 0, 0]
    print("cols mismatching in c0 j0 h0:", (dd > 1e-3).nonzero().flatten().tolist()[:40])
    print("tail A", A[0, 0, 0, :3, 512:516].tolist(), "tail B", B[0, 0, 0, :3, 512:516].tolist())
    print("diff tails", (A[..., 512:515] - B[..., 512:515]).abs().max().item())
finally:
    close_row(full, md)
