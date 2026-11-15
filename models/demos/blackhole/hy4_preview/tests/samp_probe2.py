"""device-sampling probe 2: DeviceLoop.derive on the mesh vs the host helpers (write rows, key counts, col info, ids, upd, rope), fake decoder shell (no model)."""
import json
import types

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_device_loop import NC, ROWS, DeviceLoop, K
from models.demos.blackhole.hy4_preview.tt.hy4_rope import rope_tables


def res(**kw):
    print("RES " + json.dumps(kw, default=str), flush=True)


def shell(md, interleave, first):
    up = lambda t, dt, lay=ttnn.ROW_MAJOR_LAYOUT, **k: upload(md, t, dt, lay, **k)
    z = torch.zeros
    ctx = types.SimpleNamespace(
        wr=up(HP.write_rows([0] * ROWS, ROWS), ttnn.int32, shard_dim=1, row_dim=0),
        upd=up(z(ROWS, NC, dtype=torch.int32), ttnn.int32, shard_dim=1, row_dim=0),
        kvcnt=up(HP.key_counts([0] * ROWS, ROWS), ttnn.uint32, shard_dim=1, row_dim=0),
        colinfo=up(HP.colid_host(ROWS, [1] * ROWS), ttnn.uint32, shard_dim=1, row_dim=0),
        ids_g=up(torch.full((ROWS, 1, 1, K), -1, dtype=torch.int32), ttnn.uint32, row_dim=0),
    )
    rope = types.SimpleNamespace(dim=64, theta=1e7, interleave=interleave)
    attn = types.SimpleNamespace(rope=rope, indexer=types.SimpleNamespace(idx_rope_first=first))
    ch = types.SimpleNamespace(layers=[types.SimpleNamespace(attn=attn)])
    dec = types.SimpleNamespace(
        md=md,
        ch=ch,
        ctx=ctx,
        tok=up(z(ROWS, 1, 1, dtype=torch.int32), ttnn.uint32, row_dim=0),
        cos=up(z(ROWS, 1, 1, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0),
        sin=up(z(ROWS, 1, 1, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0),
        c128=up(z(ROWS, 1, 1, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0),
        s128=up(z(ROWS, 1, 1, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0),
    )
    return dec


def rd(md, t, row, col=0):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[row * NC + col])


def main():
    md = open_mesh(trace=50_000_000)
    try:
        for interleave, first in ((True, False), (False, True)):
            dec = shell(md, interleave, first)
            dl = DeviceLoop(dec)
            poses = [
                [0, 1, 63, 64],
                [511, 512, 513, 2047],
                [2048, 4095, 65535, 65536],
                [262143, 524287, 1048575, 1048574],
            ]
            torch.manual_seed(3)
            poses += [torch.randint(0, 1 << 20, (4,)).tolist() for _ in range(6)]
            bad = {"wr": 0, "upd": 0, "kv": 0, "ci": 0, "ids": 0}
            rope_err = 0.0
            exact = 0
            tot = 0
            for ps in poses:
                dl.load_inputs([1, 2, 3, 4], ps)
                dl.derive()
                ttnn.synchronize_device(md)
                wr_h = HP.write_rows(ps, ROWS)
                kv_h = HP.key_counts(ps, ROWS)
                ci_h = HP.colid_host(ROWS, [p + 1 for p in ps])
                for r in range(ROWS):
                    for c in range(NC):
                        wr = rd(md, dec.ctx.wr, r, c).reshape(-1).int()
                        bad["wr"] += int((wr != wr_h[r, c, 0]).any())
                        bad["upd"] += int(rd(md, dec.ctx.upd, r, c).reshape(-1)[0] != wr_h[r, c, 0, 0])
                        bad["kv"] += int((rd(md, dec.ctx.kvcnt, r, c).reshape(-1).int() != kv_h[r, c, 0]).any())
                        bad["ci"] += int((rd(md, dec.ctx.colinfo, r, c).reshape(-1).int() != ci_h[r, c, 0]).any())
                    ids = rd(md, dec.ctx.ids_g, r).reshape(-1).int()
                    n = min(ps[r] + 1, K)
                    want = torch.full((K,), -1, dtype=torch.int32)
                    want[:n] = torch.arange(n, dtype=torch.int32)
                    bad["ids"] += int((ids != want).any())
                    cs = rope_tables(torch.tensor([ps[r]]), 64, 1e7, interleave)
                    cos = rd(md, dec.cos, r).reshape(-1).float()
                    sin = rd(md, dec.sin, r).reshape(-1).float()
                    # fp64 truth
                    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
                    ang = ps[r] * inv
                    c64 = ang.cos().repeat_interleave(2) if interleave else torch.cat([ang.cos()] * 2)
                    s64 = ang.sin().repeat_interleave(2) if interleave else torch.cat([ang.sin()] * 2)
                    pe = max(float((cos.double() - c64).abs().max()), float((sin.double() - s64).abs().max()))
                    if pe > 0.01:
                        res(event="rope_bad", pos=ps[r], err=pe, cos_dev=cos[:4].tolist(), cos_true=c64[:4].tolist())
                    rope_err = max(
                        rope_err, float((cos.double() - c64).abs().max()), float((sin.double() - s64).abs().max())
                    )
                    if ps[r] < 2048:
                        tot += 1
                        exact += int(
                            torch.equal(cos.bfloat16(), cs[0].reshape(-1).bfloat16())
                            and torch.equal(sin.bfloat16(), cs[1].reshape(-1).bfloat16())
                        )
                    c128 = rd(md, dec.c128, r).reshape(-1).float()
                    want128 = torch.cat([torch.ones(64), cos] if first else [cos, torch.ones(64)])
                    bad["c128"] = bad.get("c128", 0) + int(not torch.equal(c128, want128))
            res(
                event="derive_vs_host",
                interleave=interleave,
                first=first,
                mismatches=bad,
                rope_max_abs_err_vs_fp64_after_bf16=rope_err,
                rope_bf16_equal_host_rows_lt2048=f"{exact}/{tot}",
                positions=poses,
            )
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
