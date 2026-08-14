import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn
from ttnn.operations.rms_norm.perf_experiments.scale_gamma_fusion.scale_gamma_bench import (
    plan_for,
    run_variant,
    sharded_memory_config,
    dest_block_for,
    GRID,
)

TILE = 32
NC = GRID[0] * GRID[1]


def build(device, plan):
    r, s = plan["row_tiles"], plan["S"]
    rows, width = NC * r * TILE, s * TILE
    torch.manual_seed(11)
    x = (torch.rand(rows, width) * 2 - 1).to(torch.bfloat16)
    stat = torch.rand(rows, TILE) * 2 - 1
    stat[:, 0] = torch.rand(rows) + 0.5
    gfull = (torch.rand(NC * TILE, width) * 2 - 1).to(torch.bfloat16)
    g = (torch.rand(width) + 0.5).to(torch.bfloat16)
    gfull[0::TILE, :] = g
    q = lambda t: t.to(torch.bfloat16).to(torch.float32)
    xf, gf = q(x), q(g)
    scaled = xf * stat[:, 0:1].to(torch.float32)  # x*(1/rms)   (no gamma)
    expected = scaled * gf.unsqueeze(0)

    def dev(t, dt):
        return ttnn.from_torch(
            t,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_memory_config((t.shape[0] // NC, t.shape[1])),
        )

    xd = dev(x, ttnn.bfloat16)
    sd = dev(stat.to(torch.float32), ttnn.float32)
    gd = dev(gfull, ttnn.bfloat16)
    od = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, sharded_memory_config((r * TILE, width))
    )
    return xd, sd, gd, od, expected, scaled, gf


def report(actual, expected, scaled, gf, plan, tag):
    r, s = plan["row_tiles"], plan["S"]
    # per-tile relative error, core 0 shard
    a = actual[: r * TILE]
    e = expected[: r * TILE]
    sc = scaled[: r * TILE]
    print(f"\n##### {tag}  (R={r} S={s} dest_blk={plan['blk']})")
    print("per-tile max |a-e| / rms(e)   rows=tile_row, cols=tile_col")
    hdr = "      " + "".join(f"{c:>9}" for c in range(s))
    print(hdr)
    bad = []
    for tr in range(r):
        row = []
        for tc in range(s):
            aa = a[tr * TILE : (tr + 1) * TILE, tc * TILE : (tc + 1) * TILE]
            ee = e[tr * TILE : (tr + 1) * TILE, tc * TILE : (tc + 1) * TILE]
            rel = (aa - ee).abs().max().item() / (ee.pow(2).mean().sqrt().item() + 1e-9)
            row.append(rel)
            if rel > 0.1:
                bad.append((tr, tc))
        if tr < 6:
            print(f"r{tr:<4} " + "".join(f"{v:>9.3f}" for v in row))
    print(
        f"bad tiles: {len(bad)} of {r*s}; tile_cols hit = {sorted({c for _, c in bad})}; "
        f"tile_rows hit = {sorted({t for t, _ in bad})[:20]}"
    )

    if bad:
        tr, tc = bad[0]
        aa = a[tr * TILE : (tr + 1) * TILE, tc * TILE : (tc + 1) * TILE]
        ee = e[tr * TILE : (tr + 1) * TILE, tc * TILE : (tc + 1) * TILE]
        ss = sc[tr * TILE : (tr + 1) * TILE, tc * TILE : (tc + 1) * TILE]
        print(f"\nfirst bad tile (r{tr},c{tc}) -- per-face(16x16) max rel err:")
        for fr in (0, 1):
            print(
                "   "
                + "".join(
                    f"{(aa[fr*16:(fr+1)*16, fc*16:(fc+1)*16] - ee[fr*16:(fr+1)*16, fc*16:(fc+1)*16]).abs().max().item()/ (ee.pow(2).mean().sqrt().item()+1e-9):>9.3f}"
                    for fc in (0, 1)
                )
            )
        print(f"   a[0,:4]={aa[0,:4].tolist()}  e[0,:4]={ee[0,:4].tolist()}")
        print(f"   zero frac in a: {(aa == 0).float().mean().item():.3f}")
        # hypothesis: gamma from a different column?
        cand = []
        for c2 in range(s):
            alt = ss * gf[c2 * TILE : (c2 + 1) * TILE].unsqueeze(0)
            cand.append(((aa - alt).abs().max().item(), c2))
        cand.sort()
        print(
            f"   best-matching gamma column: {cand[0][1]} (err {cand[0][0]:.4g}); "
            f"true col {tc} err {[e for e, c in cand if c == tc][0]:.4g}"
        )
        print(f"   err vs 'no gamma' (x*1/rms): {(aa - ss).abs().max().item():.4g}")
        # per-row-of-tile ratio a/e, to see structure
        rat = aa / (ee + 1e-30)
        print(f"   a/e row means (32): {[round(v,3) for v in rat.mean(1).tolist()]}")


device = ttnn.open_device(device_id=0)
try:
    for b, s, cap in ((1, 4, 8), (1, 5, 8), (1, 6, 8), (1, 7, 8), (1, 8, 8), (1, 8, 4), (1, 8, 7), (2, 8, 8)):
        plan = plan_for(b, s)
        plan["blk"] = dest_block_for("dest_srca", s, cap)
        x, st, g, o, expected, scaled, gf = build(device, plan)
        run_variant(x, st, g, o, variant="dest_srca", plan=plan, dest_cap=cap)
        ttnn.synchronize_device(device)
        act = ttnn.to_torch(o).to(torch.float32)
        report(act, expected, scaled, gf, plan, f"dest_srca B={b} S={s} dest_cap={cap}")
        for t in (x, st, g, o):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
