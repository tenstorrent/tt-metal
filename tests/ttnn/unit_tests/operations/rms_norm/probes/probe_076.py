import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
gx = grid.x
print(f"grid=({grid.x},{grid.y}) gx={gx}")


def torch_rms(x, gamma, eps):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(ms + eps)
    if gamma is not None:
        y = y * gamma
    return y


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


eps = 1e-5
results = []
fails = []
for Wt in [329, 331, 513, 334]:
    W = 32 * Wt
    Wt_pad = ((Wt + gx - 1) // gx) * gx
    K = pd._select_k(Wt_pad, 1, grid, grid.x * grid.y, False, ttnn.bfloat16, True, ttnn.bfloat16)
    print(f"Wt={Wt} W={W} Wt_pad={Wt_pad} _select_k(Wt_pad)={K} routes_B={K is not None}")
    for H in [32, 128] if Wt == 329 else [32]:
        for dt, ttdt in [("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)]:
            for layout, ttl in [("TILE", ttnn.TILE_LAYOUT), ("RM", ttnn.ROW_MAJOR_LAYOUT)]:
                for use_g in [False, True]:
                    shape = (1, 1, H, W)
                    x1 = torch.ones(shape, dtype=torch.float32)
                    g = torch.ones((W,), dtype=torch.float32) if use_g else None
                    tx = ttnn.from_torch(x1, dtype=ttdt, layout=ttl, device=device)
                    tg = ttnn.from_torch(g, dtype=ttdt, layout=ttl, device=device) if use_g else None
                    try:
                        out = rms_norm(tx, gamma=tg, epsilon=eps)
                        r = ttnn.to_torch(out).float()
                    except Exception as e:
                        fails.append((Wt, H, dt, layout, use_g, f"EXC ones: {e}"))
                        continue
                    exp1 = torch_rms(x1, g, eps)
                    maxerr = (r - exp1).abs().max().item()
                    xr = torch.randn(shape, dtype=torch.float32)
                    gr = torch.randn((W,), dtype=torch.float32) if use_g else None
                    txr = ttnn.from_torch(xr, dtype=ttdt, layout=ttl, device=device)
                    tgr = ttnn.from_torch(gr, dtype=ttdt, layout=ttl, device=device) if use_g else None
                    try:
                        outr = rms_norm(txr, gamma=tgr, epsilon=eps)
                        rr = ttnn.to_torch(outr).float()
                    except Exception as e:
                        fails.append((Wt, H, dt, layout, use_g, f"EXC rand: {e}"))
                        continue
                    expr = torch_rms(xr, gr, eps)
                    p = pcc(rr, expr).item()
                    ok = (maxerr <= 0.01) and (p >= 0.999)
                    tag = "OK" if ok else "BAD"
                    if not ok:
                        fails.append((Wt, H, dt, layout, use_g, f"maxerr={maxerr:.4f} pcc={p:.5f}"))
                    results.append((Wt, H, dt, layout, use_g, maxerr, p))
                    print(
                        f"  [{tag}] Wt={Wt} H={H} {dt} {layout} g={int(use_g)} ones_maxerr={maxerr:.4f} rand_pcc={p:.5f}"
                    )

print("\n=== SUMMARY ===")
print(f"total cases: {len(results)}, fails: {len(fails)}")
for f in fails:
    print("FAIL", f)
ttnn.close_device(device)
assert len(fails) == 0, f"{len(fails)} failures"
print("ALL PROBE CASES PASS")
