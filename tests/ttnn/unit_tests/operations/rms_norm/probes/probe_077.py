import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
gx = grid.x


def torch_rms(x, gamma, eps):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(ms + eps)
    return y * gamma if gamma is not None else y


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


eps = 1e-5
fails = []
n = 0
for Wt in [329, 331, 513, 334]:
    W = 32 * Wt
    Wt_pad = ((Wt + gx - 1) // gx) * gx
    K = pd._select_k(Wt_pad, 1, grid, grid.x * grid.y, False, ttnn.bfloat16, True, ttnn.bfloat16)
    print(f"ROUTE Wt={Wt} W={W} Wt_pad={Wt_pad} K={K} routes_B={K is not None}")
    for H in [32, 128] if Wt == 329 else [32]:
        for dt, ttdt in [("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)]:
            for layout, ttl in [("TILE", ttnn.TILE_LAYOUT), ("RM", ttnn.ROW_MAJOR_LAYOUT)]:
                for use_g in [False, True]:
                    shape = (1, 1, H, W)
                    x1 = torch.ones(shape, dtype=torch.float32)
                    g = torch.ones((W,), dtype=torch.float32) if use_g else None
                    tx = ttnn.from_torch(x1, dtype=ttdt, layout=ttl, device=device)
                    tg = ttnn.from_torch(g, dtype=ttdt, layout=ttl, device=device) if use_g else None
                    out = rms_norm(tx, gamma=tg, epsilon=eps)
                    r = ttnn.to_torch(out).float()
                    maxerr = (r - torch_rms(x1, g, eps)).abs().max().item()
                    xr = torch.randn(shape, dtype=torch.float32)
                    gr = torch.randn((W,), dtype=torch.float32) if use_g else None
                    txr = ttnn.from_torch(xr, dtype=ttdt, layout=ttl, device=device)
                    tgr = ttnn.from_torch(gr, dtype=ttdt, layout=ttl, device=device) if use_g else None
                    outr = rms_norm(txr, gamma=tgr, epsilon=eps)
                    p = pcc(ttnn.to_torch(outr).float(), torch_rms(xr, gr, eps))
                    ok = (maxerr <= 0.01) and (p >= 0.999)
                    n += 1
                    if not ok:
                        fails.append((Wt, H, dt, layout, use_g, maxerr, p))
                    print(
                        f"  [{'OK' if ok else 'BAD'}] Wt={Wt} H={H} {dt} {layout} g={int(use_g)} maxerr={maxerr:.4f} pcc={p:.5f}"
                    )
print(f"\nSUMMARY cases={n} fails={len(fails)}")
for f in fails:
    print("FAIL", f)
ttnn.close_device(device)
print("ALL PROBE CASES PASS" if not fails else "PROBE HAD FAILURES")
