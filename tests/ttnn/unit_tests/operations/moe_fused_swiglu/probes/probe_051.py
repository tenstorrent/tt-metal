import torch, ttnn
from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_scatter_swiglu.program_descriptor import (
    TILE,
    build_layout,
    make_sharded_config,
    run_variant,
)


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d else 0.0


device = ttnn.open_device(device_id=0)
try:
    K, M, HN, NCOLS = 10, 8, 6, 11
    layout = build_layout(K, M, HN, NCOLS)
    T = M * HN
    n = K * NCOLS
    cfg = make_sharded_config(device, K, T, NCOLS)
    g = torch.empty((n * TILE, T * TILE))
    u = torch.empty((n * TILE, T * TILE))
    for s in range(n):
        g[s * TILE : (s + 1) * TILE] = 0.11 * (s + 1)
        u[s * TILE : (s + 1) * TILE] = 0.07 * (n - s)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    ut = ttnn.from_torch(u, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    ht = ttnn.from_torch(
        torch.zeros((n * TILE, T * TILE)),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    gq = ttnn.to_torch(gt).float()
    uq = ttnn.to_torch(ut).float()
    run_variant(device, gt, ut, ht, "tree", layout)
    got = ttnn.to_torch(ht).float()

    for name, idx in (
        ("row_major (row*ncols+col)", lambda c, r: r * NCOLS + c),
        ("col_major (col*k+row)", lambda c, r: c * K + r),
    ):
        worst = 1.0
        for c in range(NCOLS):
            gs = sum(gq[idx(c, r) * TILE : (idx(c, r) + 1) * TILE] for r in range(K))
            us = sum(uq[idx(c, r) * TILE : (idx(c, r) + 1) * TILE] for r in range(K))
            ref = torch.nn.functional.silu(gs) * us
            p = pcc(got[idx(c, 0) * TILE : (idx(c, 0) + 1) * TILE], ref)
            worst = min(worst, p)
            print(f"  {name}: col {c} pcc {p:.6f}")
        print(f"{name}: MIN = {worst:.6f}")
finally:
    ttnn.close_device(device)
