import torch, ttnn
from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_scatter_swiglu.program_descriptor import (
    TILE,
    build_layout,
    make_sharded_config,
    run_variant,
)

device = ttnn.open_device(device_id=0)
try:
    K, M, HN, NCOLS = 10, 8, 6, 11
    layout = build_layout(K, M, HN, NCOLS)
    T = M * HN
    n = K * NCOLS
    cfg = make_sharded_config(device, K, T, NCOLS)
    # gate shard s == constant s ; up shard s == constant 1  =>  h == silu(sum_col s) * 10
    g = torch.empty((n * TILE, T * TILE))
    u = torch.ones((n * TILE, T * TILE))
    for s in range(n):
        g[s * TILE : (s + 1) * TILE] = float(s)
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    ut = ttnn.from_torch(u, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    ht = ttnn.from_torch(
        torch.zeros((n * TILE, T * TILE)),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    run_variant(device, gt, ut, ht, "tree", layout)
    got = ttnn.to_torch(ht).float()
    nz = [
        (o, float(got[o * TILE : (o + 1) * TILE].mean()))
        for o in range(n)
        if float(got[o * TILE : (o + 1) * TILE].abs().max()) > 0
    ]
    print("NONZERO OUTPUT SHARDS (index, mean/10 ~= sum of that column's gate constants):")
    print([(o, round(v / 10, 1)) for o, v in nz])
    print("row_major expects roots at shards 0..10 with sums", [495 + 10 * c for c in range(NCOLS)])
    print(
        "col_major expects roots at shards",
        [c * K for c in range(NCOLS)],
        "with sums",
        [100 * c + 45 for c in range(NCOLS)],
    )
finally:
    ttnn.close_device(device)
