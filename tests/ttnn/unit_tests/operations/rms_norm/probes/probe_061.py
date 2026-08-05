import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod

device = ttnn.open_device(device_id=0)
print("tree arity:", {g: pdmod._combine_tree_arity(g, 1) for g in (4, 8, 9, 16, 28, 29, 30, 31, 32, 33, 56)})


def rel_rms(o, r):
    return (torch.sqrt(torch.mean((o - r) ** 2)) / (torch.sqrt(torch.mean(r**2)) + 1e-30)).item()


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def golden(x, g, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * g.float()).to(torch.bfloat16)


eps = 1e-6
# --- WIDTH-sharded: the tree's own regime, incl. RAGGED level-0 runs and many rounds ---
CASES = [
    ((1, 1, 32, 5120), [32, 160], (8, 4), "G32  f1=8 TREE      "),
    ((1, 1, 32, 4800), [32, 160], (6, 5), "G30  f1=8 TREE ragged"),
    ((1, 1, 32, 4960), [32, 160], (31, 1), "G31 f1=8 TREE ragged"),  # may fall back if grid.x<31
    ((1, 1, 32, 5280), [32, 160], (11, 3), "G33  f1=9 TREE ragged"),
    ((1, 1, 3200, 4800), [32, 160], (6, 5), "G30  TREE many-round"),
    ((1, 1, 32, 7168), [32, 256], (7, 4), "G28       flat       "),
]
for shape, ss, grid, label in CASES:
    try:
        torch.manual_seed(7)
        x = torch.randn(shape, dtype=torch.bfloat16)
        g = torch.randn((shape[-1],), dtype=torch.bfloat16)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, ss, ttnn.ShardOrientation.ROW_MAJOR),
        )
        xt = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        gt = ttnn.from_torch(g.reshape(1, 1, 1, -1), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.to_torch(ttnn.rms_norm(xt, weight=gt, epsilon=eps))
        ref = golden(x, g, eps)
        print(f"{label} {str(shape):20s} pcc={pcc(o, ref):.7f} rel_rms={rel_rms(o.float(), ref.float()):.6f}")
    except Exception as e:
        print(f"{label} {str(shape):20s} SKIP/ERR: {type(e).__name__}: {str(e)[:120]}")

# --- interleaved width split forced to a wide group (the tree at gw=32 and 56) ---
saved = pdmod.GRID_W
for gw in (32, 56):
    try:
        pdmod.GRID_W = gw
        shape = (1, 1, 32, 7168)
        torch.manual_seed(3)
        x = torch.randn(shape, dtype=torch.bfloat16)
        g = torch.randn((7168,), dtype=torch.bfloat16)
        xt = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        gt = ttnn.from_torch(g.reshape(1, 1, 1, -1), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.to_torch(ttnn.rms_norm(xt, weight=gt, epsilon=eps))
        ref = golden(x, g, eps)
        print(f"interleaved GRID_W={gw:3d} pcc={pcc(o, ref):.7f} rel_rms={rel_rms(o.float(), ref.float()):.6f}")
    finally:
        pdmod.GRID_W = saved
ttnn.close_device(device)
