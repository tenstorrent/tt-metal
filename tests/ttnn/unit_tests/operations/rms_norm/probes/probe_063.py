import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod


def rel_rms(o, r):
    return (torch.sqrt(torch.mean((o - r) ** 2)) / (torch.sqrt(torch.mean(r**2)) + 1e-30)).item()


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


eps = 1e-6
saved = pdmod.COMBINE_TREE_MIN_DELETED_FOLD_TILES
# fp32 in/out so the STAT's own precision is visible instead of bf16 output quantization.
CASES = [((1, 1, 32, 5120), [32, 160], (8, 4), "G32"), ((1, 1, 32, 4800), [32, 160], (6, 5), "G30_ragged")]
for shape, ss, grid, label in CASES:
    res = {}
    for name, thresh in (("flat", 10**9), ("tree", 18)):
        device = ttnn.open_device(device_id=0)
        try:
            pdmod.COMBINE_TREE_MIN_DELETED_FOLD_TILES = thresh
            torch.manual_seed(101)
            x = torch.randn(shape, dtype=torch.float32)
            g = torch.randn((shape[-1],), dtype=torch.float32)
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, ss, ttnn.ShardOrientation.ROW_MAJOR),
            )
            xt = ttnn.from_torch(x, ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            gt = ttnn.from_torch(g.reshape(1, 1, 1, -1), ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi2
            cfg.fp32_dest_acc_en = False
            cfg.math_approx_mode = False
            o = ttnn.to_torch(ttnn.rms_norm(xt, weight=gt, epsilon=eps, compute_kernel_config=cfg)).float()
            ref = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + eps) * g.double()
            ref = ref.float()
            res[name] = (
                pcc(o, ref),
                rel_rms(o, ref),
                pdmod._combine_tree_arity(grid[0] * grid[1], 1) if thresh == 18 else None,
            )
        finally:
            pdmod.COMBINE_TREE_MIN_DELETED_FOLD_TILES = saved
            ttnn.close_device(device)
    for name in ("flat", "tree"):
        p, r, t = res[name]
        print(f"{label:11s} {name:4s} arity={str(t):8s} pcc={p:.9f} rel_rms={r:.7f}")
