import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod
from ttnn.operations.rms_norm import rms_norm


def rel_rms(o, r):
    return (torch.sqrt(torch.mean((o - r) ** 2)) / (torch.sqrt(torch.mean(r**2)) + 1e-30)).item()


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


eps = 1e-6
saved = pdmod.COMBINE_TREE_MIN_DELETED_FOLD_TILES
CASES = [((1, 1, 32, 5120), [32, 160], (8, 4), "G32"), ((1, 1, 32, 4800), [32, 160], (6, 5), "G30_ragged")]
for dt, fp32acc, tag in ((ttnn.float32, True, "fp32/HiFi2/acc=T"), (ttnn.bfloat16, False, "bf16/HiFi2/acc=F")):
    for shape, ss, grid, label in CASES:
        out = {}
        for name, thresh in (("flat", 10**9), ("tree", 18)):
            device = ttnn.open_device(device_id=0)
            try:
                pdmod.COMBINE_TREE_MIN_DELETED_FOLD_TILES = thresh
                torch.manual_seed(101)
                tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
                x = torch.randn(shape, dtype=tdt)
                g = torch.randn((shape[-1],), dtype=tdt)
                crs = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))]
                )
                mc = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(crs, ss, ttnn.ShardOrientation.ROW_MAJOR),
                )
                xt = ttnn.from_torch(x, dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                gt = ttnn.from_torch(g.reshape(1, 1, 1, -1), dt, layout=ttnn.TILE_LAYOUT, device=device)
                cfg = ttnn.ComputeConfigDescriptor()
                cfg.math_fidelity = ttnn.MathFidelity.HiFi2
                cfg.fp32_dest_acc_en = fp32acc
                cfg.math_approx_mode = False
                o = ttnn.to_torch(
                    rms_norm(xt, gamma=gt, epsilon=eps, compute_kernel_config=cfg, memory_config=mc)
                ).float()
                xd = x.double()
                ref = (xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps) * g.double()).float()
                out[name] = (pcc(o, ref), rel_rms(o, ref))
            finally:
                pdmod.COMBINE_TREE_MIN_DELETED_FOLD_TILES = saved
                ttnn.close_device(device)
        (pf, rf), (pt, rt) = out["flat"], out["tree"]
        print(
            f"{tag} {label:11s} flat pcc={pf:.9f} rel_rms={rf:.7f} | tree pcc={pt:.9f} rel_rms={rt:.7f}"
            f"  -> tree/flat rel_rms = {rt/max(rf,1e-30):.4f}"
        )
