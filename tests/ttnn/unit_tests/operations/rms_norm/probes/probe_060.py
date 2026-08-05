import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod

device = ttnn.open_device(device_id=0)


def rel_rms(out, ref):
    return (torch.sqrt(torch.mean((out - ref) ** 2)) / (torch.sqrt(torch.mean(ref**2)) + 1e-30)).item()


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def golden(x, g, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * g.float()).to(torch.bfloat16)


CASES = [
    # (shape, shard_shape, grid, label)  -- the tree targets and their neighbours
    ((1, 1, 32, 5120), [32, 160], (8, 4), "w5120_32c_G32_TREE"),
    ((1, 1, 32, 7168), [32, 256], (7, 4), "w7168_28c_G28_TREE"),
    ((1, 1, 32, 1024), [32, 128], (8, 1), "w1024_8c_G8_flat"),
    ((1, 1, 32, 2304), [32, 256], (9, 1), "w2304_9c_G9_flat"),
]
eps = 1e-6
for shape, ss, grid, label in CASES:
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
    out = ttnn.rms_norm(xt, weight=gt, epsilon=eps)
    o = ttnn.to_torch(out)
    ref = golden(x, g, eps)
    print(f"{label:24s} pcc={pcc(o, ref):.7f}  rel_rms={rel_rms(o.float(), ref.float()):.6f}")

# BLOCK-sharded 64c (G=8, BLOCK_ROWS>1 -> compact, flat)
torch.manual_seed(11)
shape = (1, 1, 8192, 1024)
x = torch.randn(shape, dtype=torch.bfloat16)
g = torch.randn((1024,), dtype=torch.bfloat16)
crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
mc = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(crs, [1024, 128], ttnn.ShardOrientation.ROW_MAJOR),
)
xt = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
gt = ttnn.from_torch(g.reshape(1, 1, 1, -1), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
o = ttnn.to_torch(ttnn.rms_norm(xt, weight=gt, epsilon=eps))
ref = golden(x, g, eps)
print(f"{'bshard_64c_G8_flat':24s} pcc={pcc(o, ref):.7f}  rel_rms={rel_rms(o.float(), ref.float()):.6f}")
ttnn.close_device(device)
