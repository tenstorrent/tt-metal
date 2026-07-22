import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import _pick_block_size

# Confirm the host work-split spreads a large-R shape across the whole grid.
dev = device
gs = dev.compute_with_storage_grid_size()
print("grid:", gs.x, "x", gs.y, "=", gs.x * gs.y, "cores")

for shape in [(1, 1, 2048, 256), (1, 1, 32, 8192), (4, 8, 64, 512)]:
    NC = 1
    for d in shape[:-2]:
        NC *= d
    Htimg = (shape[-2] + 31) // 32
    R = NC * Htimg
    Wt = (shape[-1] + 31) // 32
    _, all_cores, g1, g2, p1, p2 = ttnn.split_work_to_cores(gs, R, row_wise=True)
    ncores = len(ttnn.corerange_to_cores(all_cores, None, True))
    print(
        f"shape={shape} R={R} Wt={Wt} BLOCK_SIZE={_pick_block_size(Wt)} -> cores_used={ncores} (g1={p1}/core, g2={p2}/core)"
    )

# Run on a large shape (R=64 fills grid; W=256) and check correctness.
ti = torch.randn(1, 1, 2048, 256, dtype=torch.float32)
x = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
g = ttnn.from_torch(
    torch.randn(256).reshape(1, 1, 1, 256),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=dev,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
out = ttnn.to_torch(rms_norm(x, gamma=g)).to(torch.float32)
var = ti.pow(2).mean(-1, keepdim=True)
exp = ti * torch.rsqrt(var + 1e-6) * torch.randn(0) if False else ti * torch.rsqrt(var + 1e-6)
tg = ttnn.to_torch(g).to(torch.float32).reshape(-1)
exp = ti * torch.rsqrt(var + 1e-6) * tg
pcc = torch.corrcoef(torch.stack([out.flatten(), exp.flatten()]))[0, 1].item()
print("large-shape (1,1,2048,256) pcc:", round(pcc, 5))
