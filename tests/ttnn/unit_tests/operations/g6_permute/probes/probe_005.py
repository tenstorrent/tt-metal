import torch, ttnn
from ttnn.operations.g6_permute import describe_plan

device = ttnn.open_device(device_id=0)
shapes = [
    (1, 1, 16, 64),
    (1, 16, 1, 64),
    (1, 6, 16, 64),
    (1, 16, 6, 64),
    (1, 1, 8, 64),
    (1, 8, 1, 64),
    (1, 6, 8, 64),
    (1, 8, 6, 64),
    (1, 1, 1, 32),
    (2, 3, 5, 96),
    (1, 70, 3, 64),
    (1, 32, 20, 640),
    (3, 40, 50, 512),
    (1, 5, 47, 32),
]
for s in shapes:
    x = ttnn.from_torch(
        torch.zeros(s, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    p = describe_plan(x, (0, 2, 1, 3))
    per = sorted({c["num_blocks"] for c in p["cores"]})
    print(
        "PLAN",
        s,
        {k: p[k] for k in ("block_a", "block_b", "num_blocks", "num_cores", "depth_in", "depth_staging", "regroup")},
        "blocks/core",
        per,
    )
ttnn.close_device(device)
