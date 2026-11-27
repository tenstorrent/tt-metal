import torch, ttnn
from ttnn.operations.g6_concat import g6_concat, select_plan

device = ttnn.open_device(device_id=0)
print(
    "grid",
    device.compute_with_storage_grid_size(),
    "dram_align",
    ttnn.get_dram_alignment(),
    "l1_align",
    ttnn.get_l1_alignment(),
)
try:
    a = torch.randn(1, 1, 64, 64).to(torch.bfloat16)
    b = torch.randn(1, 1, 64, 64).to(torch.bfloat16)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(select_plan([ta, tb], -1))
    out = g6_concat([ta, tb], -1)
    got = ttnn.to_torch(out)
    exp = torch.cat([a, b], -1)
    print("match", torch.equal(got.view(torch.int16), exp.view(torch.int16)))
finally:
    ttnn.close_device(device)
