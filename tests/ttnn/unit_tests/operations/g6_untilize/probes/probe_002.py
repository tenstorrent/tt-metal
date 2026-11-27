import torch, ttnn
from ttnn.operations.g6_untilize import g6_untilize, describe_plan
from ttnn.operations.g6_untilize import g6_untilize_program_descriptor as pd


def rnd(shape):
    p = torch.randint(-(2**31), 2**31, shape, dtype=torch.int64)
    return p.to(torch.int32).view(torch.uint32)


device = ttnn.open_device(device_id=0)
print("grid", device.compute_with_storage_grid_size())
ok = True
cases = [
    ((1, 1, 32, 128), {}),
    ((2, 3, 40, 100), {}),
    ((2, 3, 40, 100), {"CORE_BUDGET": 7}),
    ((1, 2, 64, 160), {"CORE_BUDGET": 5, "BLOCK_TILE_COLS_REQ": 2}),
    ((1, 1, 96, 136), {"CORE_BUDGET": 3, "BLOCK_TILE_COLS_REQ": 3, "SCRATCH_DEPTH": 3}),
    ((1, 1, 32, 128), {"BLOCK_TILE_COLS_REQ": 1}),
]
defaults = {k: getattr(pd, k) for k in ("CORE_BUDGET", "BLOCK_TILE_COLS_REQ", "SCRATCH_DEPTH")}
for shape, knobs in cases:
    for k, v in defaults.items():
        setattr(pd, k, v)
    for k, v in knobs.items():
        setattr(pd, k, v)
    x = rnd(shape)
    tx = ttnn.from_torch(
        x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = g6_untilize(tx)
    got = ttnn.to_torch(out)
    good = torch.equal(got.contiguous().view(torch.int32), x.view(torch.int32))
    ok &= good
    plan = describe_plan(tx)
    print("CASE", shape, knobs, "OK" if good else "MISMATCH", out.layout)
    print("PLAN", plan)
for k, v in defaults.items():
    setattr(pd, k, v)
print("ALL", ok)
ttnn.close_device(device)
