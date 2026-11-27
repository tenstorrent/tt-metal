import torch, ttnn
from ttnn.operations.g6_untilize import g6_untilize, describe_plan
from ttnn.operations.g6_untilize import g6_untilize_program_descriptor as pd


def rnd(shape):
    p = torch.randint(-(2**31), 2**31, shape, dtype=torch.int64, generator=torch.Generator().manual_seed(0))
    return p.to(torch.int32).view(torch.uint32)


device = ttnn.open_device(device_id=0)
ok = True
try:
    shape = (1, 1, 32, 1024)
    x = rnd(shape)
    xd = ttnn.from_torch(
        x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for budget in (pd.L1_OP_BUDGET_BYTES, 3 * 2 * 4096, pd.L1_OP_BUDGET_BYTES):
        pd.L1_OP_BUDGET_BYTES = budget
        before = device.num_program_cache_entries()
        y = ttnn.to_torch(g6_untilize(xd))
        good = torch.equal(y.contiguous().view(torch.int32), x.view(torch.int32))
        ok &= good
        print(
            "CASE",
            "OK " if good else "BAD",
            budget,
            "btc",
            describe_plan(xd)["block_tile_cols"],
            "new programs",
            device.num_program_cache_entries() - before,
        )
finally:
    pd.L1_OP_BUDGET_BYTES = 1347776
    ttnn.close_device(device)
print("ALL", ok)
