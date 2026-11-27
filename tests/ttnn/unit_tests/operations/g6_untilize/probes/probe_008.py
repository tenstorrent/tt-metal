import torch, ttnn
from ttnn.operations.g6_untilize import g6_untilize, describe_plan
from ttnn.operations.g6_untilize import g6_untilize_program_descriptor as pd


def rnd(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randint(-(2**31), 2**31, shape, dtype=torch.int64, generator=g)
    return p.to(torch.int32).view(torch.uint32)


device = ttnn.open_device(device_id=0)
knob_names = ("CORE_LIMIT", "ROW_BLOCK_DEPTH", "BANK_STAGGER", "GATHER_RISCS", "L1_OP_BUDGET_BYTES")
defaults = {k: getattr(pd, k) for k in knob_names}
ok_all = True
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    cases = [
        ((1, 1, 32, 128), {}),
        ((1, 1, 32, 128), {"GATHER_RISCS": 1}),
        ((1, 1, 32, 128), {"BANK_STAGGER": False}),
        ((1, 1, 32, 128), {"CORE_LIMIT": 4}),
        ((1, 1, 32, 128), {"CORE_LIMIT": 1}),
        ((2, 3, 40, 100), {}),
        ((1, 2, 17, 33), {}),
        ((1, 1, 1, 16), {}),
        ((1, 1, 96, 136), {"CORE_LIMIT": 1}),
        ((1, 1, 96, 136), {"CORE_LIMIT": 1, "GATHER_RISCS": 1}),
        ((1, 1, 96, 136), {"CORE_LIMIT": 3, "ROW_BLOCK_DEPTH": 1}),
        ((3, 1, 96, 4096), {}),
        ((1, 1, 64, 5440), {}),
        ((1, 1, 64, 5440), {"CORE_LIMIT": 2}),
        ((1, 1, 32, 1024), {"L1_OP_BUDGET_BYTES": 3 * 2 * 4096}),
    ]
    for shape, knobs in cases:
        for k, v in defaults.items():
            setattr(pd, k, v)
        for k, v in knobs.items():
            setattr(pd, k, v)
        x = rnd(shape)
        xd = ttnn.from_torch(
            x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        plan = describe_plan(xd)
        y = ttnn.to_torch(g6_untilize(xd))
        good = list(y.shape) == list(shape) and torch.equal(y.contiguous().view(torch.int32), x.view(torch.int32))
        ok_all &= good
        print(
            "CASE",
            "OK " if good else "BAD",
            shape,
            knobs,
            "cores",
            plan["num_cores"],
            "btc",
            plan["block_tile_cols"],
            "depth",
            plan["buffer_depth"],
        )
finally:
    for k, v in defaults.items():
        setattr(pd, k, v)
    ttnn.close_device(device)
print("ALL", ok_all)
