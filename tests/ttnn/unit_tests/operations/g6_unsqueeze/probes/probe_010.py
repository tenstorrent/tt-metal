import sys, torch, ttnn
from ttnn.operations.g6_unsqueeze import g6_unsqueeze, g6_unsqueeze_plan

op_mod = sys.modules["ttnn.operations.g6_unsqueeze.g6_unsqueeze"]


def rand_bits(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int64, generator=g)
        .to(torch.int32)
        .view(torch.float32)
    )


device = ttnn.open_device(device_id=0)
fails = 0
try:
    shapes = [(1, 512), (40,), (1, 1, 100), (1, 33), (1, 1, 32 * 131 + 7), (1, 32 * 301)]
    # (max_mover_cores, movers_per_core, staging_depth, block_w_tiles_cap)
    knobs = [(1, 2, 1, 3), (2, 2, 2, 2), (3, 1, 1, 3), (3, 1, 2, 2), (5, 2, 2, 1), (4, 1, 1, 1), (7, 2, 2, 5)]
    for knob in knobs:
        mmc, mpc, depth, cap = knob
        op_mod.MAX_MOVER_CORES, op_mod.MOVERS_PER_CORE, op_mod.STAGING_DEPTH = mmc, mpc, depth
        op_mod.OP_L1_BUDGET_BYTES = cap * 2 * depth * 1024
        for shape in shapes:
            x = rand_bits(shape, (hash((knob, shape)) & 0xFFFF))
            t = ttnn.from_torch(
                x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            plan = g6_unsqueeze_plan(t)
            out = ttnn.to_torch(g6_unsqueeze(t, -1))
            exp = torch.unsqueeze(x, -1)
            ok = list(out.shape) == list(exp.shape) and torch.equal(out.view(torch.int32), exp.view(torch.int32))
            nb = max((cw + plan.block_w_tiles - 1) // plan.block_w_tiles for _, _, cw in plan.core_work)
            if not ok:
                fails += 1
            print(
                f"knob={knob} shape={shape} cores={plan.num_cores} ranges={len(plan.core_ranges)} bw={plan.block_w_tiles} cap={plan.block_w_tiles_cap} max_blocks={nb} bit_exact={ok}"
            )
    op_mod.MAX_MOVER_CORES, op_mod.MOVERS_PER_CORE, op_mod.STAGING_DEPTH, op_mod.OP_L1_BUDGET_BYTES = (
        None,
        2,
        1,
        1347776,
    )
    t = ttnn.from_torch(
        rand_bits((1, 512), 5),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print(g6_unsqueeze_plan(t).report())
finally:
    ttnn.close_device(device)
print("TOTAL FAILS", fails)
