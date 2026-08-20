import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as D
from ttnn.operations.rms_norm._bench_rms_norm import BENCH_SHAPES, BENCH_GAMMA_LAYOUT, GATESET, BENCH_CONFIGS

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print("GRID", grid.x, grid.y)
print("virt x row0:", [D._virt(device, x, 0) for x in range(grid.x)])
print("virt y col0:", [D._virt(device, 0, y) for y in range(grid.y)])
print("l1_cb_budget", ttnn.get_max_worker_l1_unreserved_size() - D.L1_RESERVED_BYTES)


def plan_for(name, config, dtype, gamma_present, levers=None):
    shape, sdt, layout = BENCH_SHAPES[name]
    dt = dtype or sdt
    x = ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=layout, device=device)
    g = None
    if gamma_present:
        gl = BENCH_GAMMA_LAYOUT.get(name) or (ttnn.TILE_LAYOUT if dt == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT)
        g = ttnn.from_torch(torch.zeros((1, 1, 1, shape[-1])), dtype=dt, layout=gl, device=device)
    cfg = BENCH_CONFIGS[config]()
    p = D.blocking_plan(x, g, x, device, cfg, levers)
    ttnn.deallocate(x)
    if g is not None:
        ttnn.deallocate(g)
    return p


for name, config, dtype, gm in GATESET:
    p = plan_for(name, config, dtype, gm)
    tag = f"{name}/{config}" + (f"/{str(dtype).split('.')[-1]}" if dtype else "") + ("" if gm else "/no_gamma")
    print(
        f"{tag:36s} Rt={p.Rt:4d} Wt={p.Wt:5d} G={p.group_size:4d} rect={p.group_x}x{p.group_y} "
        f"ngrp={p.num_groups:3d} used={p.groups_used:3d} Wt_core={p.Wt_core:4d} reg={p.regime} "
        f"bht={p.BLOCK_HT} nrb={p.num_row_blocks:4d} wr={p.WT_REDUCE_BLOCK} L1={p.working_set_bytes()}"
    )

ttnn.close_device(device)
