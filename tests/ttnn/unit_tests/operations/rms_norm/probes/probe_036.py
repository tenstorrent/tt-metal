import torch, ttnn, os
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()


def blocking(shape, budget):
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    g = ttnn.from_torch(
        torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ht, wt = pd._tile_geometry(x)
    p = pd._select_placement(device, grid, x, ht, wt, False)
    b = pd._derive_blocking(x, g, grid.x * grid.y, p, l1_total_budget=budget)
    ttnn.deallocate(x)
    ttnn.deallocate(g)
    return (b.wt_chunk, b.nw, b.ht_block, b.x_res_depth, b.gamma_resident, pd._x_read_chunks(b), b.cb_total_bytes)


live = pd._l1_total_budget(device)
print("live budget", live, " guessed", pd.L1_CB_BUDGET_BYTES)
print(f"{'shape':22s} {'guessed (C,NW,H,xdep,gres,rdbat,bytes)':46s} {'live'}")
for shape in [
    (1, 1, 8192, 1024),
    (1, 1, 8192, 2304),
    (1, 1, 8192, 5120),
    (1, 1, 8192, 7168),
    (1, 1, 32, 1024),
    (1, 1, 32, 7168),
]:
    a = blocking(shape, pd.L1_CB_BUDGET_BYTES)
    b = blocking(shape, live)
    mark = "  <-- CHANGED" if a != b else ""
    print(f"{str(shape):22s} {str(a):46s} {b}{mark}")
ttnn.close_device(device)
