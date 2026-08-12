import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
gx, gy = int(grid.x), int(grid.y)
print("grid", gx, gy, "arch", ttnn.get_arch_name())
budget = pd._l1_cb_budget()
print("budget", budget)

for W in (1024, 2304, 5120, 7168):
    for rows in (8192,):
        shape = (1, 1, rows, W)
        ti = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        tg = ttnn.from_torch(
            torch.zeros((1, 1, 1, W), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        geo = pd._Geometry(ti, tg)
        cands = pd._select_candidates(geo, gx, gy, False, budget, pd.MAX_W_GROUP_SIZE)
        cands.sort(key=lambda c: c[0], reverse=True)
        gc, gr, C, R = pd._select_regime(geo, gx, gy, False, budget)
        Gv = gc * gr
        num_groups = (gx // gc) * (gy // gr)
        active = min(geo.tensor_row_tiles, num_groups)
        rows_per = geo.tensor_row_tiles // active
        extra = geo.tensor_row_tiles % active
        maxrow = rows_per + (1 if extra else 0)
        s1, s2 = pd._tree_for_box(Gv, gc, gr)
        fixed, per = pd._cb_bytes(geo, C, s1, False, 0, stage2_span=s2)
        print(
            f"W={W} rowtiles={geo.tensor_row_tiles} wtiles={geo.tensor_w_tiles} -> gc={gc} gr={gr} G={Gv} C={C} R={R} groups={num_groups} active={active} maxrow={maxrow} nblocks={-(-maxrow//R)} fixed={fixed} per_row={per} used={fixed+R*per} slack={budget-(fixed+R*per)}"
        )
        print("   top cands:", [(s, c) for s, c in cands[:6]])
        ti.deallocate()
        tg.deallocate()
ttnn.close_device(device)
