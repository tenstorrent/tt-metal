import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print("grid:", grid.x, grid.y, "arch:", ttnn.get_arch_name(), "budget:", pd._l1_cb_budget())

import types


def geo_for(Rt, W):
    Wt = (W + 31) // 32
    return types.SimpleNamespace(
        shape=[1, 1, Rt * 32, W],
        W=W,
        tensor_w_tiles=Wt,
        partial_w=W % 32,
        tensor_row_tiles=Rt,
        is_rm_in=False,
        num_sticks=0,
        in_dtype=ttnn.bfloat16,
        in_elem_bytes=2,
        in_tile_bytes=2048,
        has_gamma=True,
        gamma_dtype=ttnn.bfloat16,
        gamma_elem_bytes=2,
        gamma_tile_bytes=2048,
        is_rm_gamma=False,
    )


gx, gy = int(grid.x), int(grid.y)
for Rt, W in [(1, 7168), (2, 12288), (3, 6144), (256, 7168), (256, 1024), (256, 2304)]:
    g = geo_for(Rt, W)
    gc, gr, C, R = pd._select_regime(g, gx, gy, False, pd._l1_cb_budget())
    G = gc * gr
    num_groups = (gx // gc) * (gy // gr)
    ag = min(Rt, num_groups)
    crt = -(-Rt // ag)
    print(
        f"Rt={Rt:4d} W={W:6d} -> G={G:3d} C={C:4d} R={R:3d} groups={num_groups:3d} "
        f"active={ag*G:3d}/{gx*gy} core_row_tiles={crt} num_blocks={-(-crt//R)}"
    )
ttnn.close_device(device)
