import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)

    def report(shape, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, gamma=True):
        t = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
            dtype=dtype,
            layout=layout,
            device=device,
        )
        gm = None
        if gamma:
            gm = ttnn.from_torch(
                torch.zeros((1,) * (len(shape) - 1) + (shape[-1],), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        geo = pd._Geometry(t, gm)
        budget = pd._l1_cb_budget()
        gc, gr, C, R = pd._select_regime(geo, int(g.x), int(g.y), layout == ttnn.ROW_MAJOR_LAYOUT, budget)
        Gs = gc * gr
        ngroups = (int(g.x) // gc) * (int(g.y) // gr)
        active = min(geo.tensor_row_tiles, ngroups)
        print(
            f"{str(shape):22s} Rt={geo.tensor_row_tiles:5d} Wt={geo.tensor_w_tiles:5d} "
            f"group={gc}x{gr} G={Gs:3d} rowgroups={active}/{ngroups} C={C:4d} R={R:3d} "
            f"cores={active*Gs:4d} regime={'R1' if Gs==1 else 'R2'}"
        )
        ttnn.deallocate(t)
        if gm is not None:
            ttnn.deallocate(gm)

    for s in [
        (1, 1, 32, 32),
        (1, 1, 64, 128),
        (1, 1, 32, 2048),
        (1, 1, 32, 8192),
        (1, 1, 32, 16384),
        (1, 1, 64, 12288),
        (1, 1, 8192, 64),
        (1, 1, 4096, 128),
        (1, 1, 8192, 1024),
        (1, 1, 32, 7168),
        (1, 1, 8192, 7168),
        (512, 64),
        (4, 128, 320),
        (2, 1, 100, 1023),
    ]:
        report(s)
finally:
    ttnn.close_device(device)
