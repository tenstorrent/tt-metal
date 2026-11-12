import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    mv = ttnn.device.get_memory_view(dev, ttnn.BufferType.L1)
    budget = mv.largest_contiguous_bytes_free_per_bank - pd.L1_SAFETY_MARGIN
    print("KNOBS budget", budget, "grid", dev.compute_with_storage_grid_size())
    for shape, dt, lay, gam in [
        ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ((1, 1, 8192, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ((1, 1, 32, 8192), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, True),
        ((1, 1, 64, 12288), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ((1, 1, 32, 16384), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
    ]:
        x = ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=lay, device=dev)
        g = (
            ttnn.from_torch(torch.zeros(1, 1, 1, shape[-1]), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            if gam
            else None
        )
        geo = pd._Geometry(x, g, x)
        *_, n1, n2 = ttnn.split_work_to_cores(
            dev.compute_with_storage_grid_size(), geo.tensor_rows_tiles, row_wise=True
        )
        try:
            r, c, K = pd._select_blocks(geo, geo.tensor_w_tiles, max(n1, n2), budget)
            print(
                "KNOBS",
                shape,
                dt,
                lay,
                "Rt",
                geo.tensor_rows_tiles,
                "rows_cap",
                max(n1, n2),
                "r,c,K",
                r,
                c,
                K,
                "F",
                pd._footprint(geo, geo.tensor_w_tiles, r, K),
            )
        except NotImplementedError as e:
            print("KNOBS", shape, dt, "NotImplementedError", e)
finally:
    ttnn.close_device(dev)
