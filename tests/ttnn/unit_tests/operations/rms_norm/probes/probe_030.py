import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
gx, gy = int(grid.x), int(grid.y)
budget = pd._l1_cb_budget()

shapes = [
    (1, 1, 8192, 1024),
    (1, 1, 8192, 2304),
    (1, 1, 8192, 5120),
    (1, 1, 8192, 7168),
    (1, 1, 32, 1024),
    (1, 1, 32, 2304),
    (1, 1, 32, 5120),
    (1, 1, 32, 7168),
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (1, 1, 128, 256),
    (1, 1, 1024, 1024),
    (1, 1, 256, 512),
    (1, 1, 3232, 96),
    (99991, 64),
    (1, 1, 992, 3000),
    (1, 1, 4096, 8192),
    (3, 1, 736, 5119),
    (1, 1, 17, 47),
    (1, 1, 50, 50),
    (2, 3, 64, 128),
    (1, 1, 2048, 2048),
    (1, 1, 4096, 4096),
    (1, 1, 256, 8192),
    (1, 1, 1024, 4096),
    (1, 1, 8192, 8192),
    (1, 1, 2048, 11008),
    (13, 777, 1023),
    (1, 1, 512, 2048),
    (1, 1, 8192, 4096),
    (1, 1, 16384, 1024),
]
res = {}
for slack in (None, 15):
    pd.BALANCE_SLACK_PCT = slack
    for shape in shapes:
        ti = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        tg = ttnn.from_torch(
            torch.zeros((1, 1, 1, shape[-1]), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        geo = pd._Geometry(ti, tg)
        gc, gr, C, R = pd._select_regime(geo, gx, gy, False, budget)
        res.setdefault(shape, []).append((gc * gr, C, R))
        ti.deallocate()
        tg.deallocate()
print("shape                     phase0(G,C,R)   banded(G,C,R)   CHANGED")
for shape, (a, b) in res.items():
    print(f"{str(shape):24s}  {str(a):14s} {str(b):14s} {'<<< CHANGED' if a != b else ''}")
ttnn.close_device(device)
