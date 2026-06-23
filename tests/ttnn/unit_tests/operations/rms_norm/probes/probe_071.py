import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

dev = ttnn.open_device(device_id=0)
try:
    grid = dev.compute_with_storage_grid_size()
    # routing check (pure host)
    k = desc._select_k(256, 16, grid, grid.x * grid.y, False, ttnn.bfloat16, True, ttnn.bfloat16)
    print(f"ROUTING (512,8192): _select_k -> K={k} (expect 4 sub-row)")
    print("CHECK_BEGIN")
    shapes = [(1, 1, 512, 8192), (1, 1, 512, 4096), (1, 1, 1024, 8192)]
    for shp in shapes:
        for dt in [ttnn.bfloat16, ttnn.float32]:
            tdt = torch.float32
            # all-ones -> output ~1.0
            xo = torch.ones(shp, dtype=tdt)
            ti = ttnn.from_torch(xo, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            o = ttnn.to_torch(rms_norm(ti)).float()
            ones_err = (o - 1.0).abs().max().item()
            # random -> PCC vs torch
            xr = torch.randn(shp, dtype=tdt)
            ti2 = ttnn.from_torch(xr, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            o2 = ttnn.to_torch(rms_norm(ti2)).float()
            ref = xr / torch.sqrt((xr * xr).mean(-1, keepdim=True) + 1e-6)
            pcc = torch.corrcoef(torch.stack([o2.flatten(), ref.flatten()]))[0, 1].item()
            print(f"  {str(shp):18s} {str(dt):16s} all_ones_maxerr={ones_err:.4f}  PCC={pcc:.5f}")
    print("CHECK_END")
finally:
    ttnn.close_device(dev)
