import torch, ttnn
from ttnn.operations.onorm import onorm
import ttnn.operations.onorm.onorm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    grid = device.compute_with_storage_grid_size()
    print(f"grid = {grid.x} x {grid.y} = {grid.x*grid.y} cores")
    print(f"CB-available L1/core = {ttnn.get_max_worker_l1_unreserved_size()} B")

    for B, T in [(1, 32), (1, 64), (1, 128), (1, 640), (2, 64), (3, 96), (8, 640)]:
        nblocks = B * ((T + pd.TOKENS_PER_BLOCK - 1) // pd.TOKENS_PER_BLOCK)
        ncores, all_cores, assign = pd._grid_assignment(device, nblocks)
        per = sorted({n for _, _, n in assign})
        print(
            f"B={B} T={T:4d}: {nblocks:3d} token-blocks -> {ncores:2d} cores "
            f"(blocks/core {per}), ranges={all_cores.num_cores()}"
        )

    # CB footprint report at phase-1 knobs
    o = ttnn.from_torch(
        torch.randn(1, 32, 32, 128, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    g = ttnn.from_torch(
        torch.randn(1, 32, 4096, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, 1, 128, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = onorm(o, g, w)
    print("smoke ok", list(out.shape))
finally:
    ttnn.close_device(device)
