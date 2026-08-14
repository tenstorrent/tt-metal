import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    x = (torch.arange(32 * 32).reshape(1, 1, 32, 32).float() % 97) - 48
    for th in (32, 16, 8, 1):
        for dt in (ttnn.bfloat8_b, ttnn.bfloat16):
            # pure HOST round trip: torch -> TILE tensor (host tilizer) -> torch
            t = ttnn.from_torch(x.bfloat16(), dtype=dt, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([th, 32]))
            back = ttnn.to_torch(t).float()
            print(
                f"host-loopback th={th:2d} {str(dt):22s} maxdiff={(back-x).abs().max().item():8.3f} first4={back.flatten()[:4].tolist()}"
            )
finally:
    ttnn.close_device(device)
