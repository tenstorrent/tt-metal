import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    x = torch.arange(16 * 32).reshape(1, 1, 16, 32).float()
    for th in (32, 16, 8):
        for dt in (ttnn.bfloat8_b, ttnn.bfloat16):
            t = ttnn.from_torch(
                x.bfloat16(),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                tile=ttnn.Tile([th, 32]),
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # a plain device eltwise: unpack -> DEST -> pack, same packer path, nothing to do with tilize
            try:
                y = ttnn.mul(t, 1.0)
                back = ttnn.to_torch(y).float()
                print(
                    f"eltwise mul th={th:2d} {str(dt):22s} maxdiff={(back-x).abs().max().item():8.1f} first4={[round(v.item(),2) for v in back.flatten()[:4]]}"
                )
            except Exception as e:
                print(f"eltwise mul th={th:2d} {str(dt):22s} EXC {type(e).__name__}: {str(e)[:120]}")
finally:
    ttnn.close_device(device)
