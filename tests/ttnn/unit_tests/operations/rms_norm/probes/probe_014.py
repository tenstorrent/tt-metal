import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
try:
    shape = (64, 64)
    x = torch.ones(shape, dtype=torch.bfloat16)
    g = torch.arange(shape[-1], dtype=torch.float32).reshape(1, 1, 1, -1).to(torch.bfloat16)
    for lay, name in [(ttnn.ROW_MAJOR_LAYOUT, "RM"), (ttnn.TILE_LAYOUT, "TILE")]:
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tg = ttnn.from_torch(
            g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        res = ttnn.to_torch(rms_norm(tx, gamma=tg)).to(torch.float32)
        print(name, "row0[0:16]", res[0, :16].tolist())
        print(name, "row0[16:32]", res[0, 16:32].tolist())
        print(name, "row0[32:48]", res[0, 32:48].tolist())
        print(name, "row1[0:8]", res[1, :8].tolist())
        print(name, "row33[0:8]", res[33, :8].tolist())
finally:
    ttnn.close_device(device)
