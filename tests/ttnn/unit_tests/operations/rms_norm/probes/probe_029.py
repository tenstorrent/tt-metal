import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
dev = ttnn.open_device(device_id=0)
for shape, gam, osh in [
    ((1, 1, 64, 128), False, False),
    ((1, 1, 64, 128), True, False),
    ((1, 1, 64, 128), False, True),
    ((1, 1, 64, 128), True, True),
]:
    x = torch.randn(shape, dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), HS, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    print(mc.shard_spec)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev) if gam else None
    out = rms_norm(tx, gamma=tg, memory_config=tx.memory_config() if osh else ttnn.DRAM_MEMORY_CONFIG)
    res = ttnn.to_torch(out).float()
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    if gam:
        ref = ref * g.float()
    err = (res - ref).abs().amax(-1).flatten()
    print(shape, gam, osh, "bad rows", [i for i in range(err.numel()) if err[i] > 0.1][:70])
ttnn.close_device(dev)
