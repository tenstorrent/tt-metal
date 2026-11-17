import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
for shape in [(1504, 544), (7168, 1024)]:
    x = torch.randn(shape, dtype=torch.bfloat16)
    mc = auto_shard_config(
        list(shape), ttnn.TensorMemoryLayout.BLOCK_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev
    )
    print(shape, mc.shard_spec.shape, mc.shard_spec.grid)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    out = ttnn.to_torch(rms_norm(tx, memory_config=mc)).float()
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    err = (out - ref).abs().amax(-1)
    bad = (err > 0.1).nonzero().flatten()
    print("bad tile rows", sorted(set((bad // 32).tolist())), "of", shape[0] // 32)
    # col tile check
    errc = (out - ref).abs().amax(0)
    badc = (errc > 0.1).nonzero().flatten()
    print("bad col tiles", sorted(set((badc // 32).tolist())))
ttnn.close_device(dev)
