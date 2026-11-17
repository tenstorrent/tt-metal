import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
dev = ttnn.open_device(device_id=0)
shape = (1, 1, 64, 128)
x = torch.randn(shape, dtype=torch.bfloat16)
mc = auto_shard_config(list(shape), HS, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=dev)
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
print("roundtrip ok", torch.equal(ttnn.to_torch(tx), x))
out = rms_norm(tx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
res = ttnn.to_torch(out).float()
ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
for r in [0, 1, 2, 12]:
    print(r, "res", res[0, 0, r, :6].tolist(), "ref", ref[0, 0, r, :6].tolist(), "x", x[0, 0, r, :6].float().tolist())
    print("   ratio", (res[0, 0, r, :6] / ref[0, 0, r, :6]).tolist())
ttnn.close_device(dev)
