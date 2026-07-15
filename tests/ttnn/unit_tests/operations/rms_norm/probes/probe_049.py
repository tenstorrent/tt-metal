import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config
shape = (1, 1, 32, 8192)
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
torch.manual_seed(0)
x = torch.randn(shape, dtype=torch.bfloat16)
g = torch.full((shape[-1],), 3.0, dtype=torch.bfloat16)
inv = (1.0/torch.sqrt(torch.mean(x.float()**2,dim=-1,keepdim=True)+1e-6))[0,0,0].item()
# core (0,0) owns w_tile_start=0 => cols 0..1023. row 0 (tile-row 0, sub-row 0). first 6 cols:
print("CASE randn x, gamma=3. torch 1/rms row0 =", round(inv,5))
print("xnorm expected row0 col0-5 (x/rms):", [round(x[0,0,0,c].item()*inv,4) for c in range(6)])
print("xnorm*gamma expected (=out):", [round(x[0,0,0,c].item()*inv*3.0,4) for c in range(6)])
dev = ttnn.open_device(device_id=0)
try:
    mc = auto_shard_config(list(shape), B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    gt = ttnn.from_torch(g.reshape(1,1,1,shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ttnn.ComputeConfigDescriptor(); cfg.math_fidelity = ttnn.MathFidelity.HiFi4; cfg.fp32_dest_acc_en = False; cfg.math_approx_mode = False
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg, memory_config=xt.memory_config())
    res = ttnn.to_torch(out).float().reshape(32,8192)
    print("readback out row0 col0-5:", [round(v,4) for v in res[0,:6].tolist()])
finally:
    ttnn.close_device(dev)
