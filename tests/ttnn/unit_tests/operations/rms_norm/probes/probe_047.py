import torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm
device = ttnn.open_device(device_id=0)
shape=(1,1,8192,1024)
torch.manual_seed(42)
tx=torch.randn(shape,dtype=torch.float32).to(torch.bfloat16)
tg=torch.randn((1,1,1,shape[-1]),dtype=torch.float32).to(torch.bfloat16)
mc=shard_config([1024,128],(8,8),ttnn.TensorMemoryLayout.BLOCK_SHARDED,layout=ttnn.TILE_LAYOUT,dtype=ttnn.bfloat16,device=device)
x=ttnn.from_torch(tx,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device,memory_config=mc)
g=ttnn.from_torch(tg,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device,memory_config=ttnn.DRAM_MEMORY_CONFIG)
cfg=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2,fp32_dest_acc_en=False,math_approx_mode=False)
out=ttnn.to_torch(rms_norm(x,gamma=g,compute_kernel_config=cfg,memory_config=x.memory_config())).to(torch.float32)
xf=tx.to(torch.float32)
exp=xf*torch.rsqrt(xf.pow(2).mean(dim=-1,keepdim=True)+1e-6)*tg.to(torch.float32).reshape(-1)
a,b=out.flatten(),exp.flatten()
print("PCC", torch.corrcoef(torch.stack([a,b]))[0,1].item())
print("out[0,0,0,:4]", out[0,0,0,:4], "exp", exp[0,0,0,:4])
ttnn.close_device(device)
