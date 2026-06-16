import torch, ttnn, sys
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
DT = ttnn.bfloat8_b if sys.argv[-1]=="bf8b" else ttnn.bfloat16
# Q=K=ones -> QK^T = 128 everywhere pre-scale; post-scale = 128/sqrt(128)=11.3137.
b,h,s,d=1,1,1024,128
Q=torch.ones(b,h,s,d); K=torch.ones(b,h,s,d); V=torch.ones(b,h,s,d)
dev=ttnn.open_device(device_id=0)
try:
    ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
    tq=ttnn.from_torch(Q,dtype=DT,layout=ttnn.TILE_LAYOUT,device=dev)
    tk=ttnn.from_torch(K,dtype=DT,layout=ttnn.TILE_LAYOUT,device=dev)
    tv=ttnn.from_torch(V,dtype=DT,layout=ttnn.TILE_LAYOUT,device=dev)
    out=scaled_dot_product_attention(tq,tk,tv,is_causal=True,compute_kernel_config=ckc)
    o=ttnn.to_torch(out).float()
    print("DT out[0,0,0,:4]=",o[0,0,0,:4].tolist())
finally:
    ttnn.close_device(dev)
