import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
# All-ones Q,K,V. QK^T per element = D = 128. After scale 1/sqrt(128)=0.08839 -> 11.3137.
# Causal diagonal block (qi=0): row r has columns 0..r valid (others -inf). row0 -> only col0 = 11.3137.
b,h,s,d=1,1,1024,128
Q=torch.ones(b,h,s,d); K=torch.ones(b,h,s,d); V=torch.ones(b,h,s,d)
dev=ttnn.open_device(device_id=0)
try:
    ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
    tq=ttnn.from_torch(Q,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=dev)
    tk=ttnn.from_torch(K,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=dev)
    tv=ttnn.from_torch(V,dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=dev)
    out=scaled_dot_product_attention(tq,tk,tv,is_causal=True,compute_kernel_config=ckc)
    o=ttnn.to_torch(out).float()
    print("BF8B out[0,0,0,:4]=",o[0,0,0,:4].tolist())
finally:
    ttnn.close_device(dev)
