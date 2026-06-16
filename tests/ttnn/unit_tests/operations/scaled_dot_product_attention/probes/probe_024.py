import torch, ttnn, sys
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
DT = ttnn.bfloat8_b if "bf8b" in sys.argv else ttnn.bfloat16
def fa_rand(*shape):
    n1=torch.randn(shape); n2=torch.randn(shape)*10
    b=torch.bernoulli(torch.full(shape,0.001)); return n1+n2*b
torch.manual_seed(1234)
b,h,s,d=1,1,1024,128
Q=fa_rand(b,h,s,d); K=fa_rand(b,h,s,d); V=fa_rand(b,h,s,d)
scale=1.0/(d**0.5)
# Expected QK post-scale row0 cols0-7 (causal qi=0 diagonal -> col0 valid, others -inf-masked AFTER this point)
qk = (Q[0,0,0,:] @ K[0,0,:8,:].T) * scale
print("EXPECT QK*scale row0 col0-7:", [round(x,4) for x in qk.tolist()])
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
