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
# q-tile-row 1 = global query rows 32..63. qi==1 -> KV blocks j=0 (rows0..31) and j=1 (diag rows32..63).
# Debug prints query-row index 1 WITHIN tile-row 1 = global query row 33.
qrow=33
qk0=(Q[0,0,qrow,:]@K[0,0,0:8,:].T)*scale          # j=0 cols 0..7
qk1=(Q[0,0,qrow,:]@K[0,0,32:40,:].T)*scale        # j=1 cols 0..7 (global key 32..39)
print("EXPECT qi1 j=0 QK*scale row(g33):", [round(x,3) for x in qk0.tolist()])
print("EXPECT qi1 j=1 QK*scale row(g33):", [round(x,3) for x in qk1.tolist()])
ref=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float(),is_causal=True)
print("EXPECT out global-row33 [:4]:", [round(x,4) for x in ref[0,0,33,:4].tolist()])
dev=ttnn.open_device(device_id=0)
try:
    ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
    tq=ttnn.from_torch(Q,dtype=DT,layout=ttnn.TILE_LAYOUT,device=dev)
    tk=ttnn.from_torch(K,dtype=DT,layout=ttnn.TILE_LAYOUT,device=dev)
    tv=ttnn.from_torch(V,dtype=DT,layout=ttnn.TILE_LAYOUT,device=dev)
    out=scaled_dot_product_attention(tq,tk,tv,is_causal=True,compute_kernel_config=ckc)
    o=ttnn.to_torch(out).float()
    print("DT out global-row33 [:4]=",[round(x,4) for x in o[0,0,33,:4].tolist()])
finally:
    ttnn.close_device(dev)
