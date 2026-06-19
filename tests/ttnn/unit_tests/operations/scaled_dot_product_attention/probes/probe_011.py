import torch, ttnn, math
import ttnn.operations.scaled_dot_product_attention as sdpa_mod
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
sdpa_mod.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]
def pcc(a,b):
    a=a.float().flatten();b=b.float().flatten();a=a-a.mean();b=b-b.mean()
    d=a.norm()*b.norm(); return 1.0 if d==0 else float((a*b).sum()/d)
def ref(Q,K,V,scale=None):
    Qf,Kf,Vf=Q.float(),K.float(),V.float();D=Qf.shape[-1]
    s=scale if scale else 1.0/math.sqrt(D)
    return (torch.softmax((Qf@Kf.transpose(-2,-1))*s,dim=-1)@Vf)
device=ttnn.open_device(device_id=0)
try:
    # bf16 input, HiFi2 + fp32_dest_acc_en=False => acc_dtype=bf16 => bf16 pad-mask fill path
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False)
    for name,qs,ks in [("h_S47",(1,1,47,64),(1,1,47,64)),("h_S100_mb",(2,4,100,64),(2,4,100,64)),("both_50",(1,1,50,50),(1,1,50,50))]:
        torch.manual_seed(0)
        Q=torch.randn(qs,dtype=torch.bfloat16);K=torch.randn(ks,dtype=torch.bfloat16);V=torch.randn(ks,dtype=torch.bfloat16)
        exp=ref(Q,K,V)
        qt=ttnn.from_torch(Q,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device)
        kt=ttnn.from_torch(K,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device)
        vt=ttnn.from_torch(V,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device)
        out=scaled_dot_product_attention(qt,kt,vt,compute_kernel_config=cfg)
        res=ttnn.to_torch(out)
        print(f"bf16mask_{name}: PCC={pcc(res,exp):.5f} max_abs={float((res.float()-exp.float()).abs().max()):.4f}")
finally:
    ttnn.close_device(device)
