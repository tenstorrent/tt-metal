import torch, ttnn, math
import ttnn.operations.scaled_dot_product_attention as sdpa_mod
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
sdpa_mod.SUPPORTED["alignment"] = ["tile_aligned", "w_non_aligned", "h_non_aligned"]

def pcc(a, b):
    a=a.float().flatten(); b=b.float().flatten()
    a=a-a.mean(); b=b-b.mean()
    d=(a.norm()*b.norm())
    return 1.0 if d==0 else float((a*b).sum()/d)
def rrms(a,b):
    a=a.float().flatten(); b=b.float().flatten()
    return float(((a-b).norm())/(b.std()+1e-12))
def ref(Q,K,V,scale=None):
    Qf,Kf,Vf=Q.float(),K.float(),V.float(); D=Qf.shape[-1]
    s=scale if scale is not None else 1.0/math.sqrt(D)
    sc=(Qf@Kf.transpose(-2,-1))*s
    return (torch.softmax(sc,dim=-1)@Vf)
def run(name,dt,qshape,kshape,mask=False,scale=None):
    tdt = torch.bfloat16 if dt in (ttnn.bfloat16, ttnn.bfloat8_b) else torch.float32
    torch.manual_seed(0)
    Q=torch.randn(qshape,dtype=tdt); K=torch.randn(kshape,dtype=tdt); V=torch.randn(kshape,dtype=tdt)
    m=None
    if mask:
        B,_,Sq,_=qshape; Skv=kshape[-2]
        m=torch.zeros(B,1,Sq,Skv,dtype=tdt)
        m.masked_fill_(torch.triu(torch.ones(Sq,Skv,dtype=torch.bool),diagonal=1),float("-inf"))
    exp=ref(Q,K,V,scale)
    if m is not None: 
        sc=(Q.float()@K.float().transpose(-2,-1))*(scale if scale else 1.0/math.sqrt(qshape[-1]))+m.float()
        exp=(torch.softmax(sc,dim=-1)@V.float())
    qt=ttnn.from_torch(Q,dtype=dt,layout=ttnn.TILE_LAYOUT,device=device)
    kt=ttnn.from_torch(K,dtype=dt,layout=ttnn.TILE_LAYOUT,device=device)
    vt=ttnn.from_torch(V,dtype=dt,layout=ttnn.TILE_LAYOUT,device=device)
    mt=ttnn.from_torch(m,dtype=dt,layout=ttnn.TILE_LAYOUT,device=device) if m is not None else None
    try:
        out=scaled_dot_product_attention(qt,kt,vt,attention_mask=mt,scale=scale)
        res=ttnn.to_torch(out)
        print(f"{name}: PCC={pcc(res,exp):.5f} rrms={rrms(res,exp):.4f} max_abs={float((res.float()-exp.float()).abs().max()):.4f}")
    except Exception as e:
        print(f"{name}: EXCEPTION {type(e).__name__}: {e}")

device=ttnn.open_device(device_id=0)
try:
    for dt,tag in [(ttnn.float32,"fp32"),(ttnn.bfloat8_b,"bf8b")]:
        run(f"{tag}_w_D50",dt,(1,1,32,50),(1,1,32,50))
        run(f"{tag}_w_D47mh",dt,(1,8,64,47),(1,8,64,47))
        run(f"{tag}_h_S47",dt,(1,1,47,64),(1,1,47,64))
        run(f"{tag}_h_S100",dt,(2,4,100,64),(2,4,100,64))
        run(f"{tag}_both_50",dt,(1,1,50,50),(1,1,50,50))
        run(f"{tag}_h_S47_causal",dt,(1,1,47,64),(1,1,47,64),mask=True)
        run(f"{tag}_cross_100_47_50",dt,(1,4,100,50),(1,4,47,50))
finally:
    ttnn.close_device(device)
