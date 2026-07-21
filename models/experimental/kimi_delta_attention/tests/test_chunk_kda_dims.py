import pytest, torch, ttnn
import torch.nn.functional as F
from models.experimental.kimi_delta_attention.torch_functional import kda_ops as ref
from models.common.utility_functions import comp_pcc
from loguru import logger
@pytest.mark.parametrize("K", [64, 128])
def test_kt(device, K):
    B,T,HV,C=1,64,4,32
    q=ref.l2norm(torch.randn(B,T,HV,K)); k=ref.l2norm(torch.randn(B,T,HV,K))
    v=torch.randn(B,T,HV,K); g=-F.softplus(torch.randn(B,T,HV,K)); beta=torch.sigmoid(torch.randn(B,T,HV))
    o_ref,_=ref.naive_chunk_kda(q,k,v,g,beta,chunk_size=C)
    up=lambda x: ttnn.from_torch(x,dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=device)
    o=ttnn.transformer.chunk_kda(up(q),up(k),up(v),up(g),up(beta),scale=K**-0.5,chunk_size=C)
    if isinstance(o,(tuple,list)): o=o[0]
    ok,pcc=comp_pcc(o_ref,ttnn.to_torch(o),0.98); logger.info(f"[kt] K={K} PCC={pcc}"); assert ok
