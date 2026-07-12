# debug: compare device-exported m,l to host reference for ONE shard (all indices, single k-chunk)
import sys

import torch

import ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/sdpa")
from sparse_sdpa_test_utils import make_inputs, to_dev  # noqa

K_DIM, V_DIM = 576, 512
H, S, T, TOPK = 32, 32, 32, 32  # single chunk (k_chunk=32), all valid -> m=max, l=sum, no SALAD
scale = K_DIM**-0.5

q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 10**9, seed=0)

# host reference m (raw max), l (sum exp(scale*(qk-m)))
idx = indices.reshape(1, S, TOPK).to(torch.int64)
sel = kv[0, 0][idx.reshape(-1)].reshape(1, S, TOPK, K_DIM)
raw = torch.einsum("bhsd,bsjd->bhsj", q, sel)  # [1,H,S,TOPK] raw q·k
m_host = raw.max(dim=-1).values  # [1,H,S]
p = torch.exp((raw - m_host.unsqueeze(-1)) * scale)
l_host = p.sum(dim=-1)  # [1,H,S]

dev = ttnn.open_device(device_id=0)
tt_q = to_dev(q.to(torch.bfloat16), dev, ttnn.bfloat16)
tt_kv = to_dev(kv.to(torch.bfloat16), dev, ttnn.bfloat16)
tt_idx = to_dev(indices.to(torch.int32), dev, ttnn.uint32)
outs = ttnn.transformer.sparse_sdpa_stats(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=32)
m_dev = ttnn.to_torch(outs[1]).float()  # [1,H,S,32]
l_dev = ttnn.to_torch(outs[2]).float()
ttnn.close_device(dev)

print("m_dev shape", tuple(m_dev.shape), "l_dev shape", tuple(l_dev.shape))
# print col 0..3 of first few heads for head0,query0
print("m_host[0,0,0]     =", m_host[0, 0, 0].item())
print("m_dev [0,0,0,:6]  =", [round(x, 4) for x in m_dev[0, 0, 0, :6].tolist()])
print("l_host[0,0,0]     =", l_host[0, 0, 0].item())
print("l_dev [0,0,0,:6]  =", [round(x, 4) for x in l_dev[0, 0, 0, :6].tolist()])
print()
print("m_host[0,1,0]     =", m_host[0, 1, 0].item(), " m_dev[0,1,0,0]=", m_dev[0, 1, 0, 0].item())
print("m_host[0,0,1]     =", m_host[0, 0, 1].item(), " m_dev[0,0,1,0]=", m_dev[0, 0, 1, 0].item())
# where does the max value actually sit in m_dev? scan cols
mx_col = m_dev[0, 0, 0].argmax().item()
print("argmax col of m_dev[0,0,0] =", mx_col, "value", m_dev[0, 0, 0].max().item())
# overall correlation of col0
mc = torch.corrcoef(torch.stack([m_dev[..., 0].flatten(), m_host.flatten()]))[0, 1].item()
lc = torch.corrcoef(torch.stack([l_dev[..., 0].flatten(), l_host.flatten()]))[0, 1].item()
print(f"corr(m_dev[...,0], m_host)={mc:.5f}   corr(l_dev[...,0], l_host)={lc:.5f}")
l_sum = l_dev.sum(dim=-1)  # hypothesis: l spread across 32 cols
print(f"l_dev.sum(-1)[0,0,0]={l_sum[0,0,0].item():.4f}  vs l_host[0,0,0]={l_host[0,0,0].item():.4f}")
lcs = torch.corrcoef(torch.stack([l_sum.flatten(), l_host.flatten()]))[0, 1].item()
print(f"corr(l_dev.sum(-1), l_host)={lcs:.5f}")
mmx = m_dev.max(dim=-1).values
mmc = torch.corrcoef(torch.stack([mmx.flatten(), m_host.flatten()]))[0, 1].item()
print(f"corr(m_dev.max(-1), m_host)={mmc:.5f}")
