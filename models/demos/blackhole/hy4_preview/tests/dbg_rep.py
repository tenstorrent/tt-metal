"""same inputs, N calls: is the wrong result deterministic?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from dk_dev import *

import ttnn

mod = load_split()
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
D, S, G, kc = [int(x) for x in os.environ.get("CASE", "576:12:1:256").split(":")]
T = 65536
H, Hr, topk = 32, 8, 2048
try:
    torch.manual_seed(0)
    kv = torch.randn(T, D).to(torch.bfloat16)
    per = T // S
    idx = make_idx(S, topk, [(s * per, (s + 1) * per) for s in range(S)], int(os.environ.get("VALID", topk)))
    q = torch.randn(Hr, S, D).to(torch.bfloat16)
    sl = torch.linspace(-2, 2, Hr)
    qp = torch.zeros(H, S, D, dtype=torch.bfloat16)
    qp[:Hr] = q
    sp = torch.zeros(H)
    sp[:Hr] = sl
    _q, _kv, _ix = (
        to_dev(md, qp.reshape(1, H, S, D), ttnn.bfloat16),
        to_dev(md, kv.reshape(1, 1, T, D), ttnn.bfloat16),
        idx_dev(md, idx),
    )
    sk = to_dev(md, (sp / SCALE).reshape(1, 1, 1, H).to(torch.bfloat16), ttnn.bfloat16)
    ref = ref_sparse(q, kv, idx, sl)
    outs = []
    for i in range(int(os.environ.get("N", 5))):
        o = ttnn.to_torch(mod.sparse_sdpa_split(_q, _kv, _ix, 512, SCALE, kc, G, ckc_of(md), sk))[0].float()
        outs.append(o)
        emit({"call": i, "pcc_per_row": [round(pcc(o[:Hr, s], ref[:, s]), 3) for s in range(S)]})
    emit({"identical_to_first": [bool((o == outs[0]).all()) for o in outs]})
finally:
    ttnn.close_mesh_device(md)
