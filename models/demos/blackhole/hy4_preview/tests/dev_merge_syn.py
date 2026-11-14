"""fuse-sparse-merge: synthetic numerics of the fused merge kernel stages (HY4_MRG_DBG 0 full, 6 numerator only, 7 rden, 8 w_j) vs torch"""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
from dk_dev import pcc
from dm_block import close_row, open_row
import ttnn
import hy4_merge

full, md = open_row()
try:
    S = 1
    torch.manual_seed(0)
    x = torch.zeros(8 * S, 32, 576)
    x[:, :8, :512] = torch.randn(8 * S, 8, 512)
    ms = torch.randn(8 * S, 8) * 2
    l = torch.rand(8 * S, 8) * 3 + 0.5
    x[:, :8, 512] = ms
    x[:, :8, 544] = l
    sk = torch.zeros(S, 32, 1)
    sk[:, :8, 0] = torch.randn(S, 8)
    xb, skb = x.to(torch.bfloat16), sk.to(torch.bfloat16)
    xd = ttnn.from_torch(xb.reshape(1, 8 * S, 32, 576), device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(md), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sd = ttnn.from_torch(skb.reshape(1, S, 32, 1), device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(md), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    xf, skf = xb.float(), skb.float()
    msf, lf = xf[:, :8, 512], xf[:, :8, 544]
    w = lf * torch.exp(msf - skf[0, :8, 0][None, :])  # [8src, 8 heads]
    den = w.sum(0) + 1
    num = (xf[:, :8, :512] * w[:, :, None]).sum(0)
    dbg = os.environ.get("HY4_MRG_DBG", "0")
    out = ttnn.to_torch(hy4_merge.merge_fused(xd, sd, S), mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()
    out = out.reshape(8, S, 32, 512)[0, 0]  # device 0
    print("RES dbg", dbg, "finite", bool(torch.isfinite(out).all()), "nan_rows", [i for i in range(32) if not torch.isfinite(out[i]).all()][:8], flush=True)
    if dbg == "0":
        ref = num / den[:, None]
        print("RES pcc", pcc(out[:8], ref), flush=True)
    elif dbg == "6":
        print("RES pcc_num", pcc(out[:8], num), flush=True)
    elif dbg == "7":
        print("RES rden out[0,:4]", out[:8, :3].tolist(), "ref", (1 / den)[:8].tolist(), flush=True)
    elif dbg == "8":
        for j in range(8):
            print("RES w", j, out[:8, 0].tolist(), w[j].tolist(), flush=True) if False else None
        print("RES w_ok", [pcc(out[:8, 0:1], w[0][:, None])], flush=True)
finally:
    close_row(full, md)
