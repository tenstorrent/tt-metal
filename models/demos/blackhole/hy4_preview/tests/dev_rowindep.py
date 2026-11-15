# product3: is row 0 of the TP32 (tp24) routed-expert kernel bit-independent of the content of rows 1..T-1? (verify S=4 lossless question)
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp24 import Tp24Experts
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import route_tensor

HID = 6144
T = int(os.environ.get("TP_T", "4"))
acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
R = torch.cat([r.reshape(-1, 8) for h, y, r in acts["moe"]]).long()
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
try:
    tp = Tp24Experts(md, layer=int(os.environ.get("VL", "1")), chip0=0)
    assert tp.from_cache
    comp = ttnn.ConcatMeshToTensor(md, dim=0)

    def run(x, ids, sc):
        tt_x = ttnn.from_torch(
            x.reshape(1, 1, T, HID),
            device=md,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        o = tp(tt_x, route_tensor(md, ids, sc), T)
        ttnn.synchronize_device(md)
        return ttnn.to_torch(o, mesh_composer=comp).float()[:, 0, :T]  # [8 chips, T, HID]

    res = []
    for trial in range(6):
        g = torch.Generator().manual_seed(7 + trial)
        sel = torch.randperm(X.shape[0], generator=g)[:T]
        x = X[sel].to(torch.bfloat16)
        ids = R[sel]
        sc = (torch.rand(T, 8, generator=g) * 0.4 + 0.05).to(torch.bfloat16)
        a = run(x, ids, sc)
        sel2 = torch.randperm(X.shape[0], generator=g)[:T]
        x2, ids2 = x.clone(), ids.clone()
        sc2 = sc.clone()
        x2[1:] = X[sel2[1:]].to(torch.bfloat16)
        ids2[1:] = R[sel2[1:]]
        sc2[1:] = (torch.rand(T - 1, 8, generator=g) * 0.4 + 0.05).to(torch.bfloat16)
        b = run(x2, ids2, sc2)
        a_rep = run(x, ids, sc)
        d0 = (a[:, 0] - b[:, 0]).abs().max().item()
        rep = (a - a_rep).abs().max().item()
        # also row 1 moved to row 0 position (row permutation)
        perm = [1, 0] + list(range(2, T))
        c = run(x[perm], ids[perm], sc[perm])
        dperm = (a[:, 1] - c[:, 0]).abs().max().item()
        line = {
            "trial": trial,
            "row0_vs_otherrows_maxabs": d0,
            "repeat_maxabs": rep,
            "row1_vs_row0pos_maxabs": dperm,
            "scale": a[:, 0].abs().max().item(),
        }
        p(json.dumps(line))
        res.append(line)
    json.dump(res, open(os.environ.get("OUT", "/tmp/rowindep.json"), "w"))
finally:
    ttnn.close_mesh_device(md)
