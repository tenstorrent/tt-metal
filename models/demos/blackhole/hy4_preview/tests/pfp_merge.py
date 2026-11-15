"""pool-prefill: unit test of Hy4AttentionPF._pf_merge_scale (default + HY4_PF_SAFE_MERGE) against a float64 reference on synthetic per-column stats incl. hot logits (to 120) and negative sinks."""
import json
import os
import types

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_pf_attention import Hy4AttentionPF
from models.demos.gpt_oss.tt.ccl import CCLManager

R = lambda **kw: print("RES " + json.dumps(kw), flush=True)
SCALE = 256**-0.5
n = 512


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        gen = torch.Generator().manual_seed(2)
        sink = (torch.randn(64, generator=gen) * 1.5).clamp(-5, 1.5)  # logit domain
        so = (sink / SCALE).reshape(1, 1, 1, 64).bfloat16()
        obj = types.SimpleNamespace(md=md, ccl=ccl, scale=SCALE)
        obj.pf_sink_logit = ttnn.from_torch(
            sink.reshape(1, 1, 1, 64).float(),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        obj.pf_onehot = ttnn.from_torch(
            torch.eye(8).reshape(64, 1, 1, 1).float(),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        obj._ag = lambda t, dim: Hy4AttentionPF._ag(obj, t, dim)

        for hot in (10.0, 40.0, 80.0, 87.0, 120.0):
            # per column stats: scaled max around `hot` for a random winner column per (row, head), others lower; l in [1, 5]
            msc = torch.rand(8, n, 64, generator=gen) * 8 + (hot - 12)
            win = torch.randint(0, 8, (n, 64), generator=gen)
            msc.scatter_(0, win[None], hot)
            msc[torch.rand(8, n, 64, generator=gen) < 0.05] = float("-inf")  # empty columns
            l = torch.rand(8, n, 64, generator=gen) * 4 + 1
            l = torch.where(torch.isinf(msc), torch.zeros_like(l), l)
            raw = (msc / SCALE).bfloat16()
            mscq = raw.float() * SCALE
            lb = l.bfloat16().float()
            # float64 reference
            M_ = torch.maximum(mscq.max(0).values, sink[None, :])
            w = lb.double() * torch.exp(mscq.double() - M_.double())
            den = w.sum(0) + torch.exp(sink.double()[None, :] - M_.double())
            ref = w / den  # [8,n,64]
            stats = torch.zeros(8, 1, n, 128)
            stats[:, 0, :, :64] = raw.float()
            stats[:, 0, :, 64:] = lb
            for safe in ("0", "1"):
                os.environ["HY4_PF_SAFE_MERGE"] = safe
                st = ttnn.from_torch(
                    stats.bfloat16(),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                s_t = Hy4AttentionPF._pf_merge_scale(obj, st)
                ttnn.synchronize_device(md)
                worst = 0.0
                nans = 0
                for c in range(8):
                    got = ttnn.to_torch(ttnn.get_device_tensors(s_t)[c]).reshape(64, n).float().t()  # [n,64]
                    nans += int(torch.isnan(got).sum()) + int(torch.isinf(got).sum())
                    worst = max(worst, float((got - ref[c].float()).abs().max()))
                R(test="merge_scale", hot_logit=hot, safe=safe, max_abs_err_vs_f64=round(worst, 5), nan_inf=nans)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
