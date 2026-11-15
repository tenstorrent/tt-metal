"""pool-prefill S1a: multi-row prefill split kernel (sparse_sdpa_pf) on one chip's shard (mesh 1x8, everything replicated, device 0 read): PCC vs the CPU partial attention (o, m, l) + traced us."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import pcc, traced_us
from models.demos.blackhole.hy4_preview.tests.dk_dev import ref_sparse
from models.demos.blackhole.hy4_preview.tt import hy4_pool_prefill as PP

SCALE = 256**-0.5
R = lambda **kw: print("RES " + json.dumps(kw), flush=True)
HT = int(os.environ.get("HT", "64"))
S = int(os.environ.get("S", "512"))
T = int(os.environ.get("T", "4096"))
V, D = 512, 576


def rep(md, t, dt, lay=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def d0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
    try:
        dp = ttnn.experimental.deepseek_prefill
        g = torch.Generator().manual_seed(3)
        kvf = (torch.randn(T, D, generator=g) * 0.7).to(torch.bfloat16)
        kv_lat = rep(md, kvf[:, :V].reshape(1, 1, T, V), ttnn.bfloat16)
        kv_rope = rep(md, kvf[:, V:].reshape(1, 1, T, D - V), ttnn.bfloat16)
        f8, sc = dp.per_token_cast_to_fp8(
            rep(md, kvf[:, :V].reshape(1, 1, T, V), ttnn.bfloat16, ttnn.TILE_LAYOUT), round_scale_to_power_of_two=True
        )
        kv_dev = dp.pack_scaled_fp8_kv_cache(f8, sc, kv_rope)
        rec = ttnn.to_torch(
            ttnn.get_device_tensors(dp.per_token_cast_back(f8, sc, output_dtype=ttnn.bfloat16))[0]
        ).reshape(T, V)
        kv_ref = torch.cat([rec.to(torch.bfloat16), kvf[:, V:]], dim=-1)
        q = (torch.randn(HT, S, D, generator=g) * 0.5).to(torch.bfloat16)
        q_dev = rep(md, q.reshape(1, HT, S, D), ttnn.bfloat16)
        fid = os.environ.get("FID", "hifi4")
        KC = int(os.environ.get("KC", "128"))
        ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity={
                "hifi4": ttnn.MathFidelity.HiFi4,
                "hifi2": ttnn.MathFidelity.HiFi2,
                "lofi": ttnn.MathFidelity.LoFi,
            }[fid],
            math_approx_mode=False,
            fp32_dest_acc_en=os.environ.get("FP32", "1") == "1",
            packer_l1_acc=False,
        )
        g_ = md.compute_with_storage_grid_size()
        R(event="grid", x=g_.x, y=g_.y, fid=fid, kc=KC)
        for pat in os.environ.get("PATS", "mix,256,2048").split(","):
            K = 2048
            idx = torch.full((S, K), -1, dtype=torch.int64)
            nvs = []
            for s in range(S):
                nv = {"mix": [0, 1, 7, 100, 256, 300, 700, 2048][s % 8], "256": 256, "2048": 2048, "64": 64}[pat]
                nv = min(nv, T)
                nvs.append(nv)
                if nv:
                    idx[s, :nv] = torch.randperm(T, generator=g)[:nv]
            idx_dev = rep(md, idx.to(torch.int32).reshape(1, 1, S, K), ttnn.uint32)
            out, st = PP.sparse_sdpa_pf(q_dev, kv_dev, idx_dev, SCALE, KC, ckc)
            ttnn.synchronize_device(md)
            o = d0(out).reshape(HT, S, V).float()
            stf = d0(st).reshape(S, 2 * HT).float()
            ref, M, L = ref_sparse(q, kv_ref, idx, None, with_stats=True)  # [H,S,V], [H,S], [H,S]
            nz = torch.tensor(nvs) > 0
            pc = pcc(o[:, nz], ref[:, nz])
            minrow = min(pcc(o[h, s], ref[h, s]) for h in range(0, HT, 9) for s in range(0, S, 17) if nvs[s] > 0)
            m_dev, l_dev = stf[:, :HT].t() * SCALE, stf[:, HT:].t()
            m_err = (m_dev[:, nz] - M[:, nz]).abs().max().item()
            l_rel = ((l_dev[:, nz] - L[:, nz]).abs() / L[:, nz]).max().item()
            emp = ~nz
            emp_ok = bool(torch.isinf(m_dev[:, emp]).all() and (l_dev[:, emp] == 0).all()) if emp.any() else None
            us = traced_us(md, lambda: PP.sparse_sdpa_pf(q_dev, kv_dev, idx_dev, SCALE, KC, ckc)[0], n=5, rep_=3)
            R(
                test="pf_kernel",
                fid=fid,
                kc=KC,
                pat=pat,
                HT=HT,
                S=S,
                T=T,
                pcc_o=round(pc, 6),
                min_row_pcc=round(minrow, 5),
                m_err=round(m_err, 4),
                l_rel=round(l_rel, 4),
                empty_ok=emp_ok,
                us=round(us, 1),
            )
            ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
