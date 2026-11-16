"""B17 stage 2: fused_q (q_nope@w_uk + rope + concat + RM) vs the stock chain (nlp_create_q_heads_split, matmul, rope llama, concat, to_layout): PCC over varying inputs + traced digest."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_fuse_attn as FA
from models.tt_transformers.tt.common import get_rot_transformation_mat

NCALL = int(os.environ.get("NCALL", "6"))
S = int(os.environ.get("S", "1"))


def R(**kw):
    print("RES " + json.dumps(kw), flush=True)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        g_ = torch.Generator().manual_seed(5)
        T = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
            t,
            device=md,
            dtype=dt,
            layout=lay,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sh = lambda t, d: ttnn.from_torch(
            t,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(md, dim=d),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w = sh(torch.randn(1, 64, 192, 512, generator=g_) * 0.05, 1)
        tm = T(get_rot_transformation_mat())
        ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        worst = 1.0
        for it in range(NCALL):
            ang = torch.rand(1, 1, S, 32, generator=g_) * 6.28
            cos = T(torch.cos(ang).repeat_interleave(2, -1))
            sin = T(torch.sin(ang).repeat_interleave(2, -1))
            q = sh(torch.randn(1, 1, S, 64 * 256, generator=g_), 3)
            qn, qp = ttnn.experimental.nlp_create_q_heads_split(
                q, num_heads=8, split_head_dim=192, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            qa = ttnn.matmul(qn, w, compute_kernel_config=ckc)
            qp = ttnn.experimental.rotary_embedding_llama(
                qp, cos, sin, tm, is_decode_mode=False, compute_kernel_config=ckc
            )
            ref = ttnn.to_layout(ttnn.concat([qa, qp], dim=3), ttnn.ROW_MAJOR_LAYOUT)
            got = FA.fused_q(q, w, cos, sin, tm)
            ttnn.synchronize_device(md)
            rt = ttnn.to_torch(ref, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()
            gt = ttnn.to_torch(got, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()
            p = pcc(rt, gt)
            pa = pcc(rt[..., :512], gt[..., :512])
            pr = pcc(rt[..., 512:], gt[..., 512:])
            worst = min(worst, p)
            R(
                event="call",
                it=it,
                shape=list(rt.shape),
                pcc=p,
                pcc_abs=pa,
                pcc_rope=pr,
                maxdiff=float((rt - gt).abs().max()),
                bit_exact=bool(torch.equal(rt, gt)),
            )
        R(event="summary", worst_pcc=worst)
        if os.environ.get("PROF", "0") == "1":

            def stock():
                qn, qp = ttnn.experimental.nlp_create_q_heads_split(
                    q, num_heads=8, split_head_dim=192, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                qa = ttnn.matmul(qn, w, compute_kernel_config=ckc)
                qp = ttnn.experimental.rotary_embedding_llama(
                    qp, cos, sin, tm, is_decode_mode=False, compute_kernel_config=ckc
                )
                return ttnn.to_layout(ttnn.concat([qa, qp], dim=3), ttnn.ROW_MAJOR_LAYOUT)

            for nm, fn in (("stock", stock), ("fused", lambda: FA.fused_q(q, w, cos, sin, tm))):
                for _ in range(2):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(md)
                tid = ttnn.begin_trace_capture(md, cq_id=0)
                for _ in range(3):
                    ttnn.deallocate(fn())
                ttnn.end_trace_capture(md, tid, cq_id=0)
                ttnn.synchronize_device(md)
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(md)
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(md)
                R(event="prof", which=nm)
    finally:
        ttnn.close_mesh_device(md)


main()
