# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""R0 microbench for the GLM-5.3 prefill "glue" track (stems, small CCLs, MoE prelude/epilogue).

One mesh open, many independent sections. Every section builds production-shaped per-chip tensors on the
8x4 torus-XY mesh (640 local tokens, TP 4), times candidate op sequences by trace replay (wall time per
sequence, ops serial, so it includes the ~1.3 us inter-op gap exactly like the model's replay) and checks
PCC of each candidate against the production sequence and against a torch reference.

Select sections with GLUE_R0_SECTIONS=a,b,c (default: all). Results are printed as lines starting with
"GLUE_R0 " so they can be grepped out of the log.
"""

import copy
import os
import time
import traceback

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

SP, TP = 8, 4
S = 640  # local tokens per chip
H = 6144
HL = H // TP
Q_LORA, KV_LORA, ROPE, NOPE, VDIM = 2048, 512, 64, 192, 256
HEADS_L = 64 // TP
EPS = 1e-6
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

SECTIONS = os.environ.get("GLUE_R0_SECTIONS", "all")
N_INNER = int(os.environ.get("GLUE_R0_INNER", "8"))
N_REPLAY = int(os.environ.get("GLUE_R0_REPLAY", "20"))


def want(name):
    return SECTIONS == "all" or name in SECTIONS.split(",")


def log(*a):
    print("GLUE_R0", *a, flush=True)


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    if torch.equal(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float("nan")


def dev_torch(t, idx):
    """Per-device torch tensor of mesh device idx (row-major r*TP+c)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[idx]).float()


def cmp(name, t_new, t_ref, devs=(0, 5, 10, 31)):
    vals = []
    for d in devs:
        a = dev_torch(t_new, d) if isinstance(t_new, ttnn.Tensor) else t_new[d]
        b = dev_torch(t_ref, d) if isinstance(t_ref, ttnn.Tensor) else t_ref[d]
        a = a.reshape(b.shape) if a.numel() == b.numel() else a
        vals.append(pcc(a, b))
        md = (a - b).abs().max().item()
    log(f"PCC {name}: min={min(vals):.6f} vals={[round(v, 6) for v in vals]} maxabs(last)={md:.4g}")
    return min(vals)


class Timer:
    """Trace-replay timing of a callable that issues device ops and returns tensors to free."""

    def __init__(self, mesh):
        self.mesh = mesh

    def __call__(self, name, fn, n_inner=N_INNER, n_replay=N_REPLAY):
        mesh = self.mesh
        try:
            outs = fn()
            ttnn.synchronize_device(mesh)
            _free(outs)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(n_inner):
                _free(fn())
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(n_replay):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ts.append((time.perf_counter() - t0) / (n_replay * n_inner) * 1e6)
            ttnn.release_trace(mesh, tid)
            us = sorted(ts)[1]
            log(f"TIME {name}: {us:.2f} us  (reps {[round(t, 2) for t in ts]})")
            return us
        except Exception as e:  # keep going: one bad candidate must not kill the job
            log(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}")
            traceback.print_exc()
            try:
                ttnn.synchronize_device(mesh)
            except Exception:
                pass
            return None


def _free(outs):
    if outs is None:
        return
    if isinstance(outs, ttnn.Tensor):
        outs = [outs]
    for t in outs:
        if t is not None and isinstance(t, ttnn.Tensor) and t.is_allocated():
            ttnn.deallocate(t)


def mk(mesh, torch_t, dtype=ttnn.bfloat16, mem=DRAM, dims=(None, None), layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch_t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
    )


def rmsnorm_ref(x, g):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g.float()


def glm_cfg(key):
    for e in MLA_MATMUL_CONFIG[key][640]:
        if e.get("num_heads") == 64 and e.get("q_lora_rank") == 2048:
            return e
    raise KeyError(key)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=1536,
                trace_region_size=256 * 1024 * 1024,
            ),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole Galaxy only")
@pytest.mark.timeout(0)
def test_glue_r0(mesh_device, variant, config_only):
    mesh = mesh_device
    assert tuple(mesh.shape) == (SP, TP)
    torch.manual_seed(0)
    tt_ccl = get_tt_ccl(mesh)
    timer = Timer(mesh)

    def ck(fid, fp32, packer=True):
        return ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=packer
        )

    hifi2 = ck(ttnn.MathFidelity.HiFi2, False)
    hifi4f = ck(ttnn.MathFidelity.HiFi4, True)
    topo = ttnn.Topology.Ring

    def rs(x, dim=3, mem=DRAM, workers=None, cks=None, chunks=None, bufs=None):
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
            num_links=2,
            memory_config=mem,
            topology=topo,
            cluster_axis=1,
            num_workers_per_link=workers,
            chunks_per_sync=chunks,
            num_buffers_per_channel=bufs,
            compute_kernel_config=cks,
        )

    def hbag(x, out, dim=3):
        return ttnn.experimental.high_bw_all_gather(x, dim=dim, output_tensor=out, cluster_axis=1, num_links=2)

    def ag_async(x, dim=-1, workers=None, chunks=None, bufs=None, mem=None):
        return ttnn.experimental.all_gather_async(
            x,
            dim=dim,
            cluster_axis=1,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
            num_links=2,
            topology=topo,
            memory_config=mem,
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=bufs,
        )

    # ---------------- shared inputs ----------------
    x_full = torch.randn(1, 1, S * SP, H)
    x_l1 = mk(mesh, x_full, mem=L1, dims=(2, 3))  # attn_norm output, TP-sharded hidden, L1
    x_dram = mk(mesh, x_full, mem=DRAM, dims=(2, 3))

    def rows(t, r):  # SP row-block r of a global [.., S*SP, ..] tensor
        return t[..., r * S : (r + 1) * S, :]

    # ======================= A. empty / tiny-op floor =======================
    if want("a"):
        tiny = mk(mesh, torch.randn(1, 1, 32, 32))
        timer("A0 tiny add (per-op floor)", lambda: ttnn.add(tiny, tiny))

    # ======================= B. MLA q_a / kv_a stem =======================
    if want("b"):
        w_qa = torch.randn(H, Q_LORA) * (1 / H**0.5)
        w_kv = torch.randn(H, KV_LORA + ROPE) * (1 / H**0.5)
        g_q = torch.rand(Q_LORA) + 0.5
        g_kv = torch.rand(KV_LORA) + 0.5
        tw_qa = mk(mesh, w_qa, ttnn.bfloat8_b, dims=(None, 0))
        tw_kv = mk(mesh, w_kv, ttnn.bfloat8_b, dims=(None, 0))
        tg_q = mk(mesh, g_q.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        tg_kv = mk(mesh, g_kv.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        qa_cfg, kv_cfg = glm_cfg("q_a_proj"), glm_cfg("kv_a_proj_with_mqa")
        qb_act = glm_cfg("q_b_proj")["act_mem_config"]
        buf_q = tt_ccl.get_mla_high_bw_all_gather_buffer(
            name="q_a_latent", shape=[1, 1, S, Q_LORA], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        buf_kv = tt_ccl.get_mla_high_bw_all_gather_buffer(
            name="kv_stem", shape=[1, TP, S, KV_LORA + ROPE], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        W = Q_LORA + KV_LORA + ROPE
        PAD = -W % (TP * 32)
        WM = W + PAD
        buf_m = tt_ccl.get_mla_high_bw_all_gather_buffer(
            name="qkv_a", shape=[1, 1, S, WM], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        # merged weight built on device exactly like ttMLA._concat_qkv_a_weight
        parts = [ttnn.typecast(tw_qa, ttnn.bfloat16), ttnn.typecast(tw_kv, ttnn.bfloat16)]
        parts.append(mk(mesh, torch.zeros(HL, PAD)))
        merged = ttnn.concat(parts, dim=-1, memory_config=DRAM)
        tw_m = ttnn.typecast(merged, ttnn.bfloat8_b, memory_config=DRAM)
        for p in parts + [merged]:
            ttnn.deallocate(p)

        def qa_mm():
            return ttnn.linear(
                x_l1,
                tw_qa,
                program_config=qa_cfg["program_config"],
                memory_config=qa_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )

        def kv_mm():
            return ttnn.linear(
                x_l1,
                tw_kv,
                program_config=kv_cfg["program_config"],
                memory_config=DRAM,
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )

        def base_stem():
            q = qa_mm()
            q2 = rs(q)
            ttnn.deallocate(q)
            q3 = hbag(q2, buf_q)
            ttnn.deallocate(q2)
            qn = ttnn.rms_norm(q3, weight=tg_q, epsilon=EPS, memory_config=qb_act, compute_kernel_config=hifi2)
            kv = kv_mm()
            kvg = hbag(kv, buf_kv, dim=1)
            ttnn.deallocate(kv)
            kn, kr = ttnn.experimental.fast_reduce_nc_split(
                kvg, dim=1, split_output_width=KV_LORA, memory_config=DRAM, compute_kernel_config=hifi4f
            )
            kn2 = ttnn.rms_norm(kn, weight=tg_kv, epsilon=EPS, memory_config=DRAM, compute_kernel_config=hifi2)
            ttnn.deallocate(kn)
            return [qn, kn2, kr]

        def merged_mm(pc=None, mem=DRAM, cks=hifi2):
            return ttnn.linear(x_l1, tw_m, program_config=pc, memory_config=mem, dtype=ttnn.bfloat16, compute_kernel_config=cks)

        def merged_stem(pc=None, mm_mem=DRAM, rs_cks=None, rs_workers=None):
            y = merged_mm(pc, mm_mem)
            y2 = rs(y, cks=rs_cks, workers=rs_workers)
            ttnn.deallocate(y)
            y3 = hbag(y2, buf_m)
            ttnn.deallocate(y2)
            q = ttnn.slice(y3, [0, 0, 0, 0], [1, 1, S, Q_LORA])
            kn = ttnn.slice(y3, [0, 0, 0, Q_LORA], [1, 1, S, Q_LORA + KV_LORA])
            kr = ttnn.slice(y3, [0, 0, 0, Q_LORA + KV_LORA], [1, 1, S, W])
            qn = ttnn.rms_norm(q, weight=tg_q, epsilon=EPS, memory_config=qb_act, compute_kernel_config=hifi2)
            ttnn.deallocate(q)
            kn2 = ttnn.rms_norm(kn, weight=tg_kv, epsilon=EPS, memory_config=DRAM, compute_kernel_config=hifi2)
            ttnn.deallocate(kn)
            return [qn, kn2, kr]

        pc_m8 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=2,
            per_core_N=8,
            transpose_mcast=False,
            fuse_batch=False,
            fused_activation=None,
        )
        pc_m4 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=2,
            per_core_N=8,
            transpose_mcast=False,
            fuse_batch=False,
            fused_activation=None,
        )

        # correctness: base vs torch, merged vs base
        qn_b, kn_b, kr_b = base_stem()
        ref_q, ref_kn, ref_kr = {}, {}, {}
        for d in (0, 5, 10, 31):
            r = d // TP
            xr = rows(x_full, r)[0, 0]
            q = xr @ w_qa
            kv = xr @ w_kv
            ref_q[d] = rmsnorm_ref(q, g_q)
            ref_kn[d] = rmsnorm_ref(kv[:, :KV_LORA], g_kv)
            ref_kr[d] = kv[:, KV_LORA:]
        cmp("B base qn vs torch", qn_b, ref_q)
        cmp("B base kv_nope vs torch", kn_b, ref_kn)
        cmp("B base kv_rope vs torch", kr_b, ref_kr)
        for tag, kw in (
            ("auto", {}),
            ("pc8", {"pc": pc_m8}),
            ("pc8-L1", {"pc": pc_m8, "mm_mem": L1}),
            ("pc4-L1", {"pc": pc_m4, "mm_mem": L1}),
            ("pc8-L1-rsfp32", {"pc": pc_m8, "mm_mem": L1, "rs_cks": hifi4f}),
        ):
            try:
                qn_m, kn_m, kr_m = merged_stem(**kw)
                cmp(f"B merged[{tag}] qn vs base", qn_m, qn_b)
                cmp(f"B merged[{tag}] kv_nope vs base", kn_m, kn_b)
                cmp(f"B merged[{tag}] kv_rope vs base", kr_m, kr_b)
                cmp(f"B merged[{tag}] kv_rope vs torch", kr_m, ref_kr)
                _free([qn_m, kn_m, kr_m])
            except Exception as e:
                log(f"FAIL B merged[{tag}] pcc: {e}")
                traceback.print_exc()
        _free([qn_b, kn_b, kr_b])

        timer("B0 base stem (q_a mm,RS,AG,norm + kv_a mm,AG,reduce,norm)", base_stem)
        for tag, kw in (
            ("auto", {}),
            ("pc8", {"pc": pc_m8}),
            ("pc8-L1", {"pc": pc_m8, "mm_mem": L1}),
            ("pc4-L1", {"pc": pc_m4, "mm_mem": L1}),
            ("pc8-L1-rsfp32", {"pc": pc_m8, "mm_mem": L1, "rs_cks": hifi4f}),
            ("pc8-L1-rs4w", {"pc": pc_m8, "mm_mem": L1, "rs_workers": 4}),
        ):
            timer(f"B1 merged stem [{tag}]", lambda kw=kw: merged_stem(**kw))
        # pieces
        timer("B2 q_a mm", qa_mm)
        timer("B2 kv_a mm", kv_mm)
        timer("B2 merged mm pc8 L1", lambda: merged_mm(pc_m8, L1))
        timer("B2 merged mm auto DRAM", lambda: merged_mm())
        qa_out = qa_mm()
        m_out = merged_mm(pc_m8, L1)
        for w in (None, 4, 8):
            timer(f"B3 RS q_a [640,2048] L1 workers={w}", lambda w=w: rs(qa_out, workers=w))
            timer(f"B3 RS merged [640,{WM}] L1 workers={w}", lambda w=w: rs(m_out, workers=w))
        q_rs = rs(qa_out)
        m_rs = rs(m_out)
        timer("B4 HBAG q_a [640,512]->2048", lambda: [hbag(q_rs, buf_q)] and None)
        timer(f"B4 HBAG merged [640,{WM // 4}]->{WM}", lambda: [hbag(m_rs, buf_m)] and None)
        kv_out = kv_mm()
        timer("B4 HBAG kv dim1 [640,576]x4", lambda: [hbag(kv_out, buf_kv, dim=1)] and None)
        kvg = hbag(kv_out, buf_kv, dim=1)
        timer(
            "B5 fast_reduce_nc_split",
            lambda: list(
                ttnn.experimental.fast_reduce_nc_split(
                    kvg, dim=1, split_output_width=KV_LORA, memory_config=DRAM, compute_kernel_config=hifi4f
                )
            ),
        )
        mg = hbag(m_rs, buf_m)
        timer("B6 slice q [640,2048] from merged", lambda: ttnn.slice(mg, [0, 0, 0, 0], [1, 1, S, Q_LORA]))
        timer("B6 slice kv_nope from merged", lambda: ttnn.slice(mg, [0, 0, 0, Q_LORA], [1, 1, S, Q_LORA + KV_LORA]))
        timer("B6 slice kv_rope from merged", lambda: ttnn.slice(mg, [0, 0, 0, Q_LORA + KV_LORA], [1, 1, S, W]))
        qg = hbag(q_rs, buf_q)
        timer(
            "B7 q_a rms_norm [640,2048] DRAM->L1",
            lambda: ttnn.rms_norm(qg, weight=tg_q, epsilon=EPS, memory_config=qb_act, compute_kernel_config=hifi2),
        )
        # q_a norm alternative: distributed fused norm on the RS output, then gather the normed latent
        g_q_fused = ttnn.to_layout(ttnn.reshape(mk(mesh, g_q.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT, dims=(None, 2)), (1, 1, 1, Q_LORA // TP)), ttnn.TILE_LAYOUT)

        def dist_qnorm(src):
            sems, stats = tt_ccl.get_fused_rmsnorm_resources(src, g_q_fused, 1, 2)
            return ttnn.experimental.dit_fused_distributed_rmsnorm(
                src,
                1,
                mesh,
                sems,
                topology=topo,
                epsilon=EPS,
                weight=g_q_fused,
                persistent_output_buffer=stats,
                num_preferred_links=2,
                dtype=ttnn.bfloat16,
                memory_config=DRAM,
            )

        try:
            dn = dist_qnorm(q_rs)
            dng = hbag(dn, buf_q)
            qn_ref = ttnn.rms_norm(qg, weight=tg_q, epsilon=EPS, memory_config=qb_act, compute_kernel_config=hifi2)
            cmp("B8 RS->distnorm->AG qn vs base", dng, qn_ref)
            _free([dn, qn_ref])
        except Exception as e:
            log(f"FAIL B8 pcc: {e}")
            traceback.print_exc()
        timer("B8 distributed fused q norm on RS out [640,512]", lambda: dist_qnorm(q_rs))

        def q_path_dist():
            q = qa_mm()
            q2 = rs(q)
            ttnn.deallocate(q)
            q3 = dist_qnorm(q2)
            ttnn.deallocate(q2)
            q4 = hbag(q3, buf_q)
            ttnn.deallocate(q3)
            return None

        def q_path_base():
            q = qa_mm()
            q2 = rs(q)
            ttnn.deallocate(q)
            q3 = hbag(q2, buf_q)
            ttnn.deallocate(q2)
            return ttnn.rms_norm(q3, weight=tg_q, epsilon=EPS, memory_config=qb_act, compute_kernel_config=hifi2)

        timer("B9 q path base (mm,RS,AG,norm)", q_path_base)
        timer("B9 q path distnorm (mm,RS,distnorm,AG)", q_path_dist)
        _free([qa_out, m_out, q_rs, m_rs, kv_out])

    # ======================= C. absorbed Q stem =======================
    if want("c"):
        from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup

        hf = copy.deepcopy(config_only)
        hf.max_seq_len = S * SP
        rope_t = RotarySetup(hf, mesh, sp_axis=0).get_rope_tensors_indexed(S * SP, S * SP)
        qb_cfg, b1_cfg = glm_cfg("q_b_proj"), glm_cfg("wkv_b1")
        qn_in = mk(mesh, torch.randn(1, 1, S * SP, Q_LORA), mem=qb_cfg["act_mem_config"], dims=(2, None))
        w_qb = torch.randn(Q_LORA, 64 * (NOPE + ROPE)) * (1 / Q_LORA**0.5)
        tw_qb = mk(mesh, w_qb, ttnn.bfloat8_b, dims=(None, 1))
        w_b1 = torch.randn(1, 64, NOPE, KV_LORA) * (1 / NOPE**0.5)
        tw_b1 = mk(mesh, w_b1, ttnn.bfloat8_b, dims=(None, 1))
        # augmented absorb weight [[W, 0], [0, I]] : [16, 256, 576], built on device from the bfp8 W
        wb1_bf16 = ttnn.typecast(tw_b1, ttnn.bfloat16)
        zr = mk(mesh, torch.zeros(1, 64, NOPE, ROPE), dims=(None, 1))
        top = ttnn.concat([wb1_bf16, zr], dim=-1)
        bot_t = torch.cat([torch.zeros(1, 64, ROPE, KV_LORA), torch.eye(ROPE).expand(1, 64, ROPE, ROPE)], dim=-1)
        bot = mk(mesh, bot_t, dims=(None, 1))
        aug = ttnn.concat([top, bot], dim=-2)
        tw_aug = ttnn.typecast(aug, ttnn.bfloat8_b)
        for t in (wb1_bf16, zr, top, bot, aug):
            ttnn.deallocate(t)

        def qb():
            return ttnn.linear(
                qn_in,
                tw_qb,
                program_config=qb_cfg["program_config"],
                memory_config=qb_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )

        def heads(t):
            return ttnn.experimental.nlp_create_q_heads_split(
                t, num_heads=HEADS_L, split_head_dim=NOPE, memory_config=DRAM
            )

        def b1(t, pc=None, mem=None):
            return ttnn.linear(
                t,
                tw_b1,
                program_config=pc or b1_cfg["program_config"],
                memory_config=mem or b1_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )

        def rope(t, prefix=None):
            return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                t,
                rope_t["cos_matrix"],
                rope_t["sin_matrix"],
                rope_t["trans_matrix"],
                kv_actual_global=0,
                cluster_axis=0,
                concat_prefix=prefix,
            )

        pc_aug5 = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(11, 10), in0_block_w=8, out_subblock_h=1, out_subblock_w=6, per_core_M=5, per_core_N=18
        )
        pc_aug4 = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(11, 10), in0_block_w=8, out_subblock_h=1, out_subblock_w=6, per_core_M=4, per_core_N=18
        )
        pc_aug4b = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(11, 10), in0_block_w=4, out_subblock_h=1, out_subblock_w=6, per_core_M=4, per_core_N=18
        )
        pc_b1_4 = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(11, 10), in0_block_w=6, out_subblock_h=1, out_subblock_w=8, per_core_M=4, per_core_N=16
        )
        pc_b1_4b = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(11, 10), in0_block_w=3, out_subblock_h=1, out_subblock_w=8, per_core_M=4, per_core_N=16
        )

        def aug_mm(t, pc, mem=DRAM):
            return ttnn.linear(t, tw_aug, program_config=pc, memory_config=mem, dtype=ttnn.bfloat16, compute_kernel_config=hifi2)

        def base_chain(b1pc=None):
            q = qb()
            qnope, qrope = heads(q)
            ttnn.deallocate(q)
            qa = b1(qnope, b1pc)
            ttnn.deallocate(qnope)
            out = rope(qrope, prefix=qa)
            ttnn.deallocate(qa)
            ttnn.deallocate(qrope)
            return out

        def aug_chain(pc, mem=DRAM):
            q = qb()
            qnope, qrope = heads(q)
            ttnn.deallocate(q)
            pre = rope(qrope, prefix=qnope)
            ttnn.deallocate(qnope)
            ttnn.deallocate(qrope)
            out = aug_mm(pre, pc, mem)
            ttnn.deallocate(pre)
            return out

        qb_ref = base_chain()
        for tag, pc in (("aug5", pc_aug5), ("aug4", pc_aug4), ("aug4b", pc_aug4b)):
            try:
                o = aug_chain(pc)
                cmp(f"C aug chain [{tag}] vs base", o, qb_ref)
                o_n = ttnn.slice(o, [0, 0, 0, 0], [1, HEADS_L, S, KV_LORA])
                b_n = ttnn.slice(qb_ref, [0, 0, 0, 0], [1, HEADS_L, S, KV_LORA])
                o_r = ttnn.slice(o, [0, 0, 0, KV_LORA], [1, HEADS_L, S, KV_LORA + ROPE])
                b_r = ttnn.slice(qb_ref, [0, 0, 0, KV_LORA], [1, HEADS_L, S, KV_LORA + ROPE])
                cmp(f"C aug chain [{tag}] nope part vs base", o_n, b_n)
                cmp(f"C aug chain [{tag}] rope part vs base", o_r, b_r)
                _free([o, o_n, b_n, o_r, b_r])
            except Exception as e:
                log(f"FAIL C aug[{tag}] pcc: {e}")
                traceback.print_exc()
        for tag, pc in (("b1_4", pc_b1_4), ("b1_4b", pc_b1_4b)):
            try:
                o = base_chain(pc)
                cmp(f"C base chain [{tag}] vs base", o, qb_ref)
                _free(o)
            except Exception as e:
                log(f"FAIL C base[{tag}] pcc: {e}")
        _free(qb_ref)

        timer("C0 q chain base (q_b, heads, wkv_b1, rope+concat512)", base_chain)
        timer("C0 q chain base wkv_b1 M4", lambda: base_chain(pc_b1_4))
        timer("C0 q chain base wkv_b1 M4 bw3", lambda: base_chain(pc_b1_4b))
        for tag, pc in (("aug5", pc_aug5), ("aug4", pc_aug4), ("aug4b", pc_aug4b)):
            timer(f"C1 q chain aug [{tag}] (q_b, heads, rope+concat192, wkv_b1_aug)", lambda pc=pc: aug_chain(pc))
        # pieces
        timer("C2 q_b mm", qb)
        q = qb()
        timer("C2 create_q_heads_split", lambda: list(heads(q)))
        qnope, qrope = heads(q)
        timer("C2 wkv_b1 base", lambda: b1(qnope))
        timer("C2 wkv_b1 M4", lambda: b1(qnope, pc_b1_4))
        qa = b1(qnope)
        timer("C2 rope + concat512 prefix (base)", lambda: rope(qrope, prefix=qa))
        timer("C2 rope alone [16,640,64]", lambda: rope(qrope))
        timer("C2 rope + concat192 prefix", lambda: rope(qrope, prefix=qnope))
        pre = rope(qrope, prefix=qnope)
        for tag, pc in (("aug5", pc_aug5), ("aug4", pc_aug4), ("aug4b", pc_aug4b)):
            timer(f"C2 wkv_b1 aug [{tag}] DRAM out", lambda pc=pc: aug_mm(pre, pc))
        _free([q, qnope, qrope, qa, pre])

    # ======================= D. MoE gate =======================
    if want("d"):
        from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import TtMoEGateConfig, gate_mm_config_key

        gcfg = TtMoEGateConfig.from_model_cfg(GLM53Config)
        key = gate_mm_config_key(gcfg.sp_dim, HL, 256)
        pc_g = gcfg.mm_configs_interleaved.get(key) or gcfg.mm_configs.get(key)
        ckg = gcfg.mm_configs["COMPUTE_CONFIG"]
        w_g = torch.randn(H, 256) * (1 / H**0.5)
        tw_g = mk(mesh, w_g, dims=(None, 0))  # production: K-sharded over TP
        tw_g_full = mk(mesh, w_g)  # replicated [6144, 256]
        w_gN = w_g.reshape(H, TP, 64).permute(1, 0, 2).reshape(TP * H, 64)  # [TP*6144, 64]
        tw_g_n = mk(mesh, w_gN, dims=(None, 0))  # per TP chip: [6144, 64] = its 64 experts, full K
        x_ffn = x_dram
        x_gath = ag_async(x_ffn)  # [1,1,640,6144] DRAM (MoE x all-gather, already in production)

        def gate_base():
            lg = ttnn.matmul(x_ffn, tw_g, compute_kernel_config=ckg, program_config=pc_g, memory_config=L1)
            out = ttnn.experimental.all_reduce_async(
                lg,
                cluster_axis=1,
                mesh_device=mesh,
                barrier_semaphores=tt_ccl.barrier_semaphore_handles[1],
                rs_global_semaphores=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
                ag_global_semaphores=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
                num_links=2,
                math_op=ttnn.ReduceType.Sum,
                memory_config=L1,
                topology=topo,
            )
            ttnn.deallocate(lg)
            return out

        def gate_full(pc=None):
            return ttnn.matmul(x_gath, tw_g_full, compute_kernel_config=ckg, program_config=pc, memory_config=L1)

        def gate_nshard(pc=None, agmem=L1):
            lg = ttnn.matmul(x_gath, tw_g_n, compute_kernel_config=ckg, program_config=pc, memory_config=L1)
            out = ag_async(lg, mem=agmem)
            ttnn.deallocate(lg)
            return out

        ref_lg = {}
        for d in (0, 5, 10, 31):
            ref_lg[d] = rows(x_full, d // TP)[0, 0] @ w_g
        try:
            gb = gate_base()
            cmp("D gate base vs torch", gb, ref_lg)
            gf = gate_full()
            cmp("D gate full-K replicated vs torch", gf, ref_lg)
            cmp("D gate full-K replicated vs base", gf, gb)
            gn = gate_nshard()
            cmp("D gate N-shard+AG vs torch", gn, ref_lg)
            _free([gb, gf, gn])
        except Exception as e:
            log(f"FAIL D pcc: {e}")
            traceback.print_exc()
        timer("D0 gate base (mm K-shard + all_reduce_async)", gate_base)
        timer("D1 gate full-K replicated mm (auto cfg)", gate_full)
        timer("D2 gate N-shard full-K mm + AG (auto cfg)", gate_nshard)
        lg = ttnn.matmul(x_ffn, tw_g, compute_kernel_config=ckg, program_config=pc_g, memory_config=L1)
        timer("D3 gate mm base alone", lambda: ttnn.matmul(x_ffn, tw_g, compute_kernel_config=ckg, program_config=pc_g, memory_config=L1))
        timer(
            "D3 gate all_reduce_async alone",
            lambda: ttnn.experimental.all_reduce_async(
                lg,
                cluster_axis=1,
                mesh_device=mesh,
                barrier_semaphores=tt_ccl.barrier_semaphore_handles[1],
                rs_global_semaphores=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
                ag_global_semaphores=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
                num_links=2,
                math_op=ttnn.ReduceType.Sum,
                memory_config=L1,
                topology=topo,
            ),
        )
        _free([lg, x_gath])

    # ======================= E. MoE x all-gather and big TP CCLs =======================
    if want("e"):
        x3 = mk(mesh, x_full.reshape(1, S * SP, H), mem=DRAM, dims=(1, 2))  # MoE input is 3D [1,640,1536]
        for w in (None, 2, 4, 8):
            timer(f"E0 MoE x all_gather_async workers={w}", lambda w=w: ag_async(x3, workers=w))
        buf_x = ttnn.empty([1, 1, S, H], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
        try:
            xg_hb = hbag(x_dram, buf_x)
            xg_as = ag_async(x_dram)
            cmp("E1 HBAG x vs all_gather_async", xg_hb, xg_as)
            ttnn.deallocate(xg_as)
        except Exception as e:
            log(f"FAIL E1 pcc: {e}")
        timer("E1 MoE x high_bw_all_gather [640,1536]->6144", lambda: [hbag(x_dram, buf_x)] and None)
        o_full = mk(mesh, torch.randn(1, 1, S, H), mem=L1)  # o_proj partial [640,6144] L1 (replicated data)
        for w in (None, 4):
            timer(f"E2 o_proj RS [640,6144] L1 workers={w}", lambda w=w: rs(o_full, workers=w))
        summed = mk(mesh, torch.randn(1, 1, S, H), mem=DRAM)  # post_combine_reduce output, DRAM TILE
        try:
            r1 = ttnn.reduce_scatter(summed, dim=-1, cluster_axis=1, num_links=2, topology=topo)
            r2 = rs(summed)
            cmp("E3 post-combine RS: minimal_async vs ttnn.reduce_scatter", r2, r1)
            _free([r1, r2])
        except Exception as e:
            log(f"FAIL E3 pcc: {e}")
        timer("E3 post-combine ttnn.reduce_scatter [640,6144] DRAM", lambda: ttnn.reduce_scatter(summed, dim=-1, cluster_axis=1, num_links=2, topology=topo))
        for w in (None, 4):
            timer(f"E3 post-combine RS minimal_async DRAM workers={w}", lambda w=w: rs(summed, workers=w))
        _free([o_full, x3, summed])

    # ======================= F. residual adds =======================
    if want("f"):
        a_d = x_dram
        b_d = mk(mesh, torch.randn(1, 1, S * SP, H), mem=DRAM, dims=(2, 3))
        b_l1 = mk(mesh, torch.randn(1, 1, S * SP, H), mem=L1, dims=(2, 3))
        timer("F0 add DRAM+DRAM->DRAM [640,1536]", lambda: ttnn.add(a_d, b_d))
        timer("F1 add DRAM+L1->DRAM", lambda: ttnn.add(a_d, b_l1))
        timer("F2 add DRAM+DRAM->L1", lambda: ttnn.add(a_d, b_d, memory_config=L1))
        timer("F3 add L1+L1->L1", lambda: ttnn.add(x_l1, b_l1, memory_config=L1))
        _free([b_d, b_l1])

    # ======================= G. R0-2: Option X q path, q_b input placement, L1 RS outputs =======================
    if want("g"):
        w_qa = torch.randn(H, Q_LORA) * (1 / H**0.5)
        g_q = torch.rand(Q_LORA) + 0.5
        tw_qa = mk(mesh, w_qa, ttnn.bfloat8_b, dims=(None, 0))
        tg_q = mk(mesh, g_q.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        # TP-sharded TILE row, built exactly like ttMLA._tp_shard_norm_weight
        host_g = ttnn.to_torch(ttnn.get_device_tensors(tg_q)[0]).reshape(1, 1, 1, Q_LORA)
        tg_q_tp = ttnn.from_torch(
            host_g,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3)),
        )
        qa_cfg, qb_cfg = glm_cfg("q_a_proj"), glm_cfg("q_b_proj")
        buf_q = tt_ccl.get_mla_high_bw_all_gather_buffer(
            name="q_a_latent", shape=[1, 1, S, Q_LORA], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        buf_q2 = ttnn.empty([1, 1, S, Q_LORA], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)

        def qa_mm():
            return ttnn.linear(
                x_l1, tw_qa, program_config=qa_cfg["program_config"], memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=hifi2
            )

        def distnorm(src):
            sems, stats = tt_ccl.get_fused_rmsnorm_resources(src, tg_q_tp, 1, 2)
            return ttnn.experimental.dit_fused_distributed_rmsnorm(
                src, 1, mesh, sems, topology=topo, epsilon=EPS, weight=tg_q_tp, persistent_output_buffer=stats,
                num_preferred_links=2, dtype=ttnn.bfloat16, memory_config=DRAM,
            )

        def q_base(workers=None):
            q = qa_mm()
            q2 = rs(q, workers=workers)
            ttnn.deallocate(q)
            q3 = hbag(q2, buf_q)
            ttnn.deallocate(q2)
            return ttnn.rms_norm(q3, weight=tg_q, epsilon=EPS, memory_config=L1, compute_kernel_config=hifi2)

        def q_x(workers=8, out=buf_q2):
            q = qa_mm()
            q2 = rs(q, workers=workers)
            ttnn.deallocate(q)
            q3 = distnorm(q2)
            ttnn.deallocate(q2)
            q4 = hbag(q3, out)
            ttnn.deallocate(q3)
            return None

        try:
            ref = q_base()
            q_x()
            cmp("G1 optionX qn (DRAM persistent) vs base", buf_q2, ref)
            refs = {}
            for d in (0, 5, 10, 31):
                refs[d] = rmsnorm_ref(rows(x_full, d // TP)[0, 0] @ w_qa, g_q)
            cmp("G1 optionX qn vs torch", buf_q2, refs)
            cmp("G1 base qn vs torch", ref, refs)
            _free(ref)
        except Exception as e:
            log(f"FAIL G1 pcc: {e}")
            traceback.print_exc()
        timer("G2 q path base (RS default workers)", lambda: q_base())
        timer("G2 q path base (RS 8 workers)", lambda: q_base(8))
        timer("G2 q path optionX (RS 8 workers, distnorm, AG)", lambda: q_x(8))
        timer("G2 q path optionX (RS default, distnorm, AG)", lambda: q_x(None))
        # q_b with in0 in L1 (production: norm output) vs DRAM (option X: persistent gather buffer)
        w_qb = torch.randn(Q_LORA, 64 * (NOPE + ROPE)) * (1 / Q_LORA**0.5)
        tw_qb = mk(mesh, w_qb, ttnn.bfloat8_b, dims=(None, 1))
        qn_l1 = mk(mesh, torch.randn(1, 1, S * SP, Q_LORA), mem=L1, dims=(2, None))
        qn_dram = mk(mesh, torch.randn(1, 1, S * SP, Q_LORA), mem=DRAM, dims=(2, None))
        for tag, src in (("L1", qn_l1), ("DRAM", qn_dram)):
            timer(
                f"G3 q_b mm in0 {tag}",
                lambda src=src: ttnn.linear(
                    src, tw_qb, program_config=qb_cfg["program_config"], memory_config=qb_cfg["out_mem_config"],
                    dtype=ttnn.bfloat16, compute_kernel_config=hifi2,
                ),
            )
        # HB-AG into an L1 output buffer
        try:
            buf_l1 = ttnn.empty([1, 1, S, Q_LORA], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=L1)
            q2 = rs(qa_mm(), workers=8)
            o = hbag(q2, buf_l1)
            o2 = hbag(q2, buf_q2)
            cmp("G4 HBAG L1 out vs DRAM out", buf_l1, buf_q2)
            timer("G4 HBAG [640,512]->2048 L1 out", lambda: [hbag(q2, buf_l1)] and None)
            timer("G4 HBAG [640,512]->2048 DRAM out", lambda: [hbag(q2, buf_q2)] and None)
        except Exception as e:
            log(f"FAIL G4 HBAG L1: {type(e).__name__}: {str(e)[:300]}")
        # o_proj RS with an L1 output, and the residual add reading it
        o_full = mk(mesh, torch.randn(1, 1, S, H), mem=L1)
        timer("G5 o_proj RS -> DRAM", lambda: rs(o_full))
        timer("G5 o_proj RS -> L1", lambda: rs(o_full, mem=L1))
        r_l1 = rs(o_full, mem=L1)
        r_d = rs(o_full)
        timer("G5 residual add x(DRAM) + rs(DRAM) -> DRAM", lambda: ttnn.add(x_dram, r_d))
        timer("G5 residual add x(DRAM) + rs(L1) -> DRAM", lambda: ttnn.add(x_dram, r_l1))
        # small TP RS (indexer wk partial [640,128])
        wk_p = mk(mesh, torch.randn(1, 1, S, 128), mem=DRAM)
        for w in (None, 4, 8):
            timer(f"G6 RS wk [640,128] workers={w}", lambda w=w: rs(wk_p, workers=w))
        _free([o_full, r_l1, r_d, wk_p, qn_l1, qn_dram])

    log("DONE")
