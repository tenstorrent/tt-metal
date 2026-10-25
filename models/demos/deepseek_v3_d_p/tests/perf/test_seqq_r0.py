# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""R0 microbench for the sequence-sharded absorbed-Q MLA path (GLM-5.3 prefill, 8x4 torus-XY, 5120 chunk).

Today each chip builds Q for its 640 SP rows x 16 heads, all-to-alls heads->seq (TP) for sparse_sdpa, and
all-to-alls seq->heads back before wkv_b2. The candidate builds Q for its 160 TP-quarter rows x all 64 heads
(q_b / wkv_b1 / wkv_b2 replicated), so the first all-to-all disappears and the second moves the 256-wide
wkv_b2 output instead of the 512-wide latent. Sections (SEQQ_R0_SECTIONS=w,a,q,o; default all):
  w  full weights by an on-device TP all_gather of the head-sharded ones (bit-exact vs a replicated upload)
  a  q_a stem: mm -> RS(dim3) -> AG -> norm (-> mesh_partition)  vs  mm -> RS(dim2 = rows) -> norm
  q  Q chain: q_b/heads/wkv_b1/rope/a2a/untilize  vs  q_b(full)/heads(64)/wkv_b1(full)/rope(subshard)/untilize
  o  out chain: tilize/a2a/wkv_b2/concat/o_proj/RS  vs  tilize/wkv_b2(full)/[a2a variants]/concat/o_proj/RS
Wall time per op sequence by trace replay (includes the inter-op gap as in the model's replay); PCC per device
vs the production sequence. Output lines start with "SEQQ_R0 ".
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
SQ = S // TP  # 160 rows per chip in the sequence-sharded formulation
H = 6144
HL = H // TP
Q_LORA, KV_LORA, ROPE, NOPE, VDIM = 2048, 512, 64, 192, 256
NH = 64
HEADS_L = NH // TP
EPS = 1e-6
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
GRID = (11, 10)

SECTIONS = os.environ.get("SEQQ_R0_SECTIONS", "all")
N_INNER = int(os.environ.get("SEQQ_R0_INNER", "8"))
N_REPLAY = int(os.environ.get("SEQQ_R0_REPLAY", "20"))
DEVS = (0, 5, 10, 15, 22, 31)


def want(name):
    return SECTIONS == "all" or name in SECTIONS.split(",")


def log(*a):
    print("SEQQ_R0", *a, flush=True)


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
    return ttnn.to_torch(ttnn.get_device_tensors(t)[idx]).float()


def cmp(name, t_new, t_ref, devs=DEVS, exact=False):
    vals, eq = [], []
    md = 0.0
    for d in devs:
        a = dev_torch(t_new, d) if isinstance(t_new, ttnn.Tensor) else t_new[d]
        b = dev_torch(t_ref, d) if isinstance(t_ref, ttnn.Tensor) else t_ref[d]
        if a.shape != b.shape:
            log(f"PCC {name}: SHAPE MISMATCH dev{d} {tuple(a.shape)} vs {tuple(b.shape)}")
            return -1.0
        vals.append(pcc(a, b))
        eq.append(bool(torch.equal(a, b)))
        md = max(md, (a - b).abs().max().item())
    log(f"PCC {name}: min={min(vals):.6f} vals={[round(v, 6) for v in vals]} maxabs={md:.4g} bitexact={all(eq)}")
    return min(vals)


class Timer:
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
        except Exception as e:
            log(f"FAIL {name}: {type(e).__name__}: {str(e)[:600]}")
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


def glm_cfg(key):
    for e in MLA_MATMUL_CONFIG[key][640]:
        if e.get("num_heads") == 64 and e.get("q_lora_rank") == 2048:
            return e
    raise KeyError(key)


def pc1d(per_core_n, in0_block_w=8, sub_w=None, grid=GRID):
    sub_w = sub_w or next(w for w in (8, 7, 6, 5, 4, 3, 2, 1) if per_core_n % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        per_core_M=SQ // 32,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def gather_tp(t, dim):
    """Replicate a TP-sharded weight by an on-device all_gather along the TP axis (axis 1)."""
    shape = list(t.shape)
    t4 = ttnn.reshape(t, [1] * (4 - len(shape)) + shape) if len(shape) < 4 else t
    dim4 = dim + (4 - len(shape)) if dim >= 0 else dim
    g = ttnn.all_gather(t4, dim=dim4, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear)
    if len(shape) < 4:
        out_shape = list(shape)
        out_shape[dim] *= TP
        g = ttnn.reshape(g, out_shape)
    return g


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
def test_seqq_r0(mesh_device, variant, config_only):
    mesh = mesh_device
    assert tuple(mesh.shape) == (SP, TP)
    torch.manual_seed(0)
    tt_ccl = get_tt_ccl(mesh)
    timer = Timer(mesh)
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    log(
        f"DRAM view (fresh mesh): banks={mv.num_banks} total/bank={mv.total_bytes_per_bank / 2**20:.1f} MiB "
        f"alloc/bank={mv.total_bytes_allocated_per_bank / 2**20:.1f} MiB free/bank={mv.total_bytes_free_per_bank / 2**20:.1f} MiB"
    )

    def ck(fid, fp32, packer=True):
        return ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=packer
        )

    hifi2 = ck(ttnn.MathFidelity.HiFi2, False)
    topo = ttnn.Topology.Ring

    def rs(x, dim=3, mem=DRAM):
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
        )

    def a2a(x, in_dim, out_dim):
        return ttnn.experimental.all_to_all_async_generic(
            x, in_dim=in_dim, out_dim=out_dim, num_links=2, memory_config=DRAM, cluster_axis=1
        )

    def hbag(x, out, dim=3):
        return ttnn.experimental.high_bw_all_gather(x, dim=dim, output_tensor=out, cluster_axis=1, num_links=2)

    # ======================= W. full weights by on-device gather =======================
    if want("w"):
        for name, shape, dim in (
            ("q_b", (Q_LORA, NH * (NOPE + ROPE)), 1),
            ("wkv_b1", (1, NH, NOPE, KV_LORA), 1),
            ("wkv_b2", (1, NH, KV_LORA, VDIM), 1),
        ):
            w = torch.randn(*shape)
            sh_dims = [None, None]
            sh_dims[1] = dim
            sharded = mk(mesh, w, ttnn.bfloat8_b, dims=tuple(sh_dims))
            ref = mk(mesh, w, ttnn.bfloat8_b, dims=(None, None))
            try:
                t0 = time.perf_counter()
                g = gather_tp(sharded, dim)
                ttnn.synchronize_device(mesh)
                dt = (time.perf_counter() - t0) * 1e3
                log(f"W {name}: gathered shape={tuple(g.shape)} dtype={g.dtype} in {dt:.1f} ms")
                cmp(f"W {name} all_gather(bfp8) vs replicated upload", g, ref, devs=(0, 7, 31))
                _free(g)
            except Exception as e:
                log(f"FAIL W {name} bfp8 all_gather: {type(e).__name__}: {str(e)[:400]}")
                traceback.print_exc()
                try:
                    b16 = ttnn.typecast(sharded, ttnn.bfloat16)
                    g16 = gather_tp(b16, dim)
                    g = ttnn.typecast(g16, ttnn.bfloat8_b)
                    cmp(f"W {name} typecast->all_gather->typecast vs replicated upload", g, ref, devs=(0, 7, 31))
                    _free([b16, g16, g])
                except Exception as e2:
                    log(f"FAIL W {name} bf16 fallback: {type(e2).__name__}: {str(e2)[:400]}")
                    traceback.print_exc()
            _free([sharded, ref])

    # ======================= A. q_a stem =======================
    if want("a"):
        x_full = torch.randn(1, 1, S * SP, H)
        x_l1 = mk(mesh, x_full, mem=L1, dims=(2, 3))
        w_qa = torch.randn(H, Q_LORA) * (1 / H**0.5)
        g_q = torch.rand(Q_LORA) + 0.5
        tw_qa = mk(mesh, w_qa, ttnn.bfloat8_b, dims=(None, 0))
        tg_q = mk(mesh, g_q.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
        qa_cfg = glm_cfg("q_a_proj")
        qb_act = glm_cfg("q_b_proj")["act_mem_config"]
        buf_q = tt_ccl.get_mla_high_bw_all_gather_buffer(
            name="q_a_latent", shape=[1, 1, S, Q_LORA], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        def qa_mm():
            return ttnn.linear(
                x_l1,
                tw_qa,
                program_config=qa_cfg["program_config"],
                memory_config=qa_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )

        def base_q(partition=False):
            q = qa_mm()
            q2 = rs(q)
            ttnn.deallocate(q)
            q3 = hbag(q2, buf_q)
            ttnn.deallocate(q2)
            qn = ttnn.rms_norm(q3, weight=tg_q, epsilon=EPS, memory_config=qb_act, compute_kernel_config=hifi2)
            if not partition:
                return qn
            ql = ttnn.mesh_partition(qn, dim=2, cluster_axis=1)
            ttnn.deallocate(qn)
            return ql

        def seq_q(norm_mem=DRAM, rs_mem=DRAM):
            q = qa_mm()
            q2 = rs(q, dim=2, mem=rs_mem)
            ttnn.deallocate(q)
            qn = ttnn.rms_norm(q2, weight=tg_q, epsilon=EPS, memory_config=norm_mem, compute_kernel_config=hifi2)
            ttnn.deallocate(q2)
            return qn

        ref = base_q(partition=True)
        for tag, kw in (("dram", {}), ("l1norm", {"norm_mem": L1}), ("l1rs", {"rs_mem": L1, "norm_mem": L1})):
            try:
                new = seq_q(**kw)
                log(f"A seq_q[{tag}] shape={tuple(new.shape)}")
                cmp(f"A q_a RS(dim2)+norm[{tag}] vs RS+AG+norm+partition", new, ref)
                _free(new)
            except Exception as e:
                log(f"FAIL A seq_q[{tag}]: {type(e).__name__}: {str(e)[:400]}")
                traceback.print_exc()
        _free(ref)
        timer("A0 base q_a stem (mm,RS3,AG,norm640)", base_q)
        timer("A1 base q_a stem + mesh_partition", lambda: base_q(True))
        timer("A2 seq q_a stem (mm,RS2,norm160) dram", seq_q)
        timer("A3 seq q_a stem norm->L1", lambda: seq_q(norm_mem=L1))
        timer("A4 seq q_a stem RS->L1 norm->L1", lambda: seq_q(rs_mem=L1, norm_mem=L1))
        q_mm = qa_mm()
        timer("A5 RS dim3 [640,2048]", lambda: rs(q_mm))
        timer("A6 RS dim2 [640,2048]", lambda: rs(q_mm, dim=2))
        _free([q_mm, x_l1, tw_qa])

    # ======================= Q. absorbed Q chain =======================
    if want("q"):
        from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup

        hf = copy.deepcopy(config_only)
        CACHE = 6 * S * SP
        hf.max_seq_len = CACHE
        rope_t = RotarySetup(hf, mesh, sp_axis=0).get_rope_tensors_indexed(CACHE, S * SP)
        qb_cfg, b1_cfg = glm_cfg("q_b_proj"), glm_cfg("wkv_b1")
        qr = mk(mesh, torch.randn(1, 1, S * SP, Q_LORA), mem=qb_cfg["act_mem_config"], dims=(2, None))
        qr_loc = ttnn.mesh_partition(qr, dim=2, cluster_axis=1)
        w_qb = torch.randn(Q_LORA, NH * (NOPE + ROPE)) * (1 / Q_LORA**0.5)
        tw_qb = mk(mesh, w_qb, ttnn.bfloat8_b, dims=(None, 1))
        tw_qb_full = gather_tp(tw_qb, 1)
        w_b1 = torch.randn(1, NH, NOPE, KV_LORA) * (1 / NOPE**0.5)
        tw_b1 = mk(mesh, w_b1, ttnn.bfloat8_b, dims=(None, 1))
        tw_b1_full = gather_tp(tw_b1, 1)
        b1_pc160 = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=GRID,
            in0_block_w=6,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=5,
            per_core_N=16,
        )

        def rope(t, kv, prefix, sub=None):
            return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                t,
                rope_t["cos_matrix"],
                rope_t["sin_matrix"],
                rope_t["trans_matrix"],
                kv_actual_global=kv,
                cluster_axis=0,
                seq_subshard_axis=sub,
                concat_prefix=prefix,
            )

        def base_chain(kv=5 * S * SP, rm=True):
            q = ttnn.linear(
                qr,
                tw_qb,
                program_config=qb_cfg["program_config"],
                memory_config=qb_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )
            nope, rp = ttnn.experimental.nlp_create_q_heads_split(
                q, num_heads=HEADS_L, split_head_dim=NOPE, memory_config=DRAM
            )
            ttnn.deallocate(q)
            nope2 = ttnn.linear(
                nope,
                tw_b1,
                program_config=b1_cfg["program_config"],
                memory_config=b1_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )
            ttnn.deallocate(nope)
            qq = rope(rp, kv, nope2)
            _free([rp, nope2])
            qs = a2a(qq, 1, 2)
            ttnn.deallocate(qq)
            if not rm:
                return qs
            out = ttnn.to_layout(qs, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(qs)
            return out

        def qb_full(pc=None, mem=L1):
            kw = {"program_config": pc} if pc is not None else {}
            return ttnn.linear(
                qr_loc, tw_qb_full, memory_config=mem, dtype=ttnn.bfloat16, compute_kernel_config=hifi2, **kw
            )

        def seq_chain(kv=5 * S * SP, pc=None, qb_mem=L1, b1_mem=L1, rm=True, partition=False):
            src = ttnn.mesh_partition(qr, dim=2, cluster_axis=1) if partition else qr_loc
            kw = {"program_config": pc} if pc is not None else {}
            q = ttnn.linear(
                src, tw_qb_full, memory_config=qb_mem, dtype=ttnn.bfloat16, compute_kernel_config=hifi2, **kw
            )
            if partition:
                ttnn.deallocate(src)
            nope, rp = ttnn.experimental.nlp_create_q_heads_split(
                q, num_heads=NH, split_head_dim=NOPE, memory_config=DRAM
            )
            ttnn.deallocate(q)
            nope2 = ttnn.linear(
                nope,
                tw_b1_full,
                program_config=b1_pc160,
                memory_config=b1_mem,
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )
            ttnn.deallocate(nope)
            qq = rope(rp, kv, nope2, sub=1)
            _free([rp, nope2])
            if not rm:
                return qq
            out = ttnn.to_layout(qq, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(qq)
            return out

        qb_variants = [
            ("auto", None),
            ("1d_n5_k8", pc1d(5, 8)),
            ("1d_n5_k16", pc1d(5, 16)),
            ("1d_n5_k4", pc1d(5, 4)),
            ("1d_n8_k8", pc1d(8, 8)),
            ("1d_n8_k16", pc1d(8, 16)),
            ("1d_n6_k8", pc1d(6, 8, sub_w=6)),
        ]
        # correctness (two depths: aligned and a rotated, mid-slab start)
        for kv in (5 * S * SP, 1024):
            ref = base_chain(kv)
            for tag, pc in qb_variants[:2]:
                try:
                    new = seq_chain(kv, pc)
                    log(f"Q seq_chain[{tag}] kv={kv} shape={tuple(new.shape)}")
                    cmp(f"Q seq chain[{tag}] vs base (kv={kv})", new, ref)
                    _free(new)
                except Exception as e:
                    log(f"FAIL Q seq chain[{tag}] kv={kv}: {type(e).__name__}: {str(e)[:400]}")
                    traceback.print_exc()
            _free(ref)

        timer("Q0 base chain (q_b,heads16,wkv_b1,rope+cat,a2a h2s,untilize)", base_chain)
        timer("Q0b base chain without untilize", lambda: base_chain(rm=False))
        best = None
        for tag, pc in qb_variants:
            us = timer(f"Q1 q_b full [{tag}] [160,2048]x[2048,16384]", lambda pc=pc: qb_full(pc))
            if us is not None and (best is None or us < best[0]):
                best = (us, tag, pc)
        timer("Q1 q_b full [1d_n5_k8] DRAM out", lambda: qb_full(pc1d(5, 8), mem=DRAM))
        log(f"Q best q_b full: {best[1] if best else None} {best[0] if best else None}")
        bpc = best[2] if best else None
        timer(
            f"Q2 seq chain [{best[1] if best else 'auto'}] (q_b,heads64,wkv_b1,rope sub+cat,untilize)",
            lambda: seq_chain(pc=bpc),
        )
        timer("Q2b seq chain without untilize", lambda: seq_chain(pc=bpc, rm=False))
        timer("Q2c seq chain + mesh_partition", lambda: seq_chain(pc=bpc, partition=True))
        timer("Q2d seq chain wkv_b1 out DRAM", lambda: seq_chain(pc=bpc, b1_mem=DRAM))
        # pieces
        q_base = ttnn.linear(
            qr,
            tw_qb,
            program_config=qb_cfg["program_config"],
            memory_config=qb_cfg["out_mem_config"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=hifi2,
        )
        q_seq = qb_full(bpc)
        timer(
            "Q3 base q_b [640,2048]x[2048,4096]",
            lambda: ttnn.linear(
                qr,
                tw_qb,
                program_config=qb_cfg["program_config"],
                memory_config=qb_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            ),
        )
        timer(
            "Q3 base heads16",
            lambda: list(
                ttnn.experimental.nlp_create_q_heads_split(
                    q_base, num_heads=HEADS_L, split_head_dim=NOPE, memory_config=DRAM
                )
            ),
        )
        timer(
            "Q3 seq heads64",
            lambda: list(
                ttnn.experimental.nlp_create_q_heads_split(q_seq, num_heads=NH, split_head_dim=NOPE, memory_config=DRAM)
            ),
        )
        nb, rb = ttnn.experimental.nlp_create_q_heads_split(
            q_base, num_heads=HEADS_L, split_head_dim=NOPE, memory_config=DRAM
        )
        ns, rsq = ttnn.experimental.nlp_create_q_heads_split(
            q_seq, num_heads=NH, split_head_dim=NOPE, memory_config=DRAM
        )
        timer(
            "Q4 base wkv_b1 16x[640,192]x[192,512]",
            lambda: ttnn.linear(
                nb,
                tw_b1,
                program_config=b1_cfg["program_config"],
                memory_config=b1_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            ),
        )
        timer(
            "Q4 seq wkv_b1 64x[160,192]x[192,512]",
            lambda: ttnn.linear(
                ns,
                tw_b1_full,
                program_config=b1_pc160,
                memory_config=L1,
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            ),
        )
        nb2 = ttnn.linear(
            nb,
            tw_b1,
            program_config=b1_cfg["program_config"],
            memory_config=b1_cfg["out_mem_config"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=hifi2,
        )
        ns2 = ttnn.linear(
            ns, tw_b1_full, program_config=b1_pc160, memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=hifi2
        )
        timer("Q5 base rope+cat [16,640]", lambda: rope(rb, 5 * S * SP, nb2))
        timer("Q5 seq rope+cat sub [64,160]", lambda: rope(rsq, 5 * S * SP, ns2, sub=1))
        qq_b = rope(rb, 5 * S * SP, nb2)
        timer("Q6 base a2a h2s [16,640,576]->[64,160,576]", lambda: a2a(qq_b, 1, 2))
        _free([q_base, q_seq, nb, rb, ns, rsq, nb2, ns2, qq_b, qr, qr_loc, tw_qb, tw_qb_full, tw_b1, tw_b1_full])

    # ======================= O. output chain =======================
    if want("o"):
        b2_cfg, o_cfg = glm_cfg("wkv_b2"), glm_cfg("o_proj")
        attn_full = torch.randn(1, NH, S * SP, KV_LORA)
        attn_sp = mk(mesh, attn_full, dims=(2, None))
        attn_tile = ttnn.mesh_partition(attn_sp, dim=2, cluster_axis=1)  # [1,64,160,512] rows of TP quarter
        ttnn.deallocate(attn_sp)
        attn_rm = ttnn.to_layout(attn_tile, ttnn.ROW_MAJOR_LAYOUT)  # what sparse_sdpa emits
        ttnn.deallocate(attn_tile)
        w_b2 = torch.randn(1, NH, KV_LORA, VDIM) * (1 / KV_LORA**0.5)
        tw_b2 = mk(mesh, w_b2, ttnn.bfloat8_b, dims=(None, 1))
        tw_b2_full = gather_tp(tw_b2, 1)
        w_o = torch.randn(NH * VDIM, H) * (1 / (NH * VDIM) ** 0.5)
        tw_o = mk(mesh, w_o, ttnn.bfloat8_b, dims=(None, 0))
        b2_pc160 = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=GRID,
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=5,
            per_core_N=8,
        )

        def oproj_rs(v):
            o = ttnn.linear(
                v,
                tw_o,
                program_config=o_cfg["program_config"],
                memory_config=o_cfg["out_mem_config"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=hifi2,
            )
            out = rs(o)
            ttnn.deallocate(o)
            return out

        def base_out(full=True):
            t = ttnn.to_layout(attn_rm, ttnn.TILE_LAYOUT)
            h = a2a(t, 2, 1)
            ttnn.deallocate(t)
            v = ttnn.linear(
                h,
                tw_b2,
                program_config=b2_cfg["program_config"],
                memory_config=b2_cfg["out_mem_config"],
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=hifi2,
            )
            ttnn.deallocate(h)
            c = ttnn.experimental.nlp_concat_heads(v, memory_config=DRAM)
            ttnn.deallocate(v)
            if not full:
                return c
            out = oproj_rs(c)
            ttnn.deallocate(c)
            return out

        def seq_out(mode="heads", dtype=ttnn.bfloat8_b, full=True, b2_mem=L1):
            t = ttnn.to_layout(attn_rm, ttnn.TILE_LAYOUT)
            v = ttnn.linear(
                t, tw_b2_full, program_config=b2_pc160, memory_config=b2_mem, dtype=dtype, compute_kernel_config=hifi2
            )
            ttnn.deallocate(t)
            if mode == "heads":  # a2a on [1,64,160,256] (seq -> heads) then concat heads
                h = a2a(v, 2, 1)
                ttnn.deallocate(v)
                c = ttnn.experimental.nlp_concat_heads(h, memory_config=DRAM)
                ttnn.deallocate(h)
            else:  # concat heads [1,1,160,16384] then a2a (cols -> rows)
                cc = ttnn.experimental.nlp_concat_heads(v, memory_config=DRAM)
                ttnn.deallocate(v)
                c = a2a(cc, 2, 3)
                ttnn.deallocate(cc)
            if not full:
                return c
            out = oproj_rs(c)
            ttnn.deallocate(c)
            return out

        ref_c = base_out(full=False)
        ref = base_out()
        variants = [
            ("heads-bfp8-dram", dict(mode="heads", dtype=ttnn.bfloat8_b, b2_mem=DRAM)),
            ("heads-bfp8-l1", dict(mode="heads", dtype=ttnn.bfloat8_b, b2_mem=L1)),
            ("cols-bfp8", dict(mode="cols", dtype=ttnn.bfloat8_b)),
            ("heads-bf16-dram", dict(mode="heads", dtype=ttnn.bfloat16, b2_mem=DRAM)),
            ("cols-bf16", dict(mode="cols", dtype=ttnn.bfloat16)),
        ]
        for tag, kw in variants:
            try:
                c = seq_out(full=False, **kw)
                cmp(f"O seq[{tag}] pre-o_proj [640,4096] vs base", c, ref_c)
                _free(c)
                o = seq_out(**kw)
                cmp(f"O seq[{tag}] after o_proj+RS vs base", o, ref)
                _free(o)
            except Exception as e:
                log(f"FAIL O seq[{tag}]: {type(e).__name__}: {str(e)[:400]}")
                traceback.print_exc()
        _free([ref, ref_c])
        timer("O0 base out chain (tilize,a2a s2h 512,wkv_b2,concat,o_proj,RS)", base_out)
        timer("O0b base out chain w/o o_proj+RS", lambda: base_out(full=False))
        for tag, kw in variants:
            timer(f"O1 seq out chain [{tag}]", lambda kw=kw: seq_out(**kw))
            timer(f"O1b seq out chain [{tag}] w/o o_proj+RS", lambda kw=kw: seq_out(full=False, **kw))
        # pieces
        t_til = ttnn.to_layout(attn_rm, ttnn.TILE_LAYOUT)
        timer("O2 tilize [64,160,512]", lambda: ttnn.to_layout(attn_rm, ttnn.TILE_LAYOUT))
        timer("O3 base a2a s2h [64,160,512] bf16", lambda: a2a(t_til, 2, 1))
        h_b = a2a(t_til, 2, 1)
        timer(
            "O4 base wkv_b2 16x[640,512]x[512,256]",
            lambda: ttnn.linear(
                h_b,
                tw_b2,
                program_config=b2_cfg["program_config"],
                memory_config=b2_cfg["out_mem_config"],
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=hifi2,
            ),
        )
        timer(
            "O4 seq wkv_b2 64x[160,512]x[512,256]",
            lambda: ttnn.linear(
                t_til,
                tw_b2_full,
                program_config=b2_pc160,
                memory_config=L1,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=hifi2,
            ),
        )
        for dt, dn in ((ttnn.bfloat8_b, "bfp8"), (ttnn.bfloat16, "bf16")):
            v = ttnn.linear(
                t_til, tw_b2_full, program_config=b2_pc160, memory_config=DRAM, dtype=dt, compute_kernel_config=hifi2
            )
            timer(f"O5 a2a s2h [64,160,256] {dn} (DRAM in)", lambda v=v: a2a(v, 2, 1))
            try:  # bf16 [64,160,256] nlp_concat_heads fails its CB allocation (R0-1); keep going
                cc = ttnn.experimental.nlp_concat_heads(v, memory_config=DRAM)
            except Exception as e:
                log(f"FAIL O5 concat heads {dn}: {type(e).__name__}: {str(e)[:200]}")
                _free(v)
                continue
            timer(
                f"O5 concat heads [64,160,256] {dn}",
                lambda v=v: ttnn.experimental.nlp_concat_heads(v, memory_config=DRAM),
            )
            timer(f"O5 a2a cols->rows [160,16384] {dn}", lambda cc=cc: a2a(cc, 2, 3))
            _free([v, cc])
        _free([t_til, h_b, attn_rm, tw_b2, tw_b2_full, tw_o])

    log("DONE")
