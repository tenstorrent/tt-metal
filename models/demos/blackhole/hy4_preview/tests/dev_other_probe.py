# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""layer-other fusion-candidate probes (same slot / harness as dev_other.py): each candidate vs the unfused ops it would replace, traced, REP back to back.
Env: KIND, S, WDT (default bfp8), PROBES=comma list of mixes,ep,rms,fuseproj,absorb,rope,ccl (default all). Timing only (random data), PCC where cheap."""
import json
import os
import statistics
import time

import torch

import ttnn

os.environ.setdefault("WDT", "bfp8")
os.environ.setdefault("TUNE", "1")
os.environ.setdefault("RING", "1")
os.environ.setdefault("LEAN", "1")
from models.demos.blackhole.hy4_preview.tests import dev_other as DO  # noqa: E402
from models.demos.blackhole.hy4_preview.tests.dev_other import HID, S, build, cfg_1d, free, rep_up  # noqa: E402

PROBES = os.environ.get("PROBES", "mixes,ep,rms,rms2,fuseproj,absorb,rope,ccl,dsm,ihc2").split(",")
OUT = os.environ.get("OUT", "/tmp/probe.json")
R = {}


def P(*a):
    print("PROBE", *a, flush=True)


def t_us(md, fn, rep_=10, n=7):
    def body():
        for _ in range(rep_):
            o = fn()
            for t in o if isinstance(o, (tuple, list)) else [o]:
                if isinstance(t, ttnn.Tensor):
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / rep_ * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def rec(md, name, fn, **kw):
    try:
        us = t_us(md, fn, **kw)
        R[name] = us
        P(f"{name:70s} {us:8.1f} us")
    except Exception as e:
        R[name] = None
        P(f"{name:70s} FAILED {str(e)[:160]}")


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
    try:
        B = build(md)
        a, cfg = B["attn"], B["cfg"]
        ckc = a.ckc
        ia, im = B["ihc_a"], B["ihc_m"]
        x = B["x"]
        post = comb = None
        pre, post, comb = ia.mixes(x)
        y = rep_up(md, torch.randn(1, 1, S, HID))
        x2 = ia.expand(y, x, post, comb)
        ttnn.synchronize_device(md)
        if "mixes" in PROBES:
            rec(md, "ihc mixes on persistent x", lambda: ia.mixes(x))
            rec(md, "ihc mixes on expand output x2", lambda: im.mixes(x2))
            rec(md, "ihc expand", lambda: ia.expand(y, x, post, comb))
            pre2, post2, comb2 = im.mixes(x2)
            rec(md, "ihc collapse_norm (tile out)", lambda: im.collapse_norm(x2, pre2, B["w_post"], 1e-5))
            rec(md, "ihc collapse_norm_rm", lambda: im.collapse_norm_rm(x2, pre2, B["w_post"], 1e-5))
            rec(
                md,
                "chain expand + mixes + collapse_norm_rm (separate)",
                lambda: (lambda xx: (lambda m: im.collapse_norm_rm(xx, m[0], B["w_post"], 1e-5))(im.mixes(xx)))(
                    ia.expand(y, x, post, comb)
                ),
            )
        if "ep" in PROBES:
            os.environ["DSV41_MHC_EP"] = "1"
            try:
                xn, (p3, po3, c3) = ia.expand_mixes(y, x, post, comb, None, im)
                ttnn.synchronize_device(md)
                rec(
                    md,
                    "EP fused expand+next mixes (DSV41_MHC_EP=1)",
                    lambda: ia.expand_mixes(y, x, post, comb, None, im),
                )
                rec(md, "separate expand + next mixes", lambda: im.mixes(ia.expand(y, x, post, comb)))
                # accuracy: fused vs separate mixes
                xs = ia.expand(y, x, post, comb)
                ps, pos_s, cs = im.mixes(xs)
                f = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
                P(
                    "EP pre maxabs diff",
                    float((f(p3) - f(ps)).abs().max()),
                    "post",
                    float((f(po3) - f(pos_s)).abs().max()),
                    "x_new",
                    float((f(xn) - f(xs)).abs().max()),
                )
            except Exception as e:
                import traceback

                P("EP FAILED", repr(e)[:300], traceback.format_exc()[-1500:])
            os.environ["DSV41_MHC_EP"] = "0"
        if "rms" in PROBES:
            qg = rep_up(md, torch.randn(1, 1, S, 2048))
            kvl = rep_up(md, torch.randn(1, 1, S, 512))
            MF = ttnn.MathFidelity
            mk = lambda m, f32: ttnn.init_device_compute_kernel_config(
                md.arch(), math_fidelity=m, math_approx_mode=False, fp32_dest_acc_en=f32, packer_l1_acc=False
            )
            for nm, x_, w_ in (("q_a(2048)", qg, a.q_a_norm), ("kv_lat(512)", kvl, a.kv_a_norm)):
                rec(
                    md,
                    f"rms {nm} hifi4 fp32acc (as is)",
                    lambda: ttnn.rms_norm(x_, weight=w_, epsilon=1e-6, compute_kernel_config=ckc),
                )
                rec(
                    md,
                    f"rms {nm} hifi2 no-fp32",
                    lambda: ttnn.rms_norm(x_, weight=w_, epsilon=1e-6, compute_kernel_config=mk(MF.HiFi2, False)),
                )
                rec(
                    md,
                    f"rms {nm} lofi no-fp32",
                    lambda: ttnn.rms_norm(x_, weight=w_, epsilon=1e-6, compute_kernel_config=mk(MF.LoFi, False)),
                )
                rec(md, f"rms {nm} default cfg", lambda: ttnn.rms_norm(x_, weight=w_, epsilon=1e-6))
                xl = ttnn.to_memory_config(x_, ttnn.L1_MEMORY_CONFIG)
                rec(
                    md,
                    f"rms {nm} L1 in/out lofi",
                    lambda: ttnn.rms_norm(
                        xl,
                        weight=w_,
                        epsilon=1e-6,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        compute_kernel_config=mk(MF.LoFi, False),
                    ),
                )
            # sharded rms over 8 cores (width)
            try:
                W = 2048
                nc = 8
                mc = ttnn.create_sharded_memory_config(
                    shape=(32, W // nc),
                    core_grid=ttnn.CoreGrid(y=1, x=nc),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                xs_ = ttnn.to_memory_config(qg, mc)
                g = rep_up(md, torch.ones(1, 1, W // 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(nc, 1),
                    subblock_w=(W // nc // 32),
                    block_h=1,
                    block_w=(W // nc // 32),
                    inplace=False,
                )
                rec(
                    md,
                    "rms q_a(2048) WIDTH-sharded 8 cores",
                    lambda: ttnn.rms_norm(
                        xs_, weight=g, epsilon=1e-6, program_config=pc, memory_config=mc, compute_kernel_config=ckc
                    ),
                )
            except Exception as e:
                P("sharded rms FAILED", str(e)[:200])
        if "rms2" in PROBES:
            qg = rep_up(md, torch.randn(1, 1, S, 2048))
            W = 2048
            for nc, (gy, gx) in ((16, (2, 8)), (32, (4, 8)), (64, (8, 8))):
                try:
                    mc = ttnn.create_sharded_memory_config(
                        shape=(32, W // nc),
                        core_grid=ttnn.CoreGrid(y=gy, x=gx),
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    xs_ = ttnn.to_memory_config(qg, mc)
                    g = rep_up(md, torch.ones(1, 1, W // 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                    sw = W // nc // 32
                    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        subblock_w=min(sw, 4) if sw % min(sw, 4) == 0 else 1,
                        block_h=1,
                        block_w=sw,
                        inplace=False,
                    )
                    rec(
                        md,
                        f"rms q_a(2048) WIDTH-sharded {nc} cores (rms only, input already sharded)",
                        lambda: ttnn.rms_norm(
                            xs_, weight=g, epsilon=1e-6, program_config=pc, memory_config=mc, compute_kernel_config=ckc
                        ),
                    )
                    rec(
                        md,
                        f"rms q_a(2048) sharded {nc}: to_memory_config in + rms + back",
                        lambda: ttnn.to_memory_config(
                            ttnn.rms_norm(
                                ttnn.to_memory_config(qg, mc),
                                weight=g,
                                epsilon=1e-6,
                                program_config=pc,
                                memory_config=mc,
                                compute_kernel_config=ckc,
                            ),
                            ttnn.L1_MEMORY_CONFIG,
                        ),
                    )
                except Exception as e:
                    P(f"sharded {nc} FAILED", str(e)[:250])
        if "fuseproj" in PROBES:
            hw = a.host_w
            full = B["full"]
            cols = 8
            parts = []
            for c in range(cols):
                p_ = [
                    hw["q_a"].tensor[..., c * 256 : (c + 1) * 256],
                    hw["kv_a"].tensor,
                    hw["gate"].tensor[..., c * 2048 : (c + 1) * 2048],
                ]
                if full:
                    p_ += [hw["idx_wk"].tensor, hw["idx_wp"].tensor]
                parts.append(torch.cat(p_, dim=-1))
            Wcat = torch.cat(parts, dim=-1)
            Nloc = Wcat.shape[-1] // cols
            P("fused h-projection local N", Nloc, "tiles", Nloc // 32)
            from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

            wf = upload(md, Wcat, DO.WDT, ttnn.TILE_LAYOUT, shard_dim=3)
            h = rep_up(md, torch.randn(1, 1, S, HID))
            offs = [0, 256, 256 + 576, 256 + 576 + 2048] + ([256 + 576 + 2048 + 128, Nloc] if full else [Nloc])
            sep = lambda: (
                DO.lin(h, a.w_q_a, "q_a", ckc),
                DO.lin(h, a.w_kv_a, "kv_a", ckc),
                DO.lin(h, a.w_gate, "gate", ckc, "sigmoid"),
            ) + ((DO.lin(h, a.indexer.w_k, "idx_wk", ckc), DO.lin(h, a.indexer.w_p, "idx_wp", ckc)) if full else ())
            rec(
                md,
                f"h-projections SEPARATE ({'full' if full else 'shared'} layer: q_a,kv_a,gate{',idx_wk,idx_wp' if full else ''})",
                sep,
            )
            for cap in (32, 64, 96, 120):
                pc = cfg_1d(6144, Nloc, cap)
                rec(
                    md,
                    f"h-projection FUSED one linear N={Nloc} cap {cap} (no slices)",
                    lambda: ttnn.linear(h, wf, compute_kernel_config=ckc, program_config=pc),
                )

                def fused_sl(pc=pc):
                    o = ttnn.linear(h, wf, compute_kernel_config=ckc, program_config=pc)
                    sl = [ttnn.slice(o, [0, 0, 0, offs[i]], [1, 1, S, offs[i + 1]]) for i in range(len(offs) - 1)]
                    ttnn.deallocate(o)
                    return sl

                rec(md, f"h-projection FUSED cap {cap} + {len(offs)-1} slices", fused_sl)
        if "absorb" in PROBES:
            MF = ttnn.MathFidelity
            mk = lambda m, f32: ttnn.init_device_compute_kernel_config(
                md.arch(), math_fidelity=m, math_approx_mode=False, fp32_dest_acc_en=f32, packer_l1_acc=False
            )
            qn = rep_up(md, torch.randn(1, 8, S, 192))
            ol = rep_up(md, torch.randn(1, 8, S, 512))
            hwk = a.host_w["w_uk"].tensor
            hwv = a.host_w["w_uv_t"].tensor
            P("w_uk host", tuple(hwk.shape), "shard", a.host_w["w_uk"].shard_dim, "w_uv_t", tuple(hwv.shape))
            from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

            for nm, x_, w_bf, host in (
                ("w_uk q_nope[8,S,192]x[192,512]", qn, a.w_uk, hwk),
                ("w_uv o_lat[8,S,512]x[512,256]", ol, a.w_uv_t, hwv),
            ):
                rec(md, f"{nm} bf16 hifi4 (as is)", lambda: ttnn.matmul(x_, w_bf, compute_kernel_config=ckc))
                rec(
                    md,
                    f"{nm} bf16 lofi no-fp32",
                    lambda: ttnn.matmul(x_, w_bf, compute_kernel_config=mk(MF.LoFi, False)),
                )
                rec(
                    md,
                    f"{nm} bf16 hifi2 no-fp32",
                    lambda: ttnn.matmul(x_, w_bf, compute_kernel_config=mk(MF.HiFi2, False)),
                )
                w8 = upload(
                    md,
                    host,
                    ttnn.bfloat8_b,
                    ttnn.TILE_LAYOUT,
                    shard_dim=a.host_w["w_uk" if "w_uk" in nm else "w_uv_t"].shard_dim,
                )
                rec(md, f"{nm} bfp8 hifi2", lambda: ttnn.matmul(x_, w8, compute_kernel_config=mk(MF.HiFi2, False)))
                rec(md, f"{nm} bfp8 lofi", lambda: ttnn.matmul(x_, w8, compute_kernel_config=mk(MF.LoFi, False)))
                rec(
                    md,
                    f"{nm} bf16 L1 out",
                    lambda: ttnn.matmul(x_, w_bf, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG),
                )
                rec(
                    md,
                    f"{nm} bf16 core_grid 1x8",
                    lambda: ttnn.matmul(x_, w_bf, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=1, x=8)),
                )
        if "rope" in PROBES:
            from models.tt_transformers.tt.common import get_rot_transformation_mat

            tm = rep_up(md, get_rot_transformation_mat())
            qpe = rep_up(md, torch.randn(1, 8, S, 64))
            kpe = rep_up(md, torch.randn(1, 1, S, 64))
            cos, sin = B["cos"], B["sin"]
            ck = a.ckc
            rec(md, "rope q_pe [8,S,64] as is (matmul,mul,mul,add)", lambda: a.rope.apply(qpe, cos, sin, ck))
            rec(
                md,
                "rope q_pe rotary_embedding_llama (1 op)",
                lambda: ttnn.experimental.rotary_embedding_llama(
                    qpe, cos, sin, tm, is_decode_mode=False, compute_kernel_config=ck
                ),
            )
            rec(md, "rope k_pe [1,S,64] as is", lambda: a.rope.apply(kpe, cos, sin, ck))
            rec(
                md,
                "rope k_pe rotary_embedding_llama (1 op)",
                lambda: ttnn.experimental.rotary_embedding_llama(
                    kpe, cos, sin, tm, is_decode_mode=False, compute_kernel_config=ck
                ),
            )
            q2 = rep_up(md, torch.randn(1, 32, S, 128))
            rec(
                md,
                "idx q rope [32,S,128] as is (slice,rope,slice,concat)",
                lambda: a.indexer._rot_first(q2, B["icos"], B["isin"]),
            ) if B["full"] else None
        if "dsm" in PROBES:
            import math

            from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

            dcores = md.dram_grid_size().x
            P("dram grid", md.dram_grid_size())
            dram_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dcores - 1, 0))})

            def dsm_case(name, K, N, ncores, host_w, shard_dim, dt):
                Nt = math.ceil(N / (32 * dcores)) * dcores  # padded tiles
                Np = Nt * 32
                hw_ = host_w
                if Np != N:
                    pad = torch.zeros(*hw_.shape[:-1], Np - N)
                    # per column shard handled by the caller: weights are padded per device block
                wmc = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dram_set, (K, Np // dcores), ttnn.ShardOrientation.ROW_MAJOR),
                )
                return Np, wmc

            def run_dsm(name, K, N, ncores, mk_host, shard_dim, fused_act=None):
                Np, wmc = dsm_case(name, K, N, ncores, None, shard_dim, None)
                hostw = mk_host(Np)
                w = ttnn.from_torch(
                    hostw,
                    device=md,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=wmc,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, shard_dim), mesh_shape=tuple(md.shape))
                    if shard_dim is not None
                    else ttnn.ReplicateTensorToMesh(md),
                )
                gy = -(-ncores // 8)
                gx = min(ncores, 8)
                cg = ttnn.CoreGrid(y=gy, x=gx)
                in_mc = ttnn.create_sharded_memory_config(
                    shape=(32, K // ncores),
                    core_grid=cg,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                out_mc = ttnn.create_sharded_memory_config(
                    shape=(32, Np // ncores),
                    core_grid=cg,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=(K // 32) // ncores,
                    per_core_M=1,
                    per_core_N=(Np // 32) // ncores,
                    fused_activation=fused_act,
                )
                x_ = rep_up(md, torch.randn(1, 1, S, K))
                xs_ = ttnn.to_memory_config(x_, in_mc)
                f_core = lambda: ttnn.linear(xs_, w, program_config=pc, memory_config=out_mc, compute_kernel_config=ckc)
                rec(md, f"DRAM-sharded {name} K{K} N{N}->{Np} {ncores} cores (matmul only, sharded in/out)", f_core)
                rec(
                    md,
                    f"DRAM-sharded {name} + reshard in (DRAM->L1 sharded) + out to interleaved L1",
                    lambda: ttnn.to_memory_config(
                        ttnn.linear(
                            ttnn.to_memory_config(x_, in_mc),
                            w,
                            program_config=pc,
                            memory_config=out_mc,
                            compute_kernel_config=ckc,
                        ),
                        ttnn.L1_MEMORY_CONFIG,
                    ),
                )

            tw = lambda a, b: (lambda Np: torch.randn(1, 1, a, 8 * Np if False else Np))
            # per-device weights: shard over 8 columns on dim 3 -> pass [1,1,K, 8*Np]
            for name, K, N, nc in (
                ("fused h-proj", 6144, 3040, 24),
                ("o_proj", 2048, 6144, 16),
                ("q_b", 2048, 2048, 16),
                ("gate", 6144, 2048, 24),
                ("q_a+kv_a", 6144, 832, 24),
            ):
                try:
                    if name == "o_proj":
                        mk = lambda Np, K=K: torch.randn(
                            1, 1, 8 * K, Np
                        )  # row-sharded weight: each column holds [K, N]; emulate with a replicated [K,Np]
                        # row-shard on dim 2 of a [1,1,8K,N] tensor
                        run_dsm(name, K, N, nc, lambda Np, K=K: torch.randn(1, 1, 8 * K, Np), 2)
                    else:
                        run_dsm(name, K, N, nc, lambda Np, K=K: torch.randn(1, 1, K, 8 * Np), 3)
                except Exception as e:
                    P(f"dsm {name} FAILED", str(e)[:300])
        if "ihc2" in PROBES:
            from models.demos.blackhole.hy4_preview.tt.hy4_ihc import Hy4IHC

            xa = ia.expand(y, x, post, comb)
            pre_x, post_x, comb_x = ia.mixes(x)
            xm = im.expand(y, xa, post_x, comb_x)
            ttnn.synchronize_device(md)
            rec(md, "mixes(attn ihc) on persistent x", lambda: ia.mixes(x))
            rec(md, "mixes(attn ihc) on mlp-expand output", lambda: ia.mixes(xm))
            rec(md, "mixes(mlp ihc) on attn-expand output", lambda: im.mixes(xa))
            rec(md, "mixes(mlp ihc) on persistent x", lambda: im.mixes(x))
            g_ = torch.Generator().manual_seed(0)
            for it in (1, 3, 20):
                ih = Hy4IHC(
                    md,
                    torch.randn(8, 4 * HID, generator=g_) * 0.01,
                    torch.randn(8, generator=g_) * 0.02,
                    torch.tensor([0.1, 0.1]),
                    dim=HID,
                    hc_eps=1e-6,
                    norm_eps=1e-5,
                    iters=it,
                )
                r_ = ih.mixes(x)
                ttnn.synchronize_device(md)
                f = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
                P(
                    f"sinkhorn iters {it}: comb - I maxabs",
                    float((f(r_[2]).reshape(-1, 4, 4) - torch.eye(4)).abs().max()),
                    "pre",
                    f(r_[0]).flatten()[:4].tolist(),
                )
                rec(md, f"mixes sinkhorn iters={it}", lambda: ih.mixes(x))
            # EP fused with the stream padded to T=4 rows
            os.environ["DSV41_MHC_EP"] = "1"
            try:
                x4 = rep_up(md, torch.randn(4, 1, 4, HID), ttnn.float32)
                y4 = rep_up(md, torch.randn(1, 1, 4, HID))
                p4, po4, c4 = ia.mixes(x4)
                P("T=4 mixes ok")
                rec(md, "T=4 separate: expand + next mixes", lambda: im.mixes(ia.expand(y4, x4, po4, c4)))
                rec(md, "T=4 EP fused expand+next mixes", lambda: ia.expand_mixes(y4, x4, po4, c4, None, im))
                rec(md, "T=4 expand only", lambda: ia.expand(y4, x4, po4, c4))
                rec(md, "T=4 mixes only", lambda: im.mixes(x4))
                a1 = im.mixes(ia.expand(y4, x4, po4, c4))
                a2 = ia.expand_mixes(y4, x4, po4, c4, None, im)[1]
                f = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
                P("EP T=4 pre/post/comb maxabs vs separate", [float((f(u) - f(v)).abs().max()) for u, v in zip(a1, a2)])
            except Exception as e:
                import traceback

                P("EP T=4 FAILED", repr(e)[:200], traceback.format_exc()[-900:])
            os.environ["DSV41_MHC_EP"] = "0"
        if "sweep" in PROBES:
            from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

            def cfg_x(K, N, cap, ibw_max=8, sw_max=4):
                Kt, Nt = K // 32, N // 32
                pcn = -(-Nt // cap)
                n = -(-Nt // pcn)
                gx, gy = min(n, 8), -(-n // 8)
                sw = max(d for d in range(1, min(pcn, sw_max) + 1) if pcn % d == 0)
                ibw = max(d for d in range(1, ibw_max + 1) if Kt % d == 0)
                return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    in0_block_w=ibw,
                    out_subblock_h=1,
                    out_subblock_w=sw,
                    per_core_M=1,
                    per_core_N=pcn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                ), (gx, gy, pcn, ibw, sw)

            for name, K, N, shard in (
                ("fused h-proj", 6144, 3040, 3),
                ("o_proj", 2048, 6144, 2),
                ("q_b", 2048, 2048, 3),
            ):
                for dtn, dt in (("bfp8", ttnn.bfloat8_b), ("bfp4", ttnn.bfloat4_b)):
                    if shard == 2:
                        wh = torch.randn(1, 1, 8 * K, N)
                    else:
                        wh = torch.randn(1, 1, K, 8 * N)
                    w = upload(md, wh, dt, ttnn.TILE_LAYOUT, shard_dim=shard)
                    x_ = rep_up(md, torch.randn(1, 1, S, K)) if shard != 2 else rep_up(md, torch.randn(1, 1, S, K))
                    best = None
                    for cap in (16, 24, 32, 40, 48, 56, 64):
                        for ibw in (4, 8, 16):
                            for swm in (4, 8):
                                try:
                                    pc, info = cfg_x(K, N, cap, ibw, swm)
                                    us = t_us(
                                        md,
                                        lambda: ttnn.linear(x_, w, compute_kernel_config=ckc, program_config=pc),
                                        rep_=10,
                                        n=5,
                                    )
                                except Exception as e:
                                    continue
                                if best is None or us < best[0]:
                                    best = (us, cap, ibw, swm, info)
                    P(
                        f"SWEEP {name} {dtn} K{K} N{N}: best {best[0]:.1f} us cap {best[1]} ibw_max {best[2]} sw_max {best[3]} (gx,gy,pcn,ibw,sw)={best[4]}"
                    )
                    ttnn.deallocate(w)
        if "arrm" in PROBES:
            part = rep_up(md, torch.randn(1, 1, S, HID))
            mc_, ccl = a.mesh_config, a.ccl
            rec(md, "o_proj allreduce tile [S,6144] (current)", lambda: mc_.allreduce(ttnn.clone(part), ccl, axis=1))
            rm_in = ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT)
            rec(md, "allreduce ROW_MAJOR [S,6144] incl. clone", lambda: mc_.allreduce(ttnn.clone(rm_in), ccl, axis=1))
            rec(
                md,
                "to_layout RM + allreduce RM + to_layout TILE",
                lambda: ttnn.to_layout(
                    mc_.allreduce(ttnn.to_layout(ttnn.clone(part), ttnn.ROW_MAJOR_LAYOUT), ccl, axis=1),
                    ttnn.TILE_LAYOUT,
                ),
            )
            part8 = ttnn.typecast(part, ttnn.bfloat8_b)
            rec(
                md,
                "allreduce bfp8 payload (typecast in/out)",
                lambda: ttnn.typecast(
                    mc_.allreduce(ttnn.typecast(ttnn.clone(part), ttnn.bfloat8_b), ccl, axis=1), ttnn.bfloat16
                ),
            )
        if "ccl" in PROBES:
            part = rep_up(md, torch.randn(1, 1, S, HID))
            qc = rep_up(md, torch.randn(1, 1, S, 256))
            mc_, ccl = a.mesh_config, a.ccl
            rec(md, "o_proj allreduce [S,6144] (current setting)", lambda: mc_.allreduce(ttnn.clone(part), ccl, axis=1))
            rec(md, "clone only (subtract)", lambda: ttnn.clone(part))
            rec(md, "q_a allgather [S,256]->[S,2048]", lambda: mc_.allgather(qc, ccl, axis=1, dim=3))
            rec(
                md,
                "ttnn.all_reduce [S,6144] cluster_axis 1 Ring",
                lambda: ttnn.all_reduce(
                    part,
                    cluster_axis=1,
                    num_links=2,
                    topology=ttnn.Topology.Ring,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
    finally:
        json.dump(R, open(OUT, "w"), indent=1)
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
