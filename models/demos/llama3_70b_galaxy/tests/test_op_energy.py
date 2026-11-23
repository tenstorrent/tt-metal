# Energy per op of the shipped DP4 x TP8 prefill layer (random weights, all 32 chips, one prompt per mesh column).
# Under the 130 W cap a long prefill runs at  time ~= sum(op energy) / 130 W  (low-power ops let the clock recover for
# the next matmul), so the op to attack is the one with the most joules, not the most time. Each op runs alone in a
# loop of ~OE_SECS seconds while a thread polls tt-smi; reported: ms per op (short loop at full clock and the long
# loop), mean chip power, min clock, and mJ per op per chip = power x long-loop time.
#   pytest models/demos/llama3_70b_galaxy/tests/test_op_energy.py -s      env: OE_M (16384), OE_SECS (8), OE_OPS
import json
import os
import subprocess
import threading
import time

import pytest
import torch
from loguru import logger

import ttnn

H, D, NQ, NKV, FF = 8192, 128, 8, 1, 3584
DRAM = ttnn.DRAM_MEMORY_CONFIG


class Poller:
    """The ONLY tt-smi reader while a test runs: a second concurrent reader (an external poll loop) made every snapshot
    fail on Nov 22 and coincided with one tray (chips 8-15) dropping off the bus. Keeps per-chip samples too
    (OE_PERCHIP_LOG=<path> appends them as JSON lines) and flags `bad` when chips stop reporting (65535) or reads keep
    failing, so callers can abort instead of hanging."""

    def __init__(self):
        self.samples = []  # (t, mean power, min aiclk, max aiclk)
        self.perchip = []  # (t, [(aiclk, power, temp), ...])
        self.stop = False
        self.bad = None
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        fails = 0
        path = os.environ.get("OE_PERCHIP_LOG")
        while not self.stop:
            try:
                out = subprocess.run(
                    ["tt-smi", "-s", "--snapshot_no_tty"], capture_output=True, timeout=20, cwd="/tmp"
                ).stdout
                tel = [x["telemetry"] for x in json.loads(out)["device_info"]]
                p = [float(x["power"]) for x in tel]
                a = [float(x["aiclk"]) for x in tel]
                if max(a) >= 65535:
                    self.bad = f"chips {[i for i, v in enumerate(a) if v >= 65535]} stopped reporting telemetry"
                    logger.error(f"OE {self.bad}")
                    return
                fails = 0
                now = time.time()
                self.samples.append((now, sum(p) / len(p), min(a), max(a)))
                chips = [(a[i], p[i], float(x["asic_temperature"])) for i, x in enumerate(tel)]
                self.perchip.append((now, chips))
                if path:
                    with open(path, "a") as fh:
                        fh.write(json.dumps([now, chips]) + "\n")
            except Exception as e:  # noqa: BLE001
                fails += 1
                logger.warning(f"OE poll failed: {str(e)[:100]}")
                if fails >= 20:
                    self.bad = f"{fails} telemetry reads in a row failed"
                    logger.error(f"OE {self.bad}")
                    return
                time.sleep(0.5)

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return float("nan"), float("nan"), float("nan"), 0
        return sum(x[1] for x in s) / len(s), min(x[2] for x in s), max(x[3] for x in s), len(s)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_op_energy(mesh_device, device_params):
    mesh = mesh_device
    secs = float(os.environ.get("OE_SECS") or 8)
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

    def ck(fid, fp32=False, approx=False):
        return ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=True
        )

    lofi, norm_ck = ck(ttnn.MathFidelity.LoFi), ck(ttnn.MathFidelity.HiFi4, True)
    sdpa_fid = getattr(ttnn.MathFidelity, os.environ.get("OE_SDPA_FID") or "HiFi2")
    sdpa_ck = ck(sdpa_fid, False, True)
    mmrs_ck = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi)
    sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
    bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    agi = [0]

    def dev(shape, dt, scale=1.0):
        return ttnn.from_torch(
            torch.randn(shape) * scale, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=dt, memory_config=DRAM
        )

    def mm(a, w, dtype, swiglu=False, kb=16):
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=kb,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=lofi,
            memory_config=DRAM,
            dtype=dtype,
            fuse_swiglu=swiglu,
        )

    def ag(x):
        i = agi[0]
        agi[0] ^= 1
        return ttnn.experimental.all_gather_async(
            input_tensor=x,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=sems[i],
            num_links=2,
            barrier_semaphore=bars[i],
            memory_config=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            num_workers_per_link=8,
        )

    def mmrs(a, w):
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            a,
            w,
            cluster_axis=0,
            scatter_dim=-2,
            topology=ttnn.Topology.Ring,
            num_links=2,
            compute_kernel_config=mmrs_ck,
            memory_config=DRAM,
        )

    def norm(x):
        return ttnn.rms_norm(x, epsilon=1e-5, memory_config=DRAM, compute_kernel_config=norm_ck, dtype=ttnn.bfloat8_b)

    tm = torch.zeros(1, 1, 32, 32)
    tm[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    tm[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
    trans = ttnn.from_torch(tm, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    poll = Poller()
    rows = []
    for M in [int(m) for m in (os.environ.get("OE_M") or "16384").split(",")]:
        w = {
            "wqkv": dev((1, 1, H, (NQ + 2 * NKV) * D), ttnn.bfloat8_b, 0.02),
            "wo": dev((NQ * D, H), ttnn.bfloat8_b, 0.02),
            "w13": dev((1, 1, H, 2 * FF), ttnn.bfloat4_b, 0.02),
            "w2": dev((FF, H), ttnn.bfloat8_b, 0.02),
        }
        x = dev((1, 1, M // 8, H), ttnn.bfloat16)
        cos, sin = dev((1, 1, M, D), ttnn.bfloat16), dev((1, 1, M, D), ttnn.bfloat16)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=512, exp_approx_mode=True
        )
        # inputs of each op, made once
        h = norm(x)
        hg = ag(h)
        qkv = mm(hg, w["wqkv"], ttnn.bfloat16)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NQ, num_kv_heads=NKV, transpose_k_heads=False, memory_config=DRAM
        )
        qd = {"dtype": ttnn.bfloat8_b}
        q2 = ttnn.experimental.rotary_embedding_llama(q, cos, sin, trans, is_decode_mode=False, **qd)
        k8 = ttnn.experimental.rotary_embedding_llama(k, cos, sin, trans, is_decode_mode=False, **qd)
        v8 = ttnn.typecast(v, ttnn.bfloat8_b)
        o = ttnn.transformer.scaled_dot_product_attention(
            q2, k8, v8, is_causal=True, program_config=sdpa_cfg, compute_kernel_config=sdpa_ck, memory_config=DRAM
        )
        a8 = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
        if a8.dtype != ttnn.bfloat8_b:
            a8 = ttnn.typecast(a8, ttnn.bfloat8_b)
        r1 = mmrs(a8, w["wo"])
        r1b = ttnn.typecast(r1, ttnn.bfloat16) if r1.dtype != ttnn.bfloat16 else r1
        f = mm(hg, w["w13"], ttnn.bfloat8_b, swiglu=True, kb=32)

        wb4 = {}
        if os.environ.get("OE_LAYER_VARIANTS"):
            wb4 = {
                "wqkv": dev((1, 1, H, (NQ + 2 * NKV) * D), ttnn.bfloat4_b, 0.02),
                "wo": dev((NQ * D, H), ttnn.bfloat4_b, 0.02),
                "w2": dev((FF, H), ttnn.bfloat4_b, 0.02),
            }
        sdpa_lofi_ck = ck(ttnn.MathFidelity.LoFi, False, True)
        wo_full = None
        if "woa2a" in (os.environ.get("OE_LAYER_VARIANTS") or ""):
            wo_full = dev((1, 1, H, H), ttnn.bfloat8_b, 0.02)

        w4 = {}
        if "split" in (os.environ.get("OE_LAYER_VARIANTS") or ""):
            w4 = {"wo": dev((1, 1, NQ * D, H), ttnn.bfloat8_b, 0.02), "w2": dev((1, 1, FF, H), ttnn.bfloat8_b, 0.02)}

        def mm_rs(a_, wt):
            p_ = mm(a_, wt, ttnn.bfloat8_b)
            r_ = ttnn.reduce_scatter(
                p_, 2, cluster_axis=0, topology=ttnn.Topology.Ring, num_links=2, memory_config=DRAM
            )
            ttnn.deallocate(p_)
            return r_

        def wo_a2a(a_):
            g_ = ttnn.experimental.all_to_all_async_generic(
                a_, 3, 2, num_links=2, memory_config=DRAM, topology=ttnn.Topology.Ring, cluster_axis=0
            )
            y_ = mm(g_, wo_full, ttnn.bfloat8_b)
            ttnn.deallocate(g_)
            return y_

        def layer(v=""):
            WQ = wb4["wqkv"] if "qkvb4" in v else w["wqkv"]
            WO = wb4["wo"] if "wob4" in v else w["wo"]
            W2 = wb4["w2"] if "w2b4" in v else w["w2"]
            sck = sdpa_lofi_ck if "sdpalofi" in v else sdpa_ck
            qdt = ttnn.bfloat8_b if "qkv8" in v else ttnn.bfloat16
            h_ = norm(x)
            hg_ = ag(h_)
            ttnn.deallocate(h_)
            qkv_ = mm(hg_, WQ, qdt)
            ttnn.deallocate(hg_)
            q_, k_, v_ = ttnn.experimental.nlp_create_qkv_heads(
                qkv_, num_heads=NQ, num_kv_heads=NKV, transpose_k_heads=False, memory_config=DRAM
            )
            ttnn.deallocate(qkv_)
            q2_ = ttnn.experimental.rotary_embedding_llama(q_, cos, sin, trans, is_decode_mode=False, **qd)
            k8_ = ttnn.experimental.rotary_embedding_llama(k_, cos, sin, trans, is_decode_mode=False, **qd)
            v8_ = v_ if v_.dtype == ttnn.bfloat8_b else ttnn.typecast(v_, ttnn.bfloat8_b)
            ttnn.deallocate(q_), ttnn.deallocate(k_)
            if v8_ is not v_:
                ttnn.deallocate(v_)
            o_ = ttnn.transformer.scaled_dot_product_attention(
                q2_, k8_, v8_, is_causal=True, program_config=sdpa_cfg, compute_kernel_config=sck, memory_config=DRAM
            )
            ttnn.deallocate(q2_), ttnn.deallocate(k8_), ttnn.deallocate(v8_)
            a_ = ttnn.experimental.nlp_concat_heads(o_, memory_config=DRAM)
            ttnn.deallocate(o_)
            if a_.dtype != ttnn.bfloat8_b:
                b_ = ttnn.typecast(a_, ttnn.bfloat8_b)
                ttnn.deallocate(a_)
                a_ = b_
            r1_ = wo_a2a(a_) if "woa2a" in v else (mm_rs(a_, w4["wo"]) if "splitwo" in v else mmrs(a_, WO))
            ttnn.deallocate(a_)
            x1_ = ttnn.add(x, r1_, dtype=ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(r1_)
            h2_ = norm(x1_)
            hg2_ = ag(h2_)
            ttnn.deallocate(h2_)
            f_ = mm(hg2_, w["w13"], ttnn.bfloat8_b, swiglu=True, kb=32)
            ttnn.deallocate(hg2_)
            r2_ = mm_rs(f_, w4["w2"]) if "splitff2" in v else mmrs(f_, W2)
            ttnn.deallocate(f_)
            x2_ = ttnn.add(x1_, r2_, dtype=ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(r2_), ttnn.deallocate(x1_)
            return x2_

        ops = {
            "norm": lambda: norm(x),
            "ag": lambda: ag(h),
            "qkv": lambda: mm(hg, w["wqkv"], ttnn.bfloat16),
            "heads": lambda: ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=NQ, num_kv_heads=NKV, transpose_k_heads=False, memory_config=DRAM
            ),
            "rope_q": lambda: ttnn.experimental.rotary_embedding_llama(q, cos, sin, trans, is_decode_mode=False, **qd),
            "rope_k": lambda: ttnn.experimental.rotary_embedding_llama(k, cos, sin, trans, is_decode_mode=False, **qd),
            "typecast_v": lambda: ttnn.typecast(v, ttnn.bfloat8_b),
            "sdpa": lambda: ttnn.transformer.scaled_dot_product_attention(
                q2, k8, v8, is_causal=True, program_config=sdpa_cfg, compute_kernel_config=sdpa_ck, memory_config=DRAM
            ),
            "concat": lambda: ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM),
            "wo_mmrs": lambda: mmrs(a8, w["wo"]),
            "add": lambda: ttnn.add(x, r1, dtype=ttnn.bfloat16, memory_config=DRAM),
            # rms_norm with the residual folded in: norm(x + r1) in one pass (no x + r1 output)
            "norm_res": lambda: ttnn.rms_norm(
                x,
                epsilon=1e-5,
                residual_input_tensor=r1b,
                memory_config=DRAM,
                compute_kernel_config=norm_ck,
                dtype=ttnn.bfloat8_b,
            ),
            "ff13": lambda: mm(hg, w["w13"], ttnn.bfloat8_b, swiglu=True, kb=32),
            "ff2_mmrs": lambda: mmrs(f, w["w2"]),
            "layer": layer,
        }
        # variants: OE_SDPA_VARIANTS "q x k : fidelity" list, e.g. 256x512:HiFi2,128x512:LoFi -> ops sdpa_256x512_HiFi2
        for vs in filter(None, (os.environ.get("OE_SDPA_VARIANTS") or "").split(",")):
            chunks, fid = (vs.split(":") + ["HiFi2"])[:2]
            qc, kc = (int(c) for c in chunks.split("x"))
            vcfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=True
            )
            vck = ck(getattr(ttnn.MathFidelity, fid), False, True)
            ops[f"sdpa_{qc}x{kc}_{fid}"] = (
                lambda c_, k_: lambda: ttnn.transformer.scaled_dot_product_attention(
                    q2, k8, v8, is_causal=True, program_config=c_, compute_kernel_config=k_, memory_config=DRAM
                )
            )(vcfg, vck)

        # OE_MM_VARIANTS "op:Mb.Kb.Nb.sbh.sbw" list (op = ff13 or qkv), e.g. ff13:16.16.8.2.4
        def mm_v(a, wt, dtype, swiglu, mb, kb, nb, sh, sw):
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=grid,
            )
            return ttnn.experimental.minimal_matmul(
                input_tensor=a,
                weight_tensor=wt,
                config=cfg,
                compute_kernel_config=lofi,
                memory_config=DRAM,
                dtype=dtype,
                fuse_swiglu=swiglu,
            )

        for vs in filter(None, (os.environ.get("OE_MM_VARIANTS") or "").split(",")):
            op, blk = vs.split(":")
            p_ = [int(t) for t in blk.split(".")]
            if op == "ff13":
                ops[f"ff13_{blk}"] = (lambda p: lambda: mm_v(hg, w["w13"], ttnn.bfloat8_b, True, *p))(p_)
            else:
                ops[f"qkv_{blk}"] = (lambda p: lambda: mm_v(hg, w["wqkv"], ttnn.bfloat16, False, *p))(p_)
        # unfused alternatives to matmul_reduce_scatter_prefill: plain matmul to the full partial [M, 8192], then a
        # separate reduce-scatter over the column (OE_SPLIT=1)
        if os.environ.get("OE_SPLIT") == "1":
            w2_4 = dev((1, 1, FF, H), ttnn.bfloat8_b, 0.02)
            w2_4b = dev((1, 1, FF, H), ttnn.bfloat4_b, 0.02)
            wo_4 = dev((1, 1, NQ * D, H), ttnn.bfloat8_b, 0.02)
            wo_4b = dev((1, 1, NQ * D, H), ttnn.bfloat4_b, 0.02)
            part = mm(f, w2_4, ttnn.bfloat8_b)
            ops["ff2_mm"] = lambda: mm(f, w2_4, ttnn.bfloat8_b)
            ops["ff2_mm_b4"] = lambda: mm(f, w2_4b, ttnn.bfloat8_b)
            ops["wo_mm"] = lambda: mm(a8, wo_4, ttnn.bfloat8_b)
            ops["wo_mm_b4"] = lambda: mm(a8, wo_4b, ttnn.bfloat8_b)
            ops["rs"] = lambda: ttnn.reduce_scatter(
                part, 2, cluster_axis=0, topology=ttnn.Topology.Ring, num_links=2, memory_config=DRAM
            )
            ops["ff2_mmrs_b4"] = lambda: mmrs(f, ttnn.reshape(w2_4b, ttnn.Shape((FF, H))))
        # OE_LAYER_VARIANTS: '+'-joined flags per variant (w2b4, wob4, qkvb4, sdpalofi, qkv8), e.g. w2b4,wob4+w2b4
        for vs in filter(None, (os.environ.get("OE_LAYER_VARIANTS") or "").split(",")):
            ops[f"layer_{vs}"] = (lambda v_: lambda: layer(v_))(vs)
        default = ",".join(["idle"] + [k_ for k_ in ops if not k_.startswith(("sdpa_", "ff13_", "qkv_", "layer_"))])
        sel = [s for s in (os.environ.get("OE_OPS") or default).split(",") if s]
        sel += [k_ for k_ in ops if k_.startswith(("sdpa_", "ff13_", "qkv_", "layer_")) and k_ not in sel]

        def free(out):
            for t in out if isinstance(out, (list, tuple)) else [out]:
                ttnn.deallocate(t)

        for name in sel:
            if poll.bad:
                pytest.fail(f"telemetry broke: {poll.bad}")
            if name == "idle":
                time.sleep(3)
                t0 = time.time()
                time.sleep(secs)
                p, amin, amax, n = poll.window(t0 + 1, time.time())
                rows.append((M, name, 0, 0, p, amin, amax, 0))
                logger.info(f"OE M={M} idle: power {p:.1f} W, aiclk {amin:.0f}-{amax:.0f} ({n} samples)")
                continue
            fn = ops[name]
            try:
                free(fn())
                ttnn.synchronize_device(mesh)
            except RuntimeError as e:
                logger.warning(f"OE M={M} {name} ERROR {str(e)[:200]}")
                continue
            time.sleep(4)  # cool down
            t0 = time.perf_counter()
            for _ in range(10):
                free(fn())
            ttnn.synchronize_device(mesh)
            ms_fast = (time.perf_counter() - t0) / 10 * 1e3
            n = max(20, int(secs * 1e3 / ms_fast))
            w0 = time.time()
            t0 = time.perf_counter()
            for _ in range(n):
                free(fn())
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / n * 1e3
            # skip the first 2 s (clock still settling) and the dispatch tail
            p, amin, amax, ns = poll.window(w0 + 2.0, time.time() - 0.2)
            mj = p * ms
            rows.append((M, name, ms_fast, ms, p, amin, amax, mj))
            logger.info(
                f"OE M={M} {name}: {ms_fast:.3f} ms cool, {ms:.3f} ms sustained x{n}, power {p:.1f} W, aiclk "
                f"{amin:.0f}-{amax:.0f}, {mj:.1f} mJ per op per chip ({ns} samples)"
            )
    poll.stop = True
    logger.info("OE SUMMARY M op ms_cool ms_sust W aiclk_min mJ")
    for M, name, mf, ms, p, amin, amax, mj in rows:
        logger.info(f"OE SUMMARY {M} {name} {mf:.3f} {ms:.3f} {p:.1f} {amin:.0f} {mj:.1f}")
