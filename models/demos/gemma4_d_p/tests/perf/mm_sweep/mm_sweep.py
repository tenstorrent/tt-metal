"""Single-chip sweep of Gemma4-31B prefill projection matmuls (per-device shapes at CP8 / TP4).

For each (projection, M rows per device) it times the model's current config and a family of alternatives:
2D multicast, 1D in0-multicast, L1-sharded in0/out variants, ttnn's default, and matmul_auto_config_v2 (when
the build has it). Device kernel time comes from the in-process device profiler. Every config is checked
against torch and against the model config's output, so a fast-but-wrong config cannot win.

  python mm_sweep.py --m 256 1024 --proj gate down --families model 2d 1d --out out.jsonl
"""

import argparse
import json
import math
import os
import sys
import time
import traceback

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "20000")

import torch

import ttnn
from models.demos.gemma4_d_p.tt.attention.operations import projection_math_fidelity
from models.demos.gemma4_d_p.tt.matmul_config import prefill_1d_matmul_program_config, prefill_matmul_program_config

TILE = 32
FP32 = True
# (K, N, family, out memory) per device. family picks the model's compute config and grid rule.
PROJ = {
    "qkv_s": (5376, 4096, "attn", "dram"),
    "qkv_g": (5376, 4608, "attn", "dram"),
    "o_s": (2048, 5376, "attn", "dram"),
    "o_g": (4096, 5376, "attn", "dram"),
    "gate": (5376, 5376, "mlp", "l1"),
    "up": (5376, 5376, "mlp", "l1"),
    "down": (5376, 5376, "mlp", "dram"),
}
KDUR = "DEVICE KERNEL DURATION [ns]"
FPU_FLOP_PER_CYCLE_LOFI = 2 * 8 * 16 * 16  # BH: one 8x16 x 16x16 per cycle per core at LoFi
FIDELITY_PASSES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def read_durations(device):
    ttnn.ReadDeviceProfiler(device)
    data = ttnn.get_latest_programs_perf_data()
    out = []
    for _chip, programs in data.items():
        for p in programs:
            r = p.program_analyses_results.get(KDUR)
            if r is not None:
                out.append((p.program_execution_uid.runtime_id, int(r.duration), int(p.core_count)))
    out.sort()
    return out


def compute_config(device, family, m_rows, fidelity=None, fp32=True, l1acc=True):
    if fidelity is None:
        fidelity = "LoFi" if family == "mlp" else projection_math_fidelity(m_rows).name
    return fidelity, ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=l1acc,
    )


def divisors(n, lo=1, hi=None):
    hi = hi or n
    return [d for d in range(lo, hi + 1) if n % d == 0]


def subblock(pm, pn, max_tiles):
    best = None
    for w in divisors(pn):
        for h in divisors(pm):
            if h * w <= max_tiles and (best is None or h * w > best[0] * best[1] or (h * w == best[0] * best[1] and w > best[1])):
                best = (h, w)
    return best


def l1_mc(layout, cores, shape):
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(cores, shape, ttnn.ShardOrientation.ROW_MAJOR))


def rect(gx, gy):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def candidates(device, proj, m, families, x_dram):
    """Yield dicts: name, program_config, in0_mc (None = as given), out_mc, fidelity override."""
    K, N, fam, out_loc = PROJ[proj]
    grid = device.compute_with_storage_grid_size()
    mt, kt, nt = m // TILE, K // TILE, N // TILE
    out_default = ttnn.L1_MEMORY_CONFIG if out_loc == "l1" else ttnn.DRAM_MEMORY_CONFIG
    gelu = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH) if proj == "gate" else None
    maxsub = 4 if FP32 else 8  # fp32 dest halves the dest registers

    if "model" in families:
        class _W:  # noqa: N801
            padded_shape = (1, 1, K, N)
        if fam == "mlp":
            gx = max(x for x in range(1, grid.x + 1) if nt % x == 0)
            pc = prefill_1d_matmul_program_config(x_dram, _W, grid, gelu) or prefill_matmul_program_config(
                x_dram, _W, gx, grid.y, gelu, fp32_dest_acc=True
            )
        else:
            pc = prefill_1d_matmul_program_config(x_dram, _W, grid) or prefill_matmul_program_config(
                x_dram, _W, grid.x, grid.y, fp32_dest_acc=True
            )
        yield dict(name="model", pc=pc, fused_act=None if pc is not None else gelu)
    if "default" in families:
        yield dict(name="default", pc=None, default=True)
    if "auto_v2" in families and hasattr(ttnn.CONFIG, "matmul_auto_config_v2"):
        yield dict(name="auto_v2", pc=None, auto_v2=True)

    kbs = sorted({d for d in divisors(kt) if 2 <= d <= 28})
    if "2d" in families:
        for gx in sorted({x for x in range(4, grid.x + 1) if nt % x == 0} | {grid.x}):
            for gy in sorted({y for y in range(1, grid.y + 1)}):
                pm = math.ceil(mt / gy)
                if math.ceil(mt / pm) != gy:  # only grids whose rows all get work
                    continue
                pn = math.ceil(nt / gx)
                if pm > 8 or gx * gy < 32:
                    continue
                for kb in kbs:
                    h, w = subblock(pm, pn, maxsub)
                    yield dict(
                        name=f"2d_g{gx}x{gy}_m{pm}_n{pn}_k{kb}",
                        pc=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(gx, gy), in0_block_w=kb, out_subblock_h=h,
                            out_subblock_w=w, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                            fused_activation=gelu,
                        ),
                    )
    if "2d_t" in families:  # transpose_mcast: M across grid columns, N across rows
        for gy in sorted({y for y in range(4, grid.y + 1)}):
            for gx in range(1, grid.x + 1):
                pm = math.ceil(mt / gx)
                if math.ceil(mt / pm) != gx:
                    continue
                pn = math.ceil(nt / gy)
                if pm > 8 or gx * gy < 32:
                    continue
                for kb in [k for k in kbs if k in (4, 6, 7, 8, 12, 14, 16)]:
                    h, w = subblock(pm, pn, maxsub)
                    yield dict(
                        name=f"2dT_g{gx}x{gy}_m{pm}_n{pn}_k{kb}",
                        pc=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(gx, gy), in0_block_w=kb, out_subblock_h=h,
                            out_subblock_w=w, per_core_M=pm, per_core_N=pn, transpose_mcast=True,
                            fused_activation=gelu,
                        ),
                    )
    if "1d" in families:
        for pn in range(1, 9):
            ncores = math.ceil(nt / pn)
            if ncores > grid.x * grid.y or ncores < 24:
                continue
            for kb in [k for k in kbs if k <= 16]:
                h, w = subblock(mt, pn, maxsub)
                yield dict(
                    name=f"1d_in0_n{pn}_c{ncores}_k{kb}",
                    pc=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=kb, out_subblock_h=h,
                        out_subblock_w=w, per_core_M=mt, per_core_N=pn, fuse_batch=True,
                        fused_activation=gelu, mcast_in0=True,
                    ),
                )
    if "1d_in1" in families:  # M split over cores, weights multicast
        for pm in divisors(mt):
            ncores = mt // pm
            if ncores < 4:
                continue
            for kb in [k for k in kbs if k in (4, 8, 12, 14, 16)]:
                h, w = subblock(pm, nt, maxsub)
                yield dict(
                    name=f"1d_in1_m{pm}_c{ncores}_k{kb}",
                    pc=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=kb, out_subblock_h=h,
                        out_subblock_w=w, per_core_M=pm, per_core_N=nt, fuse_batch=True,
                        fused_activation=gelu, mcast_in0=False,
                    ),
                )
    if "1d_ws" in families:  # in0 width-sharded in L1 (K split over cores), out width-sharded or interleaved
        for pn in (1, 2, 3, 4):
            ncores = math.ceil(nt / pn)
            if ncores > grid.x * grid.y or nt % pn:
                continue
            for kshard in [d for d in divisors(kt) if kt // d <= ncores and d <= 28]:
                kc = kt // kshard
                in0_mc = l1_mc(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.num_cores_to_corerangeset(kc, grid, row_wise=True),
                    (m, kshard * TILE),
                )
                h, w = subblock(mt, pn, maxsub)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=kshard, out_subblock_h=h,
                    out_subblock_w=w, per_core_M=mt, per_core_N=pn, fuse_batch=True,
                    fused_activation=gelu, mcast_in0=True,
                )
                yield dict(name=f"1dws_n{pn}_ks{kshard}_c{kc}_outI", pc=pc, in0_mc=in0_mc)
                out_mc = l1_mc(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True),
                    (m, pn * TILE),
                )
                yield dict(name=f"1dws_n{pn}_ks{kshard}_c{kc}_outS", pc=pc, in0_mc=in0_mc, out_mc=out_mc)
    if "dram_sharded" in families:  # weights width-sharded across DRAM banks; in0/out width-sharded in L1
        banks = device.dram_grid_size().x
        for c in [c for c in range(1, 49) if kt % c == 0 and (kt // c) % 4 == 0 and nt % c == 0]:
            gx = max(x for x in range(1, grid.x + 1) if c % x == 0 and c // x <= grid.y)
            gy = c // gx
            for workers in (1, 2, 3):
                if nt % (banks * workers):
                    continue
                in0_mc = l1_mc(ttnn.TensorMemoryLayout.WIDTH_SHARDED, rect(gx, gy), (m, (kt // c) * TILE))
                out_mc = l1_mc(ttnn.TensorMemoryLayout.WIDTH_SHARDED, rect(gx, gy), (m, (nt // c) * TILE))
                w_mc = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(rect(banks, 1), (K, N // banks), ttnn.ShardOrientation.ROW_MAJOR),
                )
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=(kt // c) // 4, per_core_M=mt, per_core_N=nt // c, fused_activation=gelu,
                    num_workers_per_dram_bank=workers,
                )
                yield dict(name=f"dsh_c{c}_w{workers}", pc=pc, in0_mc=in0_mc, out_mc=out_mc, w_mc=w_mc)
    if "2d_bs" in families:  # in0 block-sharded over the matmul grid, out block-sharded or interleaved
        for gx in sorted({x for x in range(4, grid.x + 1) if nt % x == 0 and kt % x == 0}):
            for gy in range(1, grid.y + 1):
                if mt % gy:
                    continue
                pm, pn = mt // gy, nt // gx
                if pm > 8 or gx * gy < 32:
                    continue
                ksh = kt // gx
                in0_mc = l1_mc(ttnn.TensorMemoryLayout.BLOCK_SHARDED, rect(gx, gy), (pm * TILE, ksh * TILE))
                out_mc = l1_mc(ttnn.TensorMemoryLayout.BLOCK_SHARDED, rect(gx, gy), (pm * TILE, pn * TILE))
                for kb in [d for d in divisors(ksh) if d >= 2]:
                    h, w = subblock(pm, pn, maxsub)
                    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy), in0_block_w=kb, out_subblock_h=h,
                        out_subblock_w=w, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                        fused_activation=gelu,
                    )
                    yield dict(name=f"2dbs_g{gx}x{gy}_m{pm}_n{pn}_k{kb}_outS", pc=pc, in0_mc=in0_mc, out_mc=out_mc)
                    yield dict(name=f"2dbs_g{gx}x{gy}_m{pm}_n{pn}_k{kb}_outI", pc=pc, in0_mc=in0_mc)


def run(device, args):
    grid = device.compute_with_storage_grid_size()
    ncores_chip = grid.x * grid.y
    torch.manual_seed(0)
    out = open(args.out, "a")
    for proj in args.proj:
        K, N, fam, out_loc = PROJ[proj]
        w_t = torch.randn(1, 1, K, N) * (1.0 / math.sqrt(K))
        w = ttnn.from_torch(w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w_q = ttnn.to_torch(w).float()
        for m in args.m:
            x_t = torch.randn(1, 1, m, K)
            ref = x_t.to(torch.bfloat16).float() @ w_q
            if proj == "gate":
                ref = torch.nn.functional.gelu(ref, approximate="tanh")
            in0_loc = args.in0
            x_mc = ttnn.L1_MEMORY_CONFIG if in0_loc == "l1" else ttnn.DRAM_MEMORY_CONFIG
            x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=x_mc)
            model_out = None
            fidelity, ckc = compute_config(device, fam, m, fp32=FP32, l1acc=args.l1acc)
            flops = 2.0 * m * K * N
            wbytes = K * N * 1088 / 1024  # bfp8_b: 1 byte + shared exponent per 16
            for cand in candidates(device, proj, m, args.families, x):
                if args.filter and "_k" in cand["name"] and not any(cand["name"].endswith(f) or (f + "_") in cand["name"] for f in args.filter):
                    continue
                rec = dict(proj=proj, m=m, K=K, N=N, cfg=cand["name"], in0=in0_loc, fidelity=fidelity, tdp=args.tdp, fp32=FP32, l1acc=args.l1acc)
                try:
                    xin = x
                    reshard_ns = None
                    if cand.get("in0_mc") is not None:
                        read_durations(device)
                        xin = ttnn.to_memory_config(x, cand["in0_mc"])
                        d = read_durations(device)
                        reshard_ns = d[-1][1] if d else None
                    w_use = ttnn.to_memory_config(w, cand["w_mc"]) if cand.get("w_mc") is not None else w
                    out_mc = cand.get("out_mc") or (ttnn.L1_MEMORY_CONFIG if out_loc == "l1" else ttnn.DRAM_MEMORY_CONFIG)
                    kw = dict(memory_config=out_mc)
                    if cand.get("default"):
                        kw.update(core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x), activation="gelu_tanh" if proj == "gate" else None)
                        if fam == "attn":
                            kw.update(compute_kernel_config=ckc)
                    elif cand.get("auto_v2"):
                        kw.update(compute_kernel_config=ckc, activation="gelu_tanh" if proj == "gate" else None)
                    else:
                        kw.update(program_config=cand["pc"], compute_kernel_config=ckc)
                        if cand["pc"] is None:
                            kw.update(core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x), activation="gelu_tanh" if proj == "gate" else None)
                    if cand.get("auto_v2"):
                        ttnn.CONFIG.matmul_auto_config_v2 = True
                    try:
                        read_durations(device)
                        res = None
                        for _ in range(args.iters):
                            if res is not None:
                                res.deallocate(True)
                            res = ttnn.linear(xin, w_use, **kw)
                        ttnn.synchronize_device(device)
                        durs = read_durations(device)
                    finally:
                        if cand.get("auto_v2"):
                            ttnn.CONFIG.matmul_auto_config_v2 = False
                    unshard_ns = None
                    if res.is_sharded():
                        read_durations(device)
                        back = ttnn.sharded_to_interleaved(res, ttnn.L1_MEMORY_CONFIG if out_loc == "l1" else ttnn.DRAM_MEMORY_CONFIG)
                        d = read_durations(device)
                        unshard_ns = d[-1][1] if d else None
                        res.deallocate(True)
                        res = back
                    res_t = ttnn.to_torch(res).float()
                    res.deallocate(True)
                    if xin is not x:
                        xin.deallocate(True)
                    if w_use is not w:
                        w_use.deallocate(True)
                    if cand["name"] == "model":
                        model_out = res_t
                    mm = [d for d in durs]
                    ns = sorted(d[1] for d in mm)[len(mm) // 2] if mm else None
                    cores = max(d[2] for d in mm) if mm else None
                    peak_core = FPU_FLOP_PER_CYCLE_LOFI * args.aiclk_mhz * 1e6 / FIDELITY_PASSES[fidelity]
                    rec.update(
                        ok=True, ns=ns, n_progs=len(mm), cores=cores, reshard_ns=reshard_ns, unshard_ns=unshard_ns,
                        pcc_ref=pcc(res_t, ref), pcc_model=pcc(res_t, model_out) if model_out is not None else None,
                        util_used=(flops / (ns * 1e-9)) / (peak_core * cores) if ns and cores else None,
                        util_chip=(flops / (ns * 1e-9)) / (peak_core * ncores_chip) if ns else None,
                        gbps=wbytes / ns if ns else None,
                    )
                except Exception as e:  # noqa: BLE001
                    rec.update(ok=False, err=str(e).split("\n")[0][:300])
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if rec["ok"]:
                    print(f"{proj:6s} M={m:5d} {rec['cfg']:40s} {rec['ns']/1e3:8.1f} us cores={rec['cores']} "
                          f"util_used={rec['util_used']:.2f} util_chip={rec['util_chip']:.2f} {rec['gbps']:.0f}GB/s "
                          f"pcc_ref={rec['pcc_ref']:.5f} pcc_model={rec['pcc_model'] if rec['pcc_model'] is None else round(rec['pcc_model'], 5)}"
                          f"{'' if reshard_ns is None else f' reshard={reshard_ns/1e3:.1f}us'}", flush=True)
                else:
                    print(f"{proj:6s} M={m:5d} {rec['cfg']:40s} ERR {rec['err'][:150]}", flush=True)
            x.deallocate(True)
        w.deallocate(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--proj", nargs="+", default=list(PROJ))
    ap.add_argument("--families", nargs="+", default=["model", "default", "auto_v2", "2d", "1d"])
    ap.add_argument("--filter", nargs="*", default=None, help="substring filter on config names")
    ap.add_argument("--in0", choices=["dram", "l1"], default="dram")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--aiclk-mhz", type=float, default=1350.0)
    ap.add_argument("--tdp", default="")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--no-fp32", dest="fp32", action="store_false")
    ap.add_argument("--no-l1acc", dest="l1acc", action="store_false")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global FP32
    FP32 = args.fp32
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=0)
    try:
        run(device, args)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
