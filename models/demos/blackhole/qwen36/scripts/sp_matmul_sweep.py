#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark sweep of ttnn.linear program configs for Qwen3.5-2B SP prefill
matmul shapes at M=1024, on one Blackhole die (1x1 mesh, no fabric). Per shape/fidelity:
auto baseline, then a 2D-grid x in0_block_w x subblock search, a 1D-on-full-grid sweep,
and a transpose_mcast retry of the best 2D config -- timed via trace-capture replay.

Run: env per SP_PREFILL_HANDOFF.md sec 6, then
  python sp_matmul_sweep.py [--out-dir DIR]
Writes results.csv / results.md to --out-dir (default: cwd).
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

import ttnn

GDN_CONV1D_L1_SMALL_SIZE = 24576  # models/demos/blackhole/qwen36/tt/model_config.py

SHAPES = [
    # name, M, K, N
    ("in_proj_gdn", 1024, 2048, 8224),
    ("in_proj_attn", 1024, 2048, 5120),
    ("mlp_gate_up", 1024, 2048, 6144),
    ("mlp_down", 1024, 6144, 2048),
    ("out_proj", 1024, 2048, 2048),
]

TODAY_US = {
    "in_proj_gdn": "225",
    "in_proj_attn": "154",
    "mlp_gate_up": "91 (up) / 179 (gate)",
    "mlp_down": "135",
    "out_proj": "66-101",
}

FIDELITIES = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2]

TILE = 32
N_TRACE_CALLS = 10
N_TIMED_EXEC = 5

# Overall wall-clock safety net (script is run under `timeout 1800`). Measured empirically at
# ~0.2s/config in a smoke test, so the full filtered cross product below (a few hundred configs
# total) comfortably fits; this is just a guard against a pathological hang.
GLOBAL_DEADLINE_SECONDS = 1500

# All (gx, gy) pairs the task asks for: gy=8 (pm=4 exact) and gy=10 (pm=4, pads M 32->40).
GRIDS_2D = [(11, 8), (10, 8), (8, 8), (11, 10), (10, 10), (8, 10)]
BW_CANDIDATES = [1, 2, 4, 8]
SUBBLOCKS = [(1, 1), (2, 2), (1, 4), (2, 4), (4, 2), (1, 8)]
ONED_BW_CANDIDATES = [2, 4, 8]
ONED_SW_CANDIDATES = [1, 2]
PER_SHAPE_FIDELITY_CAP = 60  # soft cap; the divisibility filters below usually land well under this
TRANSPOSE_RETRY_TOP_N = 2


def ceil_div(a, b):
    return (a + b - 1) // b


def make_ckc(fidelity):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        math_approx_mode=False,
    )


def config_to_str(cfg):
    if cfg is None:
        return "auto"
    if cfg["kind"] == "2d":
        return (
            f"2D grid=({cfg['gx']},{cfg['gy']}) bw={cfg['bw']} sub=({cfg['sh']},{cfg['sw']}) "
            f"pm={cfg['pm']} pn={cfg['pn']} tmc={cfg['transpose_mcast']}" + (" [pads_M]" if cfg.get("pads_m") else "")
        )
    else:
        return f"1D grid=({cfg['gx']},{cfg['gy']}) bw={cfg['bw']} sw={cfg['sw']} pn={cfg['pn']}" + (
            " [pads_N]" if cfg.get("pads_n") else ""
        )


def core_count(cfg):
    if cfg is None:
        return None
    return cfg["gx"] * cfg["gy"]


def make_progcfg(cfg):
    if cfg is None:
        return None
    if cfg["kind"] == "2d":
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cfg["gx"], cfg["gy"]),
            in0_block_w=cfg["bw"],
            out_subblock_h=cfg["sh"],
            out_subblock_w=cfg["sw"],
            per_core_M=cfg["pm"],
            per_core_N=cfg["pn"],
            transpose_mcast=cfg["transpose_mcast"],
            fused_activation=None,
            fuse_batch=True,
        )
    else:
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cfg["gx"], cfg["gy"]),
            in0_block_w=cfg["bw"],
            out_subblock_h=1,
            out_subblock_w=cfg["sw"],
            per_core_M=32,
            per_core_N=cfg["pn"],
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )


def time_config(mesh_device, x, w, ckc, progcfg, memory_config):
    """Returns (us_per_matmul, method, error). error is None on success."""
    try:
        out = ttnn.linear(
            x,
            w,
            compute_kernel_config=ckc,
            program_config=progcfg,
            memory_config=memory_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.synchronize_device(mesh_device)
        out.deallocate(force=True)
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}".splitlines()[0][:300]

    try:
        outs = []
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(N_TRACE_CALLS):
            o = ttnn.linear(
                x,
                w,
                compute_kernel_config=ckc,
                program_config=progcfg,
                memory_config=memory_config,
                dtype=ttnn.bfloat16,
            )
            outs.append(o)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        samples = []
        for _ in range(N_TIMED_EXEC):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1e6)
        us_per_mm = statistics.median(samples) / N_TRACE_CALLS

        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            o.deallocate(force=True)
        return us_per_mm, "trace", None
    except Exception as e:
        err = f"TRACE_FALLBACK {type(e).__name__}: {e}".splitlines()[0][:300]
        try:
            samples = []
            for _ in range(20):
                t0 = time.perf_counter()
                o = ttnn.linear(
                    x,
                    w,
                    compute_kernel_config=ckc,
                    program_config=progcfg,
                    memory_config=memory_config,
                    dtype=ttnn.bfloat16,
                )
                ttnn.synchronize_device(mesh_device)
                t1 = time.perf_counter()
                samples.append((t1 - t0) * 1e6)
                o.deallocate(force=True)
            return statistics.median(samples), f"untraced ({err})", None
        except Exception as e2:
            return None, None, f"{type(e2).__name__}: {e2}".splitlines()[0][:300]


def run_one(mesh_device, x, w, ckc, cfg, rows, name, fid_name, shape_results, memory_config):
    cfg_str = config_to_str(cfg)
    try:
        progcfg = make_progcfg(cfg)
    except Exception as e:
        err = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        rows.append([name, fid_name, cfg_str, core_count(cfg), "", "", "ERROR", err])
        print(f"    {cfg_str}: BUILD ERROR {err}")
        return None, err
    us, method, err = time_config(mesh_device, x, w, ckc, progcfg, memory_config)
    status = "OK" if err is None else "ERROR"
    rows.append(
        [name, fid_name, cfg_str, core_count(cfg), method or "", us if us is not None else "", status, err or ""]
    )
    if err is not None:
        print(f"    {cfg_str}: ERROR {err}")
    else:
        print(f"    {cfg_str}: {us:.1f} us ({method})")
        shape_results.append((cfg_str, core_count(cfg), method, us, cfg))
    return us, err


def sweep_shape_fidelity(mesh_device, x, w, ckc, name, M, K, N, fid_name, full_grid, rows, deadline):
    M_tiles, K_tiles, N_tiles = M // TILE, K // TILE, N // TILE
    shape_results = []
    n_tried = 0

    # 0. auto
    run_one(mesh_device, x, w, ckc, None, rows, name, fid_name, shape_results, ttnn.DRAM_MEMORY_CONFIG)

    # 2D: full filtered cross product over grid x in0_block_w x subblock.
    for gx, gy in GRIDS_2D:
        if time.time() > deadline or n_tried >= PER_SHAPE_FIDELITY_CAP:
            print("    !! cap/deadline hit, stopping 2D sweep early")
            break
        pm = ceil_div(M_tiles, gy)
        pn = ceil_div(N_tiles, gx)
        pads_m = pm * gy != M_tiles
        for bw in BW_CANDIDATES:
            if K_tiles % bw != 0:
                continue
            for sh, sw in SUBBLOCKS:
                if sh > pm or sw > pn or sh * sw > 8:
                    continue
                if pm % sh != 0 or pn % sw != 0:
                    continue  # out_block_{h,w} (== per_core_{M,N} here) must divide the subblock
                if n_tried >= PER_SHAPE_FIDELITY_CAP:
                    break
                cfg = dict(
                    kind="2d", gx=gx, gy=gy, bw=bw, sh=sh, sw=sw, pm=pm, pn=pn, transpose_mcast=False, pads_m=pads_m
                )
                run_one(mesh_device, x, w, ckc, cfg, rows, name, fid_name, shape_results, ttnn.DRAM_MEMORY_CONFIG)
                n_tried += 1

    # 1D on full grid
    gx, gy = full_grid
    total_cores = gx * gy
    pn = ceil_div(N_tiles, total_cores)
    pads_n = pn * total_cores != N_tiles
    if time.time() <= deadline:
        for bw in ONED_BW_CANDIDATES:
            if K_tiles % bw != 0:
                continue
            for sw in ONED_SW_CANDIDATES:
                if sw > pn or pn % sw != 0:
                    continue
                cfg = dict(kind="1d", gx=gx, gy=gy, bw=bw, sw=sw, pn=pn, pads_n=pads_n)
                run_one(mesh_device, x, w, ckc, cfg, rows, name, fid_name, shape_results, ttnn.DRAM_MEMORY_CONFIG)

    # transpose_mcast=True retry of the best N 2D configs found so far
    best_2d = sorted((r for r in shape_results if r[4] is not None and r[4]["kind"] == "2d"), key=lambda r: r[3])
    for r in best_2d[:TRANSPOSE_RETRY_TOP_N]:
        if time.time() > deadline:
            break
        cfg2 = dict(r[4])
        cfg2["transpose_mcast"] = True
        run_one(mesh_device, x, w, ckc, cfg2, rows, name, fid_name, shape_results, ttnn.DRAM_MEMORY_CONFIG)

    print(f"    ({len(shape_results)} successful / {n_tried} 2D attempts this shape/fidelity)")
    return shape_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path.cwd(), help="directory for results.csv/results.md")
    cli_args = ap.parse_args()
    csv_path = cli_args.out_dir / "results.csv"
    md_path = cli_args.out_dir / "results.md"

    start = time.time()
    deadline = start + GLOBAL_DEADLINE_SECONDS

    print("Opening mesh device...")
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
        trace_region_size=64 * 1024 * 1024,
    )
    grid = mesh_device.compute_with_storage_grid_size()
    print(f"compute_with_storage_grid_size = {grid} ({grid.x}x{grid.y} = {grid.x * grid.y} cores)")
    full_grid = (grid.x, grid.y)

    rows = []

    try:
        for name, M, K, N in SHAPES:
            if time.time() > deadline:
                print(f"!! global deadline hit before shape {name}, skipping remaining shapes")
                break
            print(f"\n=== shape {name}: {M}x{K} -> {N} ===")
            torch.manual_seed(0)
            x_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
            w_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)

            x = ttnn.from_torch(
                x_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            w = ttnn.from_torch(
                w_torch,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            for fidelity in FIDELITIES:
                fid_name = "LoFi" if fidelity == ttnn.MathFidelity.LoFi else "HiFi2"
                print(f"  -- fidelity {fid_name} --")
                ckc = make_ckc(fidelity)
                sweep_shape_fidelity(mesh_device, x, w, ckc, name, M, K, N, fid_name, full_grid, rows, deadline)

            x.deallocate(force=True)
            w.deallocate(force=True)
    finally:
        write_csv(rows, csv_path)
        write_md(rows, full_grid, md_path)
        ttnn.close_mesh_device(mesh_device)
        print(f"\nDone in {time.time() - start:.1f}s. Device closed.")


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["shape", "fidelity", "config", "cores", "method", "us", "status", "error"])
        wtr.writerows(rows)
    print(f"Wrote {csv_path}")


def write_md(rows, full_grid, md_path):
    lines = ["# Matmul program config sweep results", ""]
    lines.append(f"Grid: {full_grid[0]}x{full_grid[1]} = {full_grid[0] * full_grid[1]} cores")
    lines.append("")
    for name, M, K, N in SHAPES:
        lines.append(f"## {name}: {M}x{K} -> {N}")
        lines.append("")
        lines.append(f"Today's model achieves: {TODAY_US[name]} us")
        lines.append("")
        for fid_name in ("LoFi", "HiFi2"):
            sub = [r for r in rows if r[0] == name and r[1] == fid_name and r[6] == "OK" and r[5] != ""]
            errs = [r for r in rows if r[0] == name and r[1] == fid_name and r[6] == "ERROR"]
            lines.append(f"### {fid_name}")
            lines.append("")
            if not sub:
                lines.append("No successful configs.")
                lines.append("")
                continue
            auto_rows = [r for r in sub if r[2] == "auto"]
            auto_us = float(auto_rows[0][5]) if auto_rows else None
            best = sorted(sub, key=lambda r: float(r[5]))
            top5 = best[:5]
            best_us = float(top5[0][5])
            flops = 2 * M * K * N
            tflops = flops / (best_us * 1e-6) / 1e12
            lines.append(f"- Auto config: {auto_us:.1f} us" if auto_us else "- Auto config: FAILED")
            lines.append(f"- Best config: {best_us:.1f} us -- `{top5[0][2]}` (cores={top5[0][3]}, method={top5[0][4]})")
            if auto_us:
                lines.append(f"- Speedup over auto: {auto_us / best_us:.2f}x")
            else:
                lines.append("- Speedup over auto: n/a (auto failed)")
            lines.append(f"- Achieved TFLOPS at best config: {tflops:.2f}")
            lines.append("")
            lines.append("Top 5 configs:")
            lines.append("")
            lines.append("| rank | us | cores | config |")
            lines.append("|---|---|---|---|")
            for i, r in enumerate(top5, 1):
                lines.append(f"| {i} | {float(r[5]):.1f} | {r[3]} | `{r[2]}` |")
            lines.append("")
            if errs:
                lines.append(f"Errors ({len(errs)}), first few:")
                lines.append("")
                for r in errs[:5]:
                    lines.append(f"- `{r[2]}`: {r[7]}")
                lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
