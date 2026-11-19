#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""One shape, one op: device-kernel time of small_m_matmul vs the tt_dit production matmul paths.

Same measurement method as prod_sweep_worker.py (device-profiler CSV demuxed by run-host-id, one process per
measurement because the CSV is only written on device close), but for several ops so small_m_matmul can be
compared against what tt_dit runs today on the same shape:

  small   ttnn.experimental.small_m_matmul, config=None (production picker), DRAM width-sharded weight
  mm      ttnn.experimental.minimal_matmul with models/tt_dit/utils/matmul.get_matmul_config on the grid
          (tables, else the 8x8x8 default -- exactly what tt_dit's 2D path resolves), DRAM interleaved weight
  mmh     as mm, but get_matmul_config(use_heuristic=True) -- the heuristic blocking for unswept shapes
  lin1d   ttnn.linear with get_1d_matmul_config (tt_dit's _linear path for <= 128 rows), DRAM interleaved weight

BENCH_CHUNKS=C (env) measures the column-split siblings instead (small_m_matmul_split / minimal_matmul_split,
the op ColParallelLinear runs when `chunks` is set); small, mm and mmh only.

All bf16 in/out, HiFi2, fp32 dest accumulation (small_m_matmul's numerics are fixed to that).

argv: OP M K N [nblocks] [grid]   grid = "11x10" (default; tt_dit clamps BH Galaxy to 11x10) or e.g. "12x10"
Emits one line: BENCH_JSON {...}. Two timings per iteration:
  span_us     max(ZONE_END) - min(ZONE_START) over every core/RISC = the ops report's DEVICE KERNEL DURATION
  maxcore_us  longest single core/RISC kernel zone (prod_sweep_worker's metric)
Cycles are converted with the CHIP_FREQ the profiler recorded in the CSV header.
"""
import csv
import json
import os
import statistics
import sys

import torch

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
sys.path.insert(0, ROOT)

import ttnn  # noqa: E402
from models.tt_dit.utils.matmul import get_1d_matmul_config, get_matmul_config  # noqa: E402

CSV_PATH = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
WARMUP, TIMED = 2, 12

OP = sys.argv[1]
M, K, N = (int(x) for x in sys.argv[2:5])
NBLOCKS = int(sys.argv[5]) if len(sys.argv) > 5 else 2
GRID = tuple(int(v) for v in (sys.argv[6] if len(sys.argv) > 6 else "11x10").split("x"))
CHUNKS = int(os.environ.get("BENCH_CHUNKS", "1"))


def chip_freq_mhz():
    try:
        return float(open(CSV_PATH).readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    except Exception:  # noqa: BLE001
        return 1350.0


def parse_runids(freq_mhz):
    if not os.path.exists(CSV_PATH):
        return {}
    raw = {}
    for row in csv.reader(open(CSV_PATH)):
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        raw.setdefault(row[7].strip(), {}).setdefault((row[1], row[2], row[3]), []).append(
            (row[11].strip(), int(row[5]))
        )
    out = {}
    for runid, cr in raw.items():
        durs, start, end = {}, None, None
        for kk, lst in cr.items():
            st = None
            for t, c in lst:
                if t == "ZONE_START":
                    st = c
                    start = c if start is None else min(start, c)
                elif t == "ZONE_END" and st is not None:
                    durs[kk] = c - st
                    end = c if end is None else max(end, c)
                    st = None
        if durs:
            out[runid] = {
                "maxcore": max(durs.values()) / freq_mhz,
                "span": (end - start) / freq_mhz,
                "start": start,
            }
    return out


def pcc(a, b):
    x, y = a.flatten().double(), b.flatten().double()
    x = x - x.mean()
    y = y - y.mean()
    d = (x.norm() * y.norm()).item()
    return 1.0 if d == 0 else torch.dot(x, y).item() / d


def main():
    try:
        os.remove(CSV_PATH)
    except OSError:
        pass
    res = {
        "op": OP,
        "M": M,
        "K": K,
        "N": N,
        "chunks": CHUNKS,
        "outcome": "runtime",
        "err": "",
        "grid": f"{GRID[0]}x{GRID[1]}",
    }
    labels = []
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    res["board"] = f"{g.x}x{g.y}"
    ran = False
    try:
        core_grid = ttnn.CoreCoord(*GRID)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        t0 = torch.randn(1, 1, M, K)
        t1 = torch.randn(1, 1, K, N)
        ref = (t0.bfloat16().float() @ t1.bfloat16().float())[0, 0]
        a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        if OP == "small":
            wc = ttnn.create_small_m_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
            a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wc)
            res["config"] = "picker"
            if CHUNKS > 1:
                run = lambda: ttnn.experimental.small_m_matmul_split(a0, a1, chunks=CHUNKS)  # noqa: E731
            else:
                run = lambda: ttnn.experimental.small_m_matmul(a0, a1)  # noqa: E731
        else:
            a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            if OP in ("mm", "mmh"):
                cfg = get_matmul_config(M, K, N, core_grid, use_heuristic=OP == "mmh")
                res["config"] = (
                    f"MKN=({cfg.M_block_size},{cfg.K_block_size},{cfg.N_block_size}) "
                    f"sb=({cfg.subblock_h},{cfg.subblock_w}) {GRID[0]}x{GRID[1]}"
                )
                if CHUNKS > 1:
                    run = lambda: ttnn.experimental.minimal_matmul_split(  # noqa: E731
                        a0, a1, chunks=CHUNKS, dim=-1, config=cfg, compute_kernel_config=ckc
                    )
                else:
                    run = lambda: ttnn.experimental.minimal_matmul(  # noqa: E731
                        input_tensor=a0, weight_tensor=a1, config=cfg, compute_kernel_config=ckc
                    )
            elif OP == "lin1d" and CHUNKS == 1:
                pc = get_1d_matmul_config(M, K, N, core_grid)
                res["config"] = (
                    f"1D in0_bw={getattr(pc, 'in0_block_w', '?')} pcN={getattr(pc, 'per_core_N', '?')} "
                    f"sbw={getattr(pc, 'out_subblock_w', '?')} {GRID[0]}x{GRID[1]}"
                )
                run = lambda: ttnn.linear(a0, a1, program_config=pc, compute_kernel_config=ckc)  # noqa: E731
            else:
                raise ValueError(f"unknown op {OP}")

        def call(label):
            out = run()
            labels.append(label)
            return out

        o = call("pcc")
        ttnn.synchronize_device(dev)
        outs = o if isinstance(o, (list, tuple)) else [o]
        got = torch.cat([ttnn.to_torch(ttnn.from_device(t)) for t in outs], dim=-1)[0, 0][:M, :N]
        res["pcc"] = round(pcc(ref, got), 6)
        res["n_nonfinite"] = int((~torch.isfinite(got)).sum())
        for b in range(NBLOCKS):
            for _ in range(WARMUP):
                call(f"b{b}_w")
            for _ in range(TIMED):
                call(f"b{b}_t")
            ttnn.synchronize_device(dev)
        ran = True
    except Exception as e:  # noqa: BLE001
        res["err"] = str(e)[:400]
    finally:
        ttnn.close_device(dev)

    if not ran:
        print("BENCH_JSON " + json.dumps(res), flush=True)
        return
    freq = chip_freq_mhz()
    res["chip_freq_mhz"] = freq
    rid = parse_runids(freq)
    order = sorted(rid, key=lambda r: rid[r]["start"])
    if len(order) != len(labels):
        res["err"] = f"demux misalign {len(order)} vs {len(labels)}"
        print("BENCH_JSON " + json.dumps(res), flush=True)
        return
    timed = [rid[i] for i, lab in zip(order, labels) if lab.endswith("_t")]
    for key in ("span", "maxcore"):
        vals = [t[key] for t in timed]
        res[f"{key}_us"] = round(statistics.median(vals), 3)
        res[f"{key}_min_us"] = round(min(vals), 3)
        res[f"{key}_max_us"] = round(max(vals), 3)
    res["n_iters"] = len(timed)
    res["weight_mb"] = round(K * N * 2 / 1e6, 3)
    res["dram_gbps"] = round(K * N * 2 / (res["span_us"] * 1e-6) / 1e9, 1)
    res["outcome"] = "ok"
    print("BENCH_JSON " + json.dumps(res), flush=True)


main()
