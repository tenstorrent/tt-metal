#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar pool per-case perf on the ZeBu emulator, measured with the device profiler.

One process = one emulator session (one ZeBu job). For every case: one warmup call (with a golden
check) and then --iters measured pool calls. After every call the L1 profiler buffers are drained
(ttnn.ReadDeviceProfiler) and profile_log_device.csv is parsed: every program (run host ID) the call
launched gets a kernel envelope -- first *-KERNEL ZONE_START to last *-KERNEL ZONE_END over all
cores and RISCs -- plus a per-RISC breakdown. A ttnn pool call launches halo, move (halo reallocate)
and the pool program; the LAST program of the call is the pool program.

Same script for every leg (before / after / after@T=1). It picks the tree from TT_METAL_HOME and
strips the shared venv's editable-ttnn finder so PYTHONPATH decides which ttnn is imported.

  TT_METAL_HOME=<tree> TT_METAL_DEVICE_PROFILER=1 ... python qpool_emu_perf.py --leg after --out <dir>

Output: <out>/results.json; compare legs with qpool_emu_report.py.
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HOME = os.environ["TT_METAL_HOME"].rstrip("/")
# The shared venv installed ttnn editable from ONE tree; its meta-path finder outranks PYTHONPATH.
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Finder"]
for _p in (HOME, f"{HOME}/ttnn"):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, HOME)
sys.path.insert(0, f"{HOME}/ttnn")

import torch  # noqa: E402
import ttnn  # noqa: E402

assert ttnn.__file__.startswith(HOME + "/"), f"ttnn imported from {ttnn.__file__}, expected tree {HOME}"
assert ttnn._ttnn.__file__.startswith(HOME + "/"), f"_ttnn from {ttnn._ttnn.__file__}, expected tree {HOME}"

SEED = 0
L1_SMALL = 24576

# ------------------------------------------------------------------------------------------------
# Cases. PR #55001's perf matrix (craq-sim, 32-cluster grid) re-hosted on the 2-cluster emu-quasar-2x3
# grid. cores=None = grid-adaptive (largest divisor of the height tiles <= grid), as in the PR harness;
# on the 2-cluster emulator that caps at 2 where the sim picked 4 (see 'note').
# ------------------------------------------------------------------------------------------------
CASES = [
    # --- PR perf matrix, unchanged where the config is already <= 2 cores ---
    ("k3x3_s1", dict(in_h=8, in_w=8, kernel=(3, 3), stride=(1, 1), padding=(1, 1)), "2 cores on sim and emu"),
    ("k5x5_s2", dict(kernel=(5, 5), stride=(2, 2), padding=(2, 2)), "sim 4 cores x 32 sticks; emu 2 cores x 64 sticks"),
    ("k7x7_s2_large", dict(kernel=(7, 7), stride=(2, 2), padding=(3, 3), cores=1), "identical"),
    ("k8x8_s2_large", dict(kernel=(8, 8), stride=(2, 2), padding=(3, 3), cores=1), "identical"),
    ("k9x9_s2_3chunks", dict(kernel=(9, 9), stride=(2, 2), padding=(4, 4), cores=1), "identical"),
    ("batch2", dict(batch=2, in_h=8, in_w=8), "sim 4 cores x 32 sticks; emu 2 cores x 64 sticks"),
    ("tall_32x4", dict(in_h=32, in_w=4), "sim 4 cores x 32 sticks; emu 2 cores x 64 sticks"),
    ("wide_4x32", dict(in_h=4, in_w=32), "sim 4 cores x 32 sticks; emu 2 cores x 64 sticks"),
    ("wide_c280_3blocks", dict(channels=280, in_h=8, in_w=4, cores=1), "identical"),
    ("avg_k3x3_s1", dict(pool="avg", kernel=(3, 3), stride=(1, 1), padding=(0, 0), cores=1), "identical"),
    ("avg_k7x7_s1_large", dict(pool="avg", kernel=(7, 7), stride=(1, 1), padding=(0, 0), cores=1), "identical"),
    ("width_1x2_c128", dict(channels=128, in_h=8, in_w=4, shard="width", grid_yx=(1, 2)), "identical"),
    # block_2x2 needs a 2x2 grid; the emulator has 2 clusters -> 1x2 block stand-in (rows x channel halves)
    ("block_1x2_c128", dict(channels=128, shard="block", grid_yx=(1, 2)), "stand-in for block_2x2 (2-cluster grid)"),
    # --- per-core-equivalent twins: 32 sticks/core on 2 cores, matching the sim's per-core load ---
    ("k5x5_s2_2c32", dict(in_h=8, in_w=8, kernel=(5, 5), stride=(2, 2), padding=(2, 2), cores=2), "k5x5_s2 @ 32 sticks/core"),
    ("batch2_4x8_2c32", dict(batch=2, in_h=4, in_w=8, cores=2), "batch2 @ 32 sticks/core"),
    ("tall_16x4_2c32", dict(in_h=16, in_w=4, cores=2), "tall_32x4 @ 32 sticks/core"),
    ("wide_4x16_2c32", dict(in_h=4, in_w=16, cores=2), "wide_4x32 @ 32 sticks/core"),
    # --- realistic size the sim could not run: resnet50 stem maxpool geometry on 2 clusters ---
    ("stem_112x112_c64_2c", dict(in_h=112, in_w=112, kernel=(3, 3), stride=(2, 2), padding=(1, 1), cores=2), "resnet50 stem, 6272 sticks/core"),
]


# ------------------------------------------------------------------------------------------------
# Device profiler CSV reader
# ------------------------------------------------------------------------------------------------
class ProfilerLog:
    """Incremental reader of profile_log_device.csv; drain() returns the programs seen since last drain."""

    def __init__(self, device):
        self.device = device
        art = os.environ.get("TT_METAL_PROFILER_DIR", f"{HOME}/generated/profiler")
        self.path = Path(art) / ".logs" / "profile_log_device.csv"
        self.consumed = 0
        self.header = None
        self.freq_line = None
        self.seen_runs = set()  # mid-run dumps may re-emit markers already reported

    def _read_new_rows(self):
        if not self.path.exists():
            return []
        with open(self.path, newline="") as f:
            lines = f.readlines()
        new = lines[self.consumed :]
        self.consumed = len(lines)
        rows = []
        for line in new:
            if line.startswith("ARCH:"):
                self.freq_line = line.strip()
                continue
            if line.startswith("PCIe slot"):
                self.header = [h.strip() for h in line.strip().split(",")]
                continue
            if not line.strip():
                continue
            rows.append([c.strip() for c in next(csv.reader([line]))])
        return rows

    def drain(self):
        ttnn.ReadDeviceProfiler(self.device)
        rows = self._read_new_rows()
        if not rows:
            return []
        assert self.header, "profiler CSV rows before header"
        col = {name: i for i, name in enumerate(self.header)}
        ix_x, ix_y, ix_risc = col["core_x"], col["core_y"], col["RISC processor type"]
        ix_t, ix_run, ix_zone, ix_type = col["time[cycles since reset]"], col["run host ID"], col["zone name"], col["type"]
        by_run = defaultdict(list)
        for r in rows:
            if len(r) <= ix_type:
                continue
            by_run[int(r[ix_run])].append(r)
        programs = []
        for run_id in sorted(by_run):
            if run_id in self.seen_runs:
                continue
            self.seen_runs.add(run_id)
            rs = by_run[run_id]
            k_start, k_end, f_start, f_end = [], [], [], []
            per_risc = {}
            for r in rs:
                zone, typ, t = r[ix_zone], r[ix_type], int(r[ix_t])
                key = f"({r[ix_x]},{r[ix_y]}):{r[ix_risc]}"
                if zone.endswith("-KERNEL"):
                    d = per_risc.setdefault(key, {})
                    if typ == "ZONE_START":
                        k_start.append(t)
                        d["start"] = min(t, d.get("start", t))
                    elif typ == "ZONE_END":
                        k_end.append(t)
                        d["end"] = max(t, d.get("end", t))
                elif zone.endswith("-FW"):
                    if typ == "ZONE_START":
                        f_start.append(t)
                    elif typ == "ZONE_END":
                        f_end.append(t)
            for v in per_risc.values():
                if "start" in v and "end" in v:
                    v["cycles"] = v["end"] - v["start"]
            programs.append(
                dict(
                    run_id=run_id,
                    kernel_cycles=(max(k_end) - min(k_start)) if k_start and k_end else None,
                    kernel_start=min(k_start) if k_start else None,
                    kernel_end=max(k_end) if k_end else None,
                    fw_cycles=(max(f_end) - min(f_start)) if f_start and f_end else None,
                    n_rows=len(rs),
                    riscs=per_risc,
                )
            )
        return programs


# ------------------------------------------------------------------------------------------------
# Case runner (mirrors tests/.../quasar/test_pool2d.py::_run_case)
# ------------------------------------------------------------------------------------------------
def build_input(batch, in_h, in_w, channels):
    torch.manual_seed(SEED)
    return torch.rand((batch, in_h, in_w, channels)).to(torch.bfloat16)


def run_case(device, prof, name, spec, iters, log):
    channels = spec.get("channels", 64)
    batch, in_h, in_w = spec.get("batch", 1), spec.get("in_h", 16), spec.get("in_w", 8)
    kernel, stride, padding = list(spec.get("kernel", (3, 3))), list(spec.get("stride", (2, 2))), list(spec.get("padding", (1, 1)))
    cores, shard, grid_yx, pool = spec.get("cores"), spec.get("shard", "height"), spec.get("grid_yx"), spec.get("pool", "max")
    tensor_height = batch * in_h * in_w
    assert tensor_height % 32 == 0, f"{name}: N*H*W={tensor_height} must be a multiple of 32"
    tiled_input = channels % 32 == 0

    x_nhwc = build_input(batch, in_h, in_w, channels)
    if pool == "max":
        golden_nchw = torch.nn.functional.max_pool2d(x_nhwc.permute(0, 3, 1, 2).float(), kernel, stride, padding)
    else:
        golden_nchw = torch.nn.functional.avg_pool2d(x_nhwc.permute(0, 3, 1, 2).float(), kernel, stride, padding)
    out_h, out_w = golden_nchw.shape[2], golden_nchw.shape[3]
    golden = golden_nchw.permute(0, 2, 3, 1).reshape(batch * out_h * out_w, channels).contiguous()

    grid = device.compute_with_storage_grid_size()
    if shard == "height":
        height_tiles = tensor_height // 32
        num_cores = cores or max(c for c in range(1, grid.x * grid.y + 1) if height_tiles % c == 0)
        assert num_cores <= grid.x * grid.y, f"{name}: cores={num_cores} > grid {grid.x}x{grid.y}"
        shard_height = (height_tiles // num_cores) * 32
        mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, shard_height, channels),
            core_grid=ttnn.num_cores_to_corerangeset(num_cores, grid, True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        core_desc = f"{num_cores}xHEIGHT ({tensor_height // num_cores} sticks/core)"
    else:
        gy, gx = grid_yx
        strategy = ttnn.ShardStrategy.BLOCK if shard == "block" else ttnn.ShardStrategy.WIDTH
        mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, tensor_height, channels),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=strategy,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        core_desc = f"{gy}x{gx}x{shard.upper()}"

    log(
        f"QPOOL-PERF: {name}: {pool} C={channels} in={batch}x{in_h}x{in_w} k={kernel} s={stride} p={padding} "
        f"{core_desc} layout={'TILE' if tiled_input else 'ROW_MAJOR'}"
    )

    x = ttnn.from_torch(
        x_nhwc.reshape(1, 1, tensor_height, channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT if tiled_input else ttnn.ROW_MAJOR_LAYOUT,
    ).to(device, mem_config)
    ttnn.synchronize_device(device)
    setup_progs = prof.drain()  # interleaved->sharded etc.; not part of the measurement

    pool_op = ttnn.experimental.quasar.max_pool2d if pool == "max" else ttnn.experimental.quasar.avg_pool2d

    def call():
        out = pool_op(
            input_tensor=x,
            batch_size=batch,
            input_h=in_h,
            input_w=in_w,
            channels=channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            **(dict(dilation=[1, 1]) if pool == "max" else {}),
        )
        ttnn.synchronize_device(device)
        return out

    result = dict(spec=dict(spec, kernel=kernel, stride=stride, padding=padding), core_desc=core_desc, grid=[grid.x, grid.y])

    # warmup + golden
    t0 = time.time()
    out = call()
    warm_progs = prof.drain()
    got = ttnn.to_torch(out).float().reshape(batch * out_h * out_w, channels)
    out.deallocate()
    ttnn.synchronize_device(device)
    prof.drain()  # sharded->interleaved readback programs
    max_diff = (got - golden).abs().max().item()
    close = torch.allclose(got, golden, rtol=0.01, atol=0.01)
    result["verdict"] = ("PASS" if close else "MISMATCH") + f" max_abs_diff={max_diff:.6f}"
    result["warmup"] = dict(programs=warm_progs, wall_s=time.time() - t0)
    result["setup_programs"] = len(setup_progs)
    log(f"QPOOL-PERF: {name}: warmup {result['verdict']} programs/call={len(warm_progs)} wall={time.time() - t0:.1f}s")

    result["iters"] = []
    for i in range(iters):
        t0 = time.time()
        out = call()
        progs = prof.drain()
        out.deallocate()
        wall = time.time() - t0
        result["iters"].append(dict(programs=progs, wall_s=wall))
        pool_prog = progs[-1] if progs else None
        total = sum(p["kernel_cycles"] or 0 for p in progs)
        log(
            f"QPOOL-PERF: {name}: iter {i}: programs={len(progs)} "
            f"pool_kernel_cycles={pool_prog['kernel_cycles'] if pool_prog else None} "
            f"all_programs_kernel_cycles={total} wall={wall:.1f}s"
        )
    x.deallocate()
    ttnn.synchronize_device(device)
    prof.drain()
    return result


def run_aux(device, prof, log):
    """What does the Quasar L1-only profiler path record per pool call? A ttnn pool call launches halo,
    move (halo reallocate) and pool2d, yet one program per call comes back. Launch `move` alone, then
    pool with and without the move, and compare run IDs and cycles: identical pool cycles with/without
    the move (and a recorded standalone move) means only the LAST launch of a call survives in L1."""
    channels, batch, in_h, in_w = 64, 1, 16, 8
    tensor_height = batch * in_h * in_w
    grid = device.compute_with_storage_grid_size()
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, tensor_height // 2, channels),
        core_grid=ttnn.num_cores_to_corerangeset(2, grid, True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(
        build_input(batch, in_h, in_w, channels).reshape(1, 1, tensor_height, channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    ).to(device, mem_config)
    ttnn.synchronize_device(device)
    prof.drain()
    out = {}

    def rec(label, progs):
        out[label] = progs
        log(
            f"QPOOL-AUX: {label}: programs={len(progs)} "
            + " ".join(f"[run_id={p['run_id']} kernel_cycles={p['kernel_cycles']} riscs={len(p['riscs'])}]" for p in progs)
        )

    for i in range(2):
        y = ttnn.experimental.quasar.move(x)
        ttnn.synchronize_device(device)
        rec(f"move_alone_{i}", prof.drain())
        x.deallocate()
        x = y
    for realloc in (True, False, True, False):
        o = ttnn.experimental.quasar.max_pool2d(
            input_tensor=x, batch_size=batch, input_h=in_h, input_w=in_w, channels=channels,
            kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False,
            output_layout=ttnn.ROW_MAJOR_LAYOUT, reallocate_halo_output=realloc,
        )
        ttnn.synchronize_device(device)
        rec(f"pool_realloc_halo={realloc}", prof.drain())
        o.deallocate()
    x.deallocate()
    ttnn.synchronize_device(device)
    prof.drain()
    return out


def summarize(results):
    lines = [f"\n{'case':<24} {'progs':>5} {'pool med':>10} {'pool spread':>28} {'halo+move+pool med':>18}  verdict"]
    for name, r in results.items():
        if "iters" not in r:
            lines.append(f"{name:<24} {r.get('error', '')[:100]}")
            continue
        pools = [it["programs"][-1]["kernel_cycles"] for it in r["iters"] if it["programs"]]
        totals = [sum(p["kernel_cycles"] or 0 for p in it["programs"]) for it in r["iters"]]
        nprog = {len(it["programs"]) for it in r["iters"]}
        lines.append(
            f"{name:<24} {'/'.join(map(str, sorted(nprog))):>5} {statistics.median(pools):>10.0f} "
            f"{str(sorted(pools)):>28} {statistics.median(totals):>18.0f}  {r['verdict']}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--only", default=os.environ.get("QPOOL_ONLY", ""))
    ap.add_argument("--aux", action="store_true", help="run the what-does-the-profiler-record experiment")
    ap.add_argument("--no-cases", action="store_true", help="skip the case matrix (e.g. --aux only)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    assert os.environ.get("TT_METAL_DEVICE_PROFILER") == "1", "set TT_METAL_DEVICE_PROFILER=1"
    git = subprocess.run(["git", "-C", HOME, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
    log(f"leg={args.leg} tree={HOME} git='{git}' ttnn={ttnn.__file__} sim={os.environ.get('TT_METAL_SIMULATOR')}")

    # fresh profiler logs for this process
    logs_dir = Path(os.environ.get("TT_METAL_PROFILER_DIR", f"{HOME}/generated/profiler")) / ".logs"
    shutil.rmtree(logs_dir, ignore_errors=True)

    cases = CASES
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in CASES if c[0] in wanted]
    if args.no_cases:
        cases = []

    log("opening device (this blocks on the ZeBu queue + emulator boot)...")
    t0 = time.time()
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=L1_SMALL)
    grid = device.compute_with_storage_grid_size()
    log(f"device open in {time.time() - t0:.0f}s; arch={ttnn.get_arch_name()} compute grid={grid.x}x{grid.y}")
    prof = ProfilerLog(device)

    results = {}
    meta = dict(leg=args.leg, tree=HOME, git=git, iters=args.iters, grid=[grid.x, grid.y], simulator=os.environ.get("TT_METAL_SIMULATOR"))
    try:
        for name, spec, note in cases:
            try:
                results[name] = run_case(device, prof, name, spec, args.iters, log)
            except Exception as e:  # keep the leg alive; report per case
                msg = str(e).splitlines()[0][:300] if str(e) else repr(e)
                log(f"QPOOL-PERF: {name}: ERROR {msg}")
                results[name] = dict(spec=spec, error=msg)
                try:
                    ttnn.synchronize_device(device)
                    prof.drain()
                except Exception:
                    pass
            results[name]["note"] = note
            meta["freq_line"] = prof.freq_line
            with open(out_dir / "results.json", "w") as f:
                json.dump(dict(meta=meta, cases=results), f, indent=1)
        if args.aux:
            meta["aux"] = run_aux(device, prof, log)
            with open(out_dir / "results.json", "w") as f:
                json.dump(dict(meta=meta, cases=results), f, indent=1)
    finally:
        log(summarize(results))
        try:
            shutil.copy(prof.path, out_dir / "profile_log_device.csv")
        except Exception as e:
            log(f"could not copy profiler csv: {e}")
        ttnn.close_mesh_device(device)
        log("device closed")


if __name__ == "__main__":
    main()
