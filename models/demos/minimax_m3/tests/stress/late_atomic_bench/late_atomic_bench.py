#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Late non-posted atomic response vs the next NCRISC kernel's NoC counter snapshot (issue 59542).

Program A: NCRISC on the A cores fires N atomic incs at the far core and exits (optionally with a barrier,
the control), while BRISC on every other core floods the same NoC. Program B: NCRISC on the same cores
probes NIU_MST_ATOMIC_RESP_RECEIVED - noc_nonposted_atomics_acked, then runs a barrier pattern (--b-mode).
A and B are enqueued back to back (captured in a trace and replayed), with no host sync in between.

Exit codes: 0 = finished, no hang; 3 = dispatch timeout (hang); 1 = error.
See README.md for the run matrix and how to read the output.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "kernels")
TT = os.environ.get("TT_METAL_HOME", os.path.abspath(os.path.join(HERE, *[".."] * 6)))

# Must match kernels/late_atomic_common.h
S_A_RUNS, S_A_BAR_MAX, S_A_BAR_SUM16, S_A_BAR_LATE, S_A_EXIT_TS = 1, 2, 3, 4, 5
S_B_RUNS, S_B_D0_NZ, S_B_D1_NZ, S_B_DMAX, S_B_STATE = 6, 7, 8, 9, 10
S_B_GAP_MIN, S_B_GAP_MAX, S_B_RESID_NZ, S_B_FIRST_HIT = 11, 12, 13, 14
S_B_ENTRY_HW, S_B_ENTRY_SW, S_B_ACTION_MAX, S_B_LAST_D1 = 15, 16, 17, 18
STATS_WORDS = 32
B_STATE_NAMES = {0: "not-started", 1: "IN-BARRIER", 2: "done"}

B_MODES = {"probe": 0, "inc_barrier": 1, "incn_barrier": 2, "late_barrier": 3, "full_barrier": 4}
FLOOD_KINDS = {"write": 0, "atomic": 1, "both": 2}
ROW_WORDS = 1024
ROW_BYTES = ROW_WORDS * 4


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-id", type=int, default=0, help="logical id after TT_VISIBLE_DEVICES filtering")
    ap.add_argument("--noc", type=int, choices=[0, 1], default=1, help="NoC used by A, B and the flood")
    ap.add_argument("--a-cores", type=int, default=0, help="number of A/B cores in column 0 (0 = whole column)")
    ap.add_argument("--a-barrier", action="store_true", help="control: A waits for its atomic responses")
    ap.add_argument("--a-incs", type=int, default=1, help="atomic incs A fires at the far core before exiting")
    ap.add_argument("--a-delay", type=int, default=4000, help="cycles A waits before firing (flood warm-up)")
    ap.add_argument("--a-jitter", type=int, default=16000, help="extra pseudo-random 0..N cycles per run")
    ap.add_argument("--late-thresh", type=int, default=1500, help="control: count A barriers longer than this")
    ap.add_argument("--b-mode", choices=list(B_MODES), default="late_barrier")
    ap.add_argument("--b-incs", type=int, default=4, help="incs for --b-mode incn_barrier")
    ap.add_argument("--probe-wait", type=int, default=20000, help="cycles B waits before probing / settling")
    ap.add_argument("--flood-kind", choices=list(FLOOD_KINDS) + ["none"], default="write")
    ap.add_argument("--flood-dest", choices=["far", "acores", "both"], default="both")
    ap.add_argument("--flood-cycles", type=int, default=40000)
    ap.add_argument("--chunk-bytes", type=int, default=16384)
    ap.add_argument("--max-congestors", type=int, default=0, help="0 = every other core")
    ap.add_argument("--pairs", type=int, default=1_000_000, help="total A->B pairs")
    ap.add_argument("--trace-pairs", type=int, default=256, help="A->B pairs per trace; 0 = eager enqueue")
    ap.add_argument("--sync-pairs", type=int, default=8192, help="pairs between host syncs / progress checks")
    ap.add_argument("--report-s", type=float, default=15.0)
    ap.add_argument("--timeout-s", type=float, default=30.0, help="TT_METAL_OPERATION_TIMEOUT_SECONDS")
    ap.add_argument("--no-triage", action="store_true", help="do not run tt-triage from the timeout handler")
    ap.add_argument("--hold-on-hang", action="store_true", help="keep the hung process alive for manual triage")
    ap.add_argument("--compile-only", action="store_true", help="enqueue one A and one B, sync, exit")
    ap.add_argument("--out-dir", default="", help="default: $TT_METAL_HOME/generated/late_atomic_bench/<utc>_<tag>")
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def setup_env(args, out_dir):
    os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", str(args.timeout_s))
    if not args.no_triage and "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE" not in os.environ:
        triage = os.path.join(out_dir, "triage_on_timeout.txt")
        os.environ["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"] = (
            f"cd {TT} && {sys.executable} tools/tt-triage.py --disable-progress --disable-colors "
            f"--run=dump_running_operations --run=dump_callstacks -vv > {triage} 2>&1"
        )


def main():
    args = parse_args()
    tag = args.tag or f"{args.b_mode}_{'abar' if args.a_barrier else 'nobar'}_{args.flood_kind}"
    out_dir = args.out_dir or os.path.join(TT, "generated", "late_atomic_bench", f"{utc()}_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    setup_env(args, out_dir)

    import torch

    import ttnn

    def log(msg):
        line = f"{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(os.path.join(out_dir, "log.txt"), "a") as f:
            f.write(line + "\n")

    log(f"args {vars(args)}")
    log(f"out_dir {out_dir}")
    trace_bytes = max(64 << 20, args.trace_pairs * 2 * 16384)
    dev = ttnn.open_device(device_id=args.device_id, trace_region_size=trace_bytes)
    chip_id = dev.get_device_ids()[0] if hasattr(dev, "get_device_ids") else dev.id()

    grid = dev.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    n_a = args.a_cores or gy
    assert 1 <= n_a <= gy, f"--a-cores must be 1..{gy}"
    a_cores = [(0, y) for y in range(n_a)]
    far = (gx - 1, gy - 1)
    taken = set(a_cores) | {far}
    congestors = (
        [] if args.flood_kind == "none" else [(x, y) for y in range(gy) for x in range(gx) if (x, y) not in taken]
    )
    if args.max_congestors:
        congestors = congestors[: args.max_congestors]

    def virt(c):
        v = dev.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        return (v.x, v.y)

    def core_set(cores):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])

    rows = (args.chunk_bytes + ROW_BYTES - 1) // ROW_BYTES + 1
    full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    scratch = ttnn.from_torch(
        torch.zeros(gx * gy * rows, ROW_WORDS, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(full_grid, (rows, ROW_WORDS), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    # generic_op wants at least one input and one output tensor; this one is never touched.
    dummy = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    scratch_addr = scratch.buffer_address()
    stats_addr = scratch_addr + (rows - 1) * ROW_BYTES

    fvx, fvy = virt(far)
    a_virt = [virt(c) for c in a_cores]
    log(
        f"chip {chip_id} grid {gx}x{gy}; A/B cores (NCRISC, NOC{args.noc}) {a_cores} virt {a_virt}; "
        f"far {far} virt {(fvx, fvy)}; {len(congestors)} congestors (BRISC, NOC{args.noc}) "
        f"kind={args.flood_kind} dest={args.flood_dest}; scratch 0x{scratch_addr:x} stats 0x{stats_addr:x}"
    )

    noc = ttnn.NOC.RISCV_1_default if args.noc == 1 else ttnn.NOC.RISCV_0_default

    def dm(processor):
        return ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc)

    def kernel(name, cores, ct, rt, processor):
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set(cores),
            compile_time_args=ct,
            runtime_args=rt,
            config=dm(processor),
            compiler_include_paths=[KDIR],
        )

    ka = kernel(
        "late_atomic_a.cpp",
        a_cores,
        [int(args.a_barrier), args.a_incs, args.a_delay, args.a_jitter, args.late_thresh],
        [(ttnn.CoreCoord(*c), [stats_addr, fvx, fvy]) for c in a_cores],
        ttnn.DataMovementProcessor.RISCV_1,
    )
    kernels_a = [ka]
    if congestors:
        flood_rt = []
        for i, c in enumerate(congestors):
            if args.flood_dest == "far" or (args.flood_dest == "both" and i % 2 == 0):
                dx, dy = fvx, fvy
            else:
                idx = i if args.flood_dest == "acores" else i // 2
                dx, dy = a_virt[idx % n_a]
            flood_rt.append((ttnn.CoreCoord(*c), [scratch_addr, stats_addr, dx, dy]))
        kernels_a.append(
            kernel(
                "late_atomic_flood.cpp",
                congestors,
                [args.flood_cycles, args.chunk_bytes, FLOOD_KINDS[args.flood_kind]],
                flood_rt,
                ttnn.DataMovementProcessor.RISCV_0,
            )
        )
    peer_args = [w for v in a_virt for w in v]
    kb = kernel(
        "late_atomic_b.cpp",
        a_cores,
        [B_MODES[args.b_mode], args.probe_wait, args.b_incs],
        [(ttnn.CoreCoord(*c), [stats_addr, *a_virt[i], n_a, *peer_args]) for i, c in enumerate(a_cores)],
        ttnn.DataMovementProcessor.RISCV_1,
    )
    pd_a = ttnn.ProgramDescriptor(kernels=kernels_a)
    pd_b = ttnn.ProgramDescriptor(kernels=[kb])

    def run_pair():
        ttnn.generic_op([scratch, dummy], pd_a)
        ttnn.generic_op([scratch, dummy], pd_b)

    def stats_from_bytes(raw):
        return [int.from_bytes(raw[4 * i : 4 * i + 4], "little") for i in range(STATS_WORDS)]

    def read_stats_cq():
        t = ttnn.to_torch(scratch).to(torch.int64) & 0xFFFFFFFF
        out = {}
        for x, y in a_cores:
            r = (y * gx + x) * rows + rows - 1
            out[(x, y)] = [int(v) for v in t[r, :STATS_WORDS]]
        return out

    def read_stats_raw():
        # UMD path, independent of the (hung) command queue.
        out = {}
        for c, (vx, vy) in zip(a_cores, a_virt):
            try:
                out[c] = stats_from_bytes(ttnn.cluster.read_from_core(chip_id, vx, vy, stats_addr, 4 * STATS_WORDS))
            except Exception as e:  # noqa: BLE001
                out[c] = f"read failed: {e}"
        return out

    def summarize(st):
        rows_ = [v for v in st.values() if isinstance(v, list)]
        a_runs = sum(v[S_A_RUNS] for v in rows_)
        s = {
            "a_runs": a_runs,
            "b_runs": sum(v[S_B_RUNS] for v in rows_),
            "b_d0_nz": sum(v[S_B_D0_NZ] for v in rows_),
            "b_d1_nz": sum(v[S_B_D1_NZ] for v in rows_),
            "b_dmax": max((v[S_B_DMAX] for v in rows_), default=0),
            "b_resid_nz": sum(v[S_B_RESID_NZ] for v in rows_),
            "b_gap_min": min((v[S_B_GAP_MIN] for v in rows_ if v[S_B_RUNS]), default=0),
            "b_gap_max": max((v[S_B_GAP_MAX] for v in rows_), default=0),
            "b_action_max": max((v[S_B_ACTION_MAX] for v in rows_), default=0),
        }
        if args.a_barrier:
            s["a_bar_max"] = max((v[S_A_BAR_MAX] for v in rows_), default=0)
            s["a_bar_mean"] = round(16 * sum(v[S_A_BAR_SUM16] for v in rows_) / max(a_runs, 1), 1)
            s[f"a_bar_gt_{args.late_thresh}"] = sum(v[S_A_BAR_LATE] for v in rows_)
        return s

    def per_core_lines(st):
        lines = []
        for c, v in st.items():
            if not isinstance(v, list):
                lines.append(f"  core {c}: {v}")
                continue
            lines.append(
                f"  core {c} virt {a_virt[a_cores.index(c)]}: A runs {v[S_A_RUNS]} B runs {v[S_B_RUNS]} "
                f"B state {B_STATE_NAMES.get(v[S_B_STATE], v[S_B_STATE])} "
                f"entry hw/sw {v[S_B_ENTRY_HW]}/{v[S_B_ENTRY_SW]} (diff {(v[S_B_ENTRY_HW] - v[S_B_ENTRY_SW]) & 0xFFFFFFFF}) "
                f"d1!=0 {v[S_B_D1_NZ]} (first at B run {v[S_B_FIRST_HIT]}) resid {v[S_B_RESID_NZ]}"
            )
        return lines

    def write_summary(status, st, done_pairs, wall):
        s = {"status": status, "pairs": done_pairs, "wall_s": round(wall, 1), "args": vars(args), **summarize(st)}
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(s, f, indent=1)
        log(f"SUMMARY {json.dumps({k: v for k, v in s.items() if k != 'args'})}")

    def on_hang(exc, done_pairs, wall):
        log(f"HANG: dispatch timeout after ~{done_pairs} completed pairs: {str(exc).splitlines()[0][:200]}")
        st = read_stats_raw()
        for line in per_core_lines(st):
            log(line)
        stuck = [c for c, v in st.items() if isinstance(v, list) and v[S_B_STATE] == 1]
        if stuck:
            log(
                f"cores stuck inside B's barrier action ({args.b_mode}): {stuck}. "
                f"hw > sw at entry means a late atomic response from A landed after B's counter snapshot."
            )
        else:
            log("no A/B core is inside B's barrier action; the hang is elsewhere (check the triage output)")
        triage = os.path.join(out_dir, "triage_on_timeout.txt")
        if os.path.exists(triage):
            log(f"tt-triage output: {triage} (look for NCRISC in noc_async_atomic_barrier / waypoint NABW or NFEW)")
        write_summary("hang", st, done_pairs, wall)
        if args.hold_on_hang:
            vis = os.environ.get("TT_VISIBLE_DEVICES", "<unset>")
            log(
                f"holding the device for manual triage (pid {os.getpid()}). From another shell: "
                f"cd {TT} && TT_VISIBLE_DEVICES={vis} python3 tools/tt-triage.py --disable-progress "
                f"--run=dump_callstacks -vv ; then kill {os.getpid()} and reset the chip"
            )
            while True:
                time.sleep(3600)
        os._exit(3)

    t0 = time.time()
    done = 0
    try:
        log("compiling + first pair (eager)")
        run_pair()
        ttnn.synchronize_device(dev)
        done = 1
        log(f"first pair ok after {time.time() - t0:.1f}s")
        if args.compile_only:
            log("compile-only: done")
            ttnn.close_device(dev)
            return 0

        trace_id = None
        if args.trace_pairs > 0:
            trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
            for _ in range(args.trace_pairs):
                run_pair()
            ttnn.end_trace_capture(dev, trace_id, cq_id=0)
            ttnn.synchronize_device(dev)
            reps = max(1, args.sync_pairs // args.trace_pairs)
            per_sync = reps * args.trace_pairs
        else:
            per_sync = args.sync_pairs

        t_run = time.time()
        last_report = t_run
        while done < args.pairs:
            if trace_id is not None:
                for _ in range(reps):
                    ttnn.execute_trace(dev, trace_id, cq_id=0, blocking=False)
            else:
                for _ in range(per_sync):
                    run_pair()
            ttnn.synchronize_device(dev)
            done += per_sync
            now = time.time()
            if now - last_report >= args.report_s or done >= args.pairs:
                last_report = now
                s = summarize(read_stats_cq())
                rate = (done - 1) / max(now - t_run, 1e-9)
                log(f"pairs {done} ({rate:.0f}/s) {json.dumps(s)}")
        if trace_id is not None:
            ttnn.release_trace(dev, trace_id)
    except RuntimeError as e:
        if "TIMEOUT" in str(e) or "timeout" in str(e):
            on_hang(e, done, time.time() - t0)
        raise

    st = read_stats_cq()
    for line in per_core_lines(st):
        log(line)
    write_summary("ok", st, done, time.time() - t0)
    s = summarize(st)
    if s["b_d1_nz"]:
        log(
            f"late A responses observed after B's snapshot in {s['b_d1_nz']} of {s['b_runs']} B runs "
            f"(B mode {args.b_mode} survived them; resid {s['b_resid_nz']} = barrier returned with hw != sw)"
        )
    else:
        log("no late A response observed after B's snapshot")
    ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
