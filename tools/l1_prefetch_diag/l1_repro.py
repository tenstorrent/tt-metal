#!/usr/bin/env python3
"""
REPRODUCER -- run this after a machine reset to bring the fault back deterministically.

Two stages, cheapest first. Stage 1 needs no Metal, no build, no mesh and no model, and if
the fault is an L1 cell it reproduces in well under a second -- you never run the model again.

  STAGE 1  L1 cell test.  Pattern-writes the dispatch cores' prefetch ring buffer and reads
           it back. A stuck bit fails here immediately, every time, forever. Reset-proof: a
           bad cell is a bad cell.
              FAIL -> reproduced. Done. Hardware defect at a named address and bit.
              PASS -> the cell is sound; the error enters during the DRAM->L1 relay. Stage 2.

  STAGE 2  Relay test.  Drives a small single-device workload that compiles and dispatches
           many distinct programs, filling and wrapping the prefetcher cache, then audits
           every cached binary against the on-disk ELFs. Repeats until it catches one.
           The workload runs in its own process and exits before each audit, so nothing
           contends for the device -- L1 keeps its contents across process exit.

Stage 1 WRITES to L1 and needs the device free. Stage 2's audit is read-only.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="all", help="device ids, or 'all'")
    ap.add_argument("--cores", default="16-2,16-3", help="dispatch cores, noc0")
    ap.add_argument("--ring-start", type=lambda s: int(s, 0), default=0x020000)
    ap.add_argument(
        "--ring-len",
        type=lambda s: int(s, 0),
        default=0x100000,
        help="prefetch ring buffer is 1024 KB (864 KB with >1 HW CQ)",
    )
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--rounds", type=int, default=20, help="stage 2 rounds before giving up")
    ap.add_argument("--iters", type=int, default=40, help="programs per stage-2 round")
    ap.add_argument("--workload-device", type=int, default=0)
    ap.add_argument("--outdir", default="l1_repro_out")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.stage in ("1", "both"):
        print("\n=== STAGE 1: L1 cell test (no Metal) ===")
        r = run(
            [
                PY,
                os.path.join(HERE, "l1_pattern_test.py"),
                "--devices",
                a.devices,
                "--cores",
                a.cores,
                "--start",
                hex(a.ring_start),
                "--len",
                hex(a.ring_len),
            ]
        )
        if r.returncode == 1:
            print("\nREPRODUCED at stage 1: an L1 cell in the prefetch ring buffer is faulty.")
            print("This is your permanent reproducer -- sub-second, one device, no model.")
            return 1
        if r.returncode != 0:
            print("\nstage 1 could not run (device busy? see message above)")
            return 2
        print("\nstage 1 PASS: L1 cells are sound. The error enters on the DRAM->L1 relay.")
        if a.stage == "1":
            return 0

    print("\n=== STAGE 2: DRAM->prefetcher-L1 relay test ===")
    for rd in range(1, a.rounds + 1):
        print(f"\n--- round {rd}/{a.rounds} ---")
        t0 = time.time()
        w = run([PY, os.path.join(HERE, "_workload.py"), "--device", str(a.workload_device), "--iters", str(a.iters)])
        if w.returncode != 0:
            print("workload failed; aborting")
            return 2
        js = os.path.join(a.outdir, f"audit_round{rd:03d}.json")
        r = run(
            [
                PY,
                os.path.join(HERE, "l1_cache_audit.py"),
                "--devices",
                a.devices,
                "--cores",
                a.cores,
                "--quiet",
                "--json",
                js,
            ]
        )
        print(f"  round took {time.time() - t0:.1f}s")
        if r.returncode == 1:
            print(f"\nREPRODUCED at stage 2, round {rd}. Corrupt binary in the prefetcher cache.")
            print(
                f"Now confirm the full chain:\n" f"  {PY} {os.path.join(HERE, 'l1_confirm_state.py')} --audit-json {js}"
            )
            return 1
    print(
        f"\nNOT reproduced in {a.rounds} rounds "
        f"({a.rounds * a.iters} programs). Widen with --rounds/--iters, or the cache never "
        f"landed on the faulty address -- see README."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
