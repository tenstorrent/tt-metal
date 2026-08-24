"""Watchdog for the emit-e2e stage.

Liveness is judged from the TERMINAL log, which is the only file written across
every phase (the builder's JSONL stops growing the moment the builder exits --
mistaking that for a hang was a false alarm once already).
Errors are counted from whatever JSONL transcripts exist, via is_error, not by
grepping for "Traceback" (which matches scripts the agent merely WRITES).
"""
import glob, json, subprocess, sys, time

TERM = "/localdev/lserbedzija/pr46283_evidence/run2_emit_e2e.log"
JSONL = "/localdev/lserbedzija/repos/tt-metal-pr46283/generated/emit_e2e__*full.log"

prev_n = prev_err = 0
stall = 0
while True:
    try:
        with open(TERM) as fh:
            n = sum(1 for _ in fh)
    except OSError:
        n = 0
    err = 0
    for path in glob.glob(JSONL):
        try:
            with open(path) as fh:
                for line in fh:
                    if '"is_error":true' in line.replace(" ", ""):
                        err += 1
        except OSError:
            pass
    if err > prev_err:
        print(f"emit-e2e: REAL tool errors {prev_err} -> {err}", flush=True)
        prev_err = err
    stall = stall + 1 if n == prev_n else 0
    if stall >= 4:
        print(f"emit-e2e: STALLED - terminal log stuck at {n} lines for ~20 min", flush=True)
        stall = 0
    prev_n = n
    ps = subprocess.run(["ps", "-ef"], capture_output=True, text=True).stdout
    alive = any("tt_hw_planner" in l and "emit-e2e" in l and "/bin/bash -c" not in l
                for l in ps.splitlines())
    if not alive:
        print(f"emit-e2e: python process gone per ps ({n} log lines, {err} real errors)", flush=True)
        sys.exit(0)
    time.sleep(300)
