"""Watchdog for the overnight optimize stage.

Liveness from the terminal log. Stall window is deliberately long: F32 records
that termination_check() legitimately blocks ~30 min with no output, so a 20-min
silence is normal here and only ~50 min is suspicious.
"""
import subprocess, sys, time

TERM = "/localdev/lserbedzija/pr46283_evidence/run2_optimize.log"
prev_n = 0
stall = 0
ticks = 0
while True:
    try:
        with open(TERM) as fh:
            n = sum(1 for _ in fh)
    except OSError:
        n = 0
    stall = stall + 1 if n == prev_n else 0
    if stall >= 10:          # 10 x 5 min = ~50 min of no output
        print(f"optimize: STALLED - terminal log stuck at {n} lines for ~50 min", flush=True)
        stall = 0
    prev_n = n
    ps = subprocess.run(["ps", "-ef"], capture_output=True, text=True).stdout
    alive = any("tt_hw_planner" in l and "optimize" in l and "/bin/bash -c" not in l
                for l in ps.splitlines())
    if not alive:
        print(f"optimize: python process gone per ps ({n} log lines)", flush=True)
        sys.exit(0)
    ticks += 1
    if ticks % 12 == 0:      # hourly heartbeat so a healthy long run is visible
        print(f"optimize: alive, {n} log lines after ~{ticks*5} min", flush=True)
    time.sleep(300)
