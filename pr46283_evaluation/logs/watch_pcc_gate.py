"""Signal-only watcher for the remaining optimize run.

Routine pcc_low/ok flips are expected now — the ladder is sweeping dtype/
fidelity/shard rungs that this model's exact-code-match gate makes unwinnable.
Alerts only on: a commit landing, an OUT-OF-RANGE pcc (F42 recurring — a PCC
cannot exceed 1.0, and catching a second instance would prove reproducibility),
or the run ending.
"""
import json, subprocess, sys, time

GATE = "/tmp/perf_mcp_gate_verdicts_voxtral_tts_full_main.json"
WT = "/tmp/tt_hw_planner_voxtral_tts_full_1786824985"
BASE = "51e208f40c"

def commits():
    r = subprocess.run(["git", "-C", WT, "log", "--oneline", f"{BASE}..HEAD"],
                       capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip()]

prev_n = len(commits())
seen_bad = set()
while True:
    try:
        g = json.load(open(GATE))
        pcc = (g.get("pcc") or {}).get("pcc")
        st = (g.get("pcc") or {}).get("status")
    except Exception:
        pcc, st = None, None
    if isinstance(pcc, (int, float)) and not (-1.0 <= pcc <= 1.0) and pcc not in seen_bad:
        seen_bad.add(pcc)
        print(f"F42 RECURRENCE: out-of-range pcc={pcc} status={st} (a PCC cannot exceed 1.0)", flush=True)
    c = commits()
    if len(c) != prev_n:
        print(f"commit landed ({prev_n} -> {len(c)}): {c[0] if c else '(none)'}", flush=True)
        prev_n = len(c)
    ps = subprocess.run(["ps", "-ef"], capture_output=True, text=True).stdout
    if not any("tt_hw_planner" in l and "optimize" in l and "/bin/bash -c" not in l
               for l in ps.splitlines()):
        print(f"optimize ENDED: {len(c)} commits, final pcc={pcc} ({st})", flush=True)
        sys.exit(0)
    time.sleep(120)
