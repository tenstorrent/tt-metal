#!/usr/bin/env python3
"""Re-measure every corpus + HeyGen shape on the current HEAD. Resumable: skips shapes already recorded.

One worker SUBPROCESS per shape, because the device-profiler CSV is only flushed at close_device.
Emits, per shape, the worker's SWEEP_JSON plus the factory's config log line, in the format
prod_sweep_report.py already consumes.
"""
import json, os, re, subprocess, sys

S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "head_sweep.jsonl")
old = json.load(open(os.path.join(S, "old.json")))
order = old["order"]

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        m = re.search(r'"M": (\d+), "K": (\d+), "N": (\d+)', line)
        if m:
            done.add("%sx%sx%s" % m.groups())

env = dict(os.environ, TT_METAL_DEVICE_PROFILER="1", TT_REGIME_A_LOG_CFG="1")
W = "tools/mm_sweep/picker_gen/prod_sweep_worker.py"
CFG = re.compile(r"regime_a_cfg M=\d+ K=\d+ N=\d+ pick=\([\d,]+\) cores=\d+ reduction=\S+ placement=\S+")

for i, name in enumerate(order):
    if name in done:
        continue
    M, K, N = (int(x) for x in name.split("x"))
    try:
        r = subprocess.run([sys.executable, W, str(M), str(K), str(N), "2"],
                           capture_output=True, text=True, env=env, timeout=300)
        js = next((l for l in r.stdout.splitlines() if l.startswith("SWEEP_JSON")), None)
        if js is None:
            tail = (r.stderr or r.stdout).strip().replace("\n", " ")[-200:]
            js = "SWEEP_JSON " + json.dumps({"M": M, "K": K, "N": N, "outcome": "runtime", "err": tail})
        cfgline = ""
        mm = CFG.search(r.stdout) or CFG.search(r.stderr)
        if mm:
            cfgline = mm.group(0)
    except subprocess.TimeoutExpired:
        js = "SWEEP_JSON " + json.dumps({"M": M, "K": K, "N": N, "outcome": "timeout", "err": "300s"})
        cfgline = ""
    with open(OUT, "a") as f:
        f.write(js + " ||CFG|| " + cfgline + "\n")
    print("[%d/%d] %s" % (i + 1, len(order), name), flush=True)
print("SWEEP COMPLETE")
