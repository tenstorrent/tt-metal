"""Iteration 45: the real-weights TP=4 40-layer trace run (device-side AdamW step scalars, replay from step 4)
against iteration 44's eager run with the host scalars, step for step (same seed and data)."""
import json, sys

O = "/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence"
eager = {e["step"]: e for e in json.load(open(f"{O}/iter44_full_tp4_real.json"))["steps"]}
trace = {e["step"]: e for e in json.load(open(f"{O}/iter45_full_tp4_real_trace.json"))["steps"]}
worst = 0.0
for s in sorted(trace):
    if s not in eager:
        continue
    d = trace[s]["loss"] - eager[s]["loss"]
    worst = max(worst, abs(d))
    print(
        f"step {s:2d}  eager(44, host scalars) {eager[s]['loss']:.4f} {eager[s]['ms']:7.1f} ms  "
        f"trace(45) {trace[s]['loss']:.4f} {trace[s]['ms']:7.1f} ms  {'replay' if trace[s].get('trace') else 'eager '}  diff {d:+.4f}"
    )
tol = 0.0625  # one bf16 ulp of a loss in [8, 16)
print(f"max |diff| {worst:.4f} (tol {tol}: one bf16 ulp at these losses)")
print("PASS" if worst <= tol else "FAIL")
