# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full per-op annotation of an EAGER (no-trace) signposted ops_perf_results CSV.

Adds five leading columns, keeps every original column and row:
  PHASE     init_one_time / warm-up / iter:vision / iter:build_prefix /
            iter:prefill / iter:kv_migration / iter:denoise / teardown
            (from run_socket_traced.py EAGER=1 tracy SIGNPOSTS; see annotate_eager_perf.py)
  STAGE     vision / prefill / denoise   (by DEVICE ID), or signpost
  STEP      1..5 for iter:denoise rows (the N Euler steps), blank elsewhere
  LAYER     1-based, resets per stage: SigLIP 1..27, VLM prefill 1..18, denoise 1..18
  SUBSTAGE  attn / mlp (per layer) + head (one-time denoise-loop setup, step 1) +
            tail (per-step velocity-wrap / Euler update) + output (step 5's final wrap)

LAYER/SUBSTAGE for vision+prefill come from _bench_runs/annotate_ops_csv_v4.py
(SDPA-boundary segmentation, 1 SDPA/layer). The denoise loop is re-segmented here so
each Euler step OWNS its trailing velocity-wrap (the x_{t+1}=x_t+dt*v update + cross-chip
Send/Recv): all 5 steps get an identical 602-op body; step 1 additionally carries the
one-time loop head; step 5's wrap is the final denoised output.

Usage:
  python annotate_full.py <eager_ops_perf_results.csv> [out.csv]
The input must be a tracy EAGER run (run_socket_traced.py EAGER=1) — it needs the
PHASE_* signpost rows and exactly 135 SDPA/inference.
"""

import csv
import os
import subprocess
import sys
import tempfile

PHASE_LABEL = {
    "PHASE_warmup": "warm-up",
    "PHASE_prefix_vision": "iter:vision",
    "PHASE_prefix_build": "iter:build_prefix",
    "PHASE_prefix_prefill": "iter:prefill",
    "PHASE_kv_migration": "iter:kv_migration",
    "PHASE_denoise": "iter:denoise",
    "PHASE_end": "teardown",
}
VISION = {"0", "4", "8", "12"}
DENOISE = {"9", "10", "11", "17", "18", "19"}


def _stage(r):
    if r.get("OP TYPE") == "signpost":
        return "signpost"
    d = (r.get("DEVICE ID") or "").strip()
    return "" if d == "" else ("vision" if d in VISION else ("denoise" if d in DENOISE else "prefill"))


def _run_v4(src):
    """Run the shared v4 annotator to a temp file; return its rows (aligned to src)."""
    root = os.environ.get("TT_METAL_HOME") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), *([os.pardir] * 6))
    )
    v4 = os.path.join(root, "_bench_runs", "annotate_ops_csv_v4.py")
    tmp = tempfile.NamedTemporaryFile(suffix="_v4.csv", delete=False).name
    subprocess.run([sys.executable, v4, src, tmp], check=True, stdout=subprocess.DEVNULL)
    rows = list(csv.DictReader(open(tmp)))
    os.unlink(tmp)
    return rows


def annotate(src, out):
    src_rows = list(csv.DictReader(open(src)))
    fields = list(src_rows[0].keys())
    v4 = _run_v4(src)
    assert len(src_rows) == len(v4), f"v4 row count {len(v4)} != src {len(src_rows)}"
    n = len(src_rows)
    PHASEc, STAGEc, STEPc, LAYERc, SUBc = ([""] * n for _ in range(5))

    # pass 1 — PHASE (signpost walk) + STAGE; LAYER/SUBSTAGE from v4 for non-denoise
    phase = "init_one_time"
    for i, (s, v) in enumerate(zip(src_rows, v4)):
        if s.get("OP TYPE") == "signpost":
            phase = PHASE_LABEL.get((s.get("OP CODE") or "").strip(), phase)
            PHASEc[i], STAGEc[i] = phase, "signpost"
            continue
        PHASEc[i], STAGEc[i] = phase, _stage(s)
        if phase != "iter:denoise":
            LAYERc[i], SUBc[i] = v["LAYER"], v["SUBSTAGE"]

    # pass 2 — re-segment iter:denoise so each step owns its trailing velocity-wrap
    D = [i for i in range(n) if PHASEc[i] == "iter:denoise" and src_rows[i].get("OP TYPE") != "signpost"]
    op = lambda li: src_rows[D[li]]["OP CODE"]
    sdpa = [p for p in range(len(D)) if op(p) == "SDPAOperation"]
    assert len(sdpa) == 90, f"expected 90 denoise SDPA (18 layers x 5 steps), got {len(sdpa)}"

    def last_ln_before(target, low):
        last = low
        for p in range(low, target):
            if op(p) == "LayerNormDeviceOperation":
                last = p
        return last

    bounds = [0] + [last_ln_before(sdpa[k * 18], 0) for k in range(1, 5)] + [len(D)]
    for k in range(5):
        s0, s1 = bounds[k], bounds[k + 1]
        step = str(k + 1)
        lsd = sdpa[k * 18 : (k + 1) * 18]
        first_attn = last_ln_before(lsd[0], s0)
        for p in range(s0, first_attn):  # head: only step 1 (one-time loop setup)
            STEPc[D[p]], SUBc[D[p]], LAYERc[D[p]] = step, "head", ""
        layer_start = first_attn
        for li in range(18):
            ls = lsd[li]
            if li + 1 < 18:
                nxt = last_ln_before(lsd[li + 1], ls + 1)
                for p in range(layer_start, ls + 1):
                    STEPc[D[p]], SUBc[D[p]], LAYERc[D[p]] = step, "attn", str(li + 1)
                for p in range(ls + 1, nxt):
                    STEPc[D[p]], SUBc[D[p]], LAYERc[D[p]] = step, "mlp", str(li + 1)
                layer_start = nxt
            else:  # last layer: split its mlp from the velocity-wrap (first Send after SDPA)
                ts = s1
                for p in range(ls + 1, s1):
                    if op(p) == "SendDirectAsyncDeviceOperation":
                        ts = p
                        break
                for p in range(layer_start, ls + 1):
                    STEPc[D[p]], SUBc[D[p]], LAYERc[D[p]] = step, "attn", str(li + 1)
                for p in range(ls + 1, ts):
                    STEPc[D[p]], SUBc[D[p]], LAYERc[D[p]] = step, "mlp", str(li + 1)
                tail = "output" if k == 4 else "tail"  # step 5's wrap = final denoised output
                for p in range(ts, s1):
                    STEPc[D[p]], SUBc[D[p]], LAYERc[D[p]] = step, tail, ""

    out_fields = ["PHASE", "STAGE", "STEP", "LAYER", "SUBSTAGE"] + fields
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for i, s in enumerate(src_rows):
            row = {"PHASE": PHASEc[i], "STAGE": STAGEc[i], "STEP": STEPc[i], "LAYER": LAYERc[i], "SUBSTAGE": SUBc[i]}
            row.update(s)
            w.writerow(row)
    print(f"wrote {out}: {n} rows, {len(out_fields)} cols (PHASE+STAGE+STEP+LAYER+SUBSTAGE + {len(fields)} original)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: annotate_full.py <eager_ops_perf_results.csv> [out.csv]", file=sys.stderr)
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src[:-4] + "_annotated.csv"
    annotate(src, out)
