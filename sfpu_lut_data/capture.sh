#!/usr/bin/env bash
# Capture every measurement for one tree state.  Usage: capture.sh <tag>
set -uo pipefail
TAG="$1"
ROOT=/localdev/ldjurovic/tt-metal-lutfit
D=$ROOT/sfpu_lut_data
export RUNNER_TEMP=$ROOT/.llkbuild          # isolated build cache: no cross-tree ELF reuse
source /localdev/ldjurovic/tt-metal/tt_metal/tt-llk/tests/.venv/bin/activate
cd $ROOT/tt_metal/tt-llk/tests/python_tests
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

echo "== $TAG: $(git -C $ROOT rev-parse --abbrev-ref HEAD) @ $(git -C $ROOT rev-parse --short HEAD)"
LUT_DUMP=$D/curves_$TAG.json python -m pytest test_sfpu_wh_lut_curves.py -q -s 2>&1 | grep wrote
python -m pytest test_sfpu_wh_lut_accuracy.py -s -q 2>&1 | strip | grep -A 8 '^=== ' > $D/accuracy_$TAG.txt
python -m pytest test_sfpu_wh_lut_probe.py    -s -q 2>&1 | strip | grep -A 50 '^=== ' > $D/probe_$TAG.txt

# Three perf repetitions so the run-to-run spread is measured, not assumed. The harness
# races on creating its per-hash elf/ dir when the cache is cold, so the first attempt is
# a warmup and each rep retries until it reports a full pass.
REPS=0
for attempt in $(seq 1 6); do
  out=$(python -m pytest perf_eltwise_unary_sfpu.py -q -k "GeluAppx or SigmoidAppx" 2>&1 | tail -1)
  case "$out" in *"8 passed"*) REPS=$((REPS+1)); echo "  perf rep $REPS ok";; *) echo "  perf warmup/retry: ${out:0:60}";; esac
  [ $REPS -ge 3 ] && break
done

python - <<PY
import pandas as pd, json
df = pd.read_parquet("$ROOT/tt_metal/tt-llk/perf_data/local.parquet")
df = df[(df["marker"] == "TILE_LOOP") & (df["approx_mode"].astype(str).str.endswith("Yes"))]
keep = sorted(df["timestamp"].unique())[-$REPS:]
df = df[df["timestamp"].isin(keep)]
g = df.groupby([df["mathop"].astype(str), df["dest_acc"].astype(str)])["mean(MATH_ISOLATE)"]
out = {f"{a}|{b}": {"reps": int(n), "min": float(mn), "max": float(mx), "mean": round(float(m), 1)}
       for (a, b), n, mn, mx, m in zip(g.groups, g.count(), g.min(), g.max(), g.mean())}
json.dump(out, open("$D/perf_$TAG.json", "w"), indent=1)
for k, v in out.items():
    print(f"  {k:<52} n={v['reps']}  {v['min']:.0f}..{v['max']:.0f}  mean {v['mean']}")
PY
echo "captured: $TAG"
