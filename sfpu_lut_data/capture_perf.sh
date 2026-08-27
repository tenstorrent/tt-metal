#!/usr/bin/env bash
# Perf-only capture for one tree state, aggregated over N clean sessions.
# Usage: capture_perf.sh <tag> [sessions]
# local.parquet is *replaced* each session, not appended, so each clean session is
# copied out before the next one runs and the aggregate is taken across the copies.
set -uo pipefail
TAG="$1"; WANT="${2:-4}"
ROOT=/localdev/ldjurovic/tt-metal-lutfit
D=$ROOT/sfpu_lut_data
PQ=$ROOT/tt_metal/tt-llk/perf_data/local.parquet
export RUNNER_TEMP=$ROOT/.llkbuild
source /localdev/ldjurovic/tt-metal/tt_metal/tt-llk/tests/.venv/bin/activate
cd $ROOT/tt_metal/tt-llk/tests/python_tests
rm -rf "$D/perf_sessions_$TAG"; mkdir -p "$D/perf_sessions_$TAG"
OK=0
for a in $(seq 1 14); do
  rm -f "$PQ"
  out=$(python -m pytest perf_eltwise_unary_sfpu.py -q -k "GeluAppx or SigmoidAppx" 2>&1 | tail -1)
  case "$out" in
    *"8 passed"*) OK=$((OK+1)); cp "$PQ" "$D/perf_sessions_$TAG/s$OK.parquet"; echo "  session $OK/$WANT ok";;
    *) echo "  discarded (harness raced on its elf dir): ${out:16:34}";;
  esac
  [ $OK -ge $WANT ] && break
done
python - <<PY
import pandas as pd, glob, json
frames = []
for i, f in enumerate(sorted(glob.glob("$D/perf_sessions_$TAG/s*.parquet"))):
    df = pd.read_parquet(f)
    df = df[(df["marker"] == "TILE_LOOP") & (df["approx_mode"].astype(str).str.endswith("Yes"))]
    df = df.assign(session=i)
    frames.append(df)
df = pd.concat(frames)
key = (df["mathop"].astype(str).str.replace("MathOperation.", "", regex=False) + "|"
       + df["dest_acc"].astype(str).str.replace("DestAccumulation.", "", regex=False))
g = df.assign(k=key).groupby("k")["mean(MATH_ISOLATE)"]
out = {k: {"sessions": int(g.count()[k]), "min": int(g.min()[k]), "max": int(g.max()[k]),
           "mean": round(float(g.mean()[k]), 1)} for k in g.groups}
json.dump(out, open("$D/perf_$TAG.json", "w"), indent=1)
for k, v in out.items():
    print(f"  {k:<16} n={v['sessions']}  {v['min']}..{v['max']}  mean {v['mean']}")
PY
