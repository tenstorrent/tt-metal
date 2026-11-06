#!/usr/bin/env bash
# usage: sweep.sh "<shapes csv>" "<configs csv>" "<buffers csv>" [reps]  -> one line per test id: median device us
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES
export NOMIC_SDPA_SHAPES="$1" NOMIC_SDPA_CONFIGS="$2" NOMIC_SDPA_BUFFERS="$3" NOMIC_SDPA_REPS="${4:-3}" \
       NOMIC_SDPA_RESERVE_KB="${NOMIC_SDPA_RESERVE_KB:-176}"
T=tests/ttnn/unit_tests/operations/nomic_sdpa/test_nomic_sdpa_perf.py
# parametrize order, filtered like the test body filters (shape / config / buffer env lists)
ran=$(python -m pytest "$T" --collect-only -q 2>/dev/null | grep -o "test_nomic_sdpa_perf\[[^]]*\]" | python -c "
import os, re, sys
f = {k: [x for x in os.environ[k].split(',') if x] for k in ('NOMIC_SDPA_SHAPES', 'NOMIC_SDPA_CONFIGS', 'NOMIC_SDPA_BUFFERS')}
for line in sys.stdin:
    s, c, b = re.match(r'.*\[(.*)-config=(.*)-buffer=(.*)\]', line).groups()
    if all(not f[k] or v in f[k] for k, v in zip(f, (s, c, b))):
        print(line.strip())
")
scripts/run_safe_pytest.sh --run-all --profile "$T" > /tmp/nomic_sweep_$$.log 2>&1
csv=$(grep -o 'PROFILER CSV: .*' /tmp/nomic_sweep_$$.log | awk '{print $3}')
[ -z "$csv" ] && { tail -20 /tmp/nomic_sweep_$$.log; exit 1; }
python - "$csv" "$NOMIC_SDPA_REPS" "$ran" <<'PY'
import csv, sys, statistics
rows = [float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in csv.DictReader(open(sys.argv[1])) if "Generic" in r["OP CODE"]]
reps = int(sys.argv[2]); ids = sys.argv[3].split()
if len(rows) != reps * len(ids):
    print(f"WARNING: {len(rows)} rows for {len(ids)} ids x {reps} reps")
for i, tid in enumerate(ids):
    xs = rows[i * reps:(i + 1) * reps]
    print(f"{tid:60s} median {statistics.median(xs):7.1f}  {[round(x,1) for x in xs]}")
PY
