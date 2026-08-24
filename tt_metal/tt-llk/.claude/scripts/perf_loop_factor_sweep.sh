#!/usr/bin/env bash
# Does the bistable jump scale with the loop factor?
#
# If the slow state costs a fixed number of cycles PER TILE, the jump shrinks in
# proportion as the loop factor falls -- and the smallest loop factor that still
# flips is the minimal reproducer, small enough for RTL.
set -uo pipefail

REPO=~/tt-metal; LLK=$REPO/tt_metal/tt-llk; PT=$LLK/tests/python_tests
OUT=~/loop_sweep
export RUNNER_TEMP=$HOME/llk-wh-build
FACTORS="${FACTORS:-1024 256 64 16}"
RUNS="${RUNS:-20}"

mkdir -p "$OUT"; cd "$PT"
source "$LLK/tests/.venv/bin/activate"
trap 'cd "$PT"; git checkout -- perf_math_matmul.py 2>/dev/null; echo "=== restored ==="' EXIT
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }

git diff --quiet -- perf_math_matmul.py || { echo "FATAL: perf_math_matmul.py dirty"; exit 1; }

for LF in $FACTORS; do
    say "loop_factor=$LF"
    git checkout -- perf_math_matmul.py
    sed -i "s/^            LOOP_FACTOR(1024),$/            LOOP_FACTOR($LF),/" perf_math_matmul.py
    sed -i "s/^    configuration\.run(perf_report)\$/    configuration.run(perf_report, run_count=$RUNS)/" perf_math_matmul.py
    if git diff --quiet -- perf_math_matmul.py; then echo "FATAL: edits did not apply"; exit 1; fi

    rm -rf "$LLK/perf_data"
    CHIP_ARCH=wormhole pytest -q --override-ini=log_cli=false --compile-producer -n 10 \
      -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul . > "$OUT/lf${LF}_compile.log" 2>&1
    CHIP_ARCH=wormhole pytest -q --override-ini=log_cli=false --compile-consumer -n 15 \
      -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul . > "$OUT/lf${LF}_run.log" 2>&1
    say "loop_factor=$LF rc=$?"
    rm -rf "$OUT/lf$LF"; cp -r "$LLK/perf_data" "$OUT/lf$LF" 2>/dev/null
done

say analysing
python3 - "$OUT" <<'PY'
import glob, os, re, sys, pandas as pd
out = sys.argv[1]
rows = []
# Directories only: the glob also matches lf<N>_run.log and lf<N>_compile.log.
dirs = [d for d in glob.glob(os.path.join(out, 'lf*')) if os.path.isdir(d)]
for d in sorted(dirs, key=lambda p: -int(re.search(r'lf(\d+)$', p).group(1))):
    lf = int(re.search(r'lf(\d+)$', d).group(1))
    fs = [f for f in glob.glob(os.path.join(d, '**', '*.csv'), recursive=True)
          if not f.endswith(('.post.csv', '.counters.csv'))]
    if not fs:
        print(f"loop_factor {lf}: no CSVs"); continue
    t = pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)
    t = t[t['marker'] == 'TILE_LOOP'].copy()
    t['std(L1_TO_L1)'] = t['std(L1_TO_L1)'].fillna(0)
    t['cv'] = t['std(L1_TO_L1)'] / t['mean(L1_TO_L1)']
    f = t[(t['std(L1_TO_L1)'] > 5) & (t['cv'] > 0.002)]
    rows.append({
        'loop_factor': lf,
        'variants': len(t),
        'median_kernel_cycles': t['mean(L1_TO_L1)'].median(),
        'flipping': len(f),
        'flip_%': round(len(f) / len(t) * 100, 2),
        'median_std': f['std(L1_TO_L1)'].median() if len(f) else 0,
        'max_std': f['std(L1_TO_L1)'].max() if len(f) else 0,
        'median_std_per_tile': (f['std(L1_TO_L1)'] / (lf * f['tile_cnt'])).median() if len(f) else 0,
    })
r = pd.DataFrame(rows)
print(r.round(2).to_string(index=False))
print("\nIf the cost is per tile, median_std_per_tile stays flat as loop_factor falls.")
print("The smallest loop_factor with a non-zero 'flipping' count is your minimal reproducer.")
PY
say DONE
