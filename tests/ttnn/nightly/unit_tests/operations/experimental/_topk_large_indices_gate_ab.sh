#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Gate-constant A/B for the row-parallel chunk skip in
# ttnn.experimental.topk_large_indices (charter item I4).
#
# Sweeps CHUNK_SKIP_GATE_DIVISOR (first_tested = max(2, USER_K/divisor)) over
# {4 (shipping), 2, 8} by sed-editing the #define in
# topk_large_indices_chunk_skip.hpp per arm (in-source -> JIT rehash; no host
# rebuild). Per arm it collects:
#
#   1. WALL TIME (telemetry OFF, Tracy DEVICE KERNEL DURATION): cells
#      rows=2/k=32, rows=8/k=32 (shipping-relevant) and rows=2/k=512
#      (discriminator: /4 and /2 put the gate at/above num_chunks -> zero
#      tests; /8 pays 64 tests/row for ~1 expected skip), W=65536,
#      3 trials x 5 iters, median per trial. Kernel cache cleared per arm.
#   2. SKIP RATE (CHUNK_SKIP_TELEMETRY ON, DPRINT -> log, seeds varied):
#      observed skips/row + per-position P(skip|c) vs the amortization law
#      e^(-USER_K/(c+1)), via _topk_large_indices_skip_telemetry_parse.py.
#
# Timing and telemetry runs are disjoint by construction (DPRINT and the
# device profiler are mutually exclusive), which is sound because skip
# decisions are pure functions of the input data.
#
# Every device invocation runs under flock /tmp/tt-device.lock and an outer
# timeout. The header is restored (divisor=4, telemetry commented out) on
# exit via trap; the script aborts if the parent env carries DPRINT/WATCHER.
#
# DECISION RULE (charter stop rule -- do not thrash):
#   Change the shipping /4 ONLY if some arm beats /4 by >=5% median on a
#   k=32 cell with cross-trial spread < 1/3 of the delta, AND the k=512
#   discriminator regresses <=0.5% on that arm, AND the adversarial battery
#   shows no regression. The law predicts /4 vs /2 is a statistical tie and
#   /8 loses the discriminator -- ties mean KEEP /4 and record the table
#   (it doubles as the paper's gate-sensitivity ablation).
#
# Usage (from anywhere; ~2h device time at defaults):
#   bash tests/ttnn/nightly/unit_tests/operations/experimental/_topk_large_indices_gate_ab.sh
# Knobs: I4_OUT, I4_DIVISORS, I4_TRIALS, I4_TELEM_SEEDS, I4_SKIP_TIMING=1,
#        I4_SKIP_TELEM=1.

set -euo pipefail

ROOT="${TT_METAL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)}"
HDR="$ROOT/ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/topk_large_indices_chunk_skip.hpp"
TDIR="tests/ttnn/nightly/unit_tests/operations/experimental"
BENCH="$TDIR/_topk_large_indices_bench.py"
PARSE="$TDIR/_topk_large_indices_skip_telemetry_parse.py"
LOCK="/tmp/tt-device.lock"
OUT="${I4_OUT:-/tmp/logs/i4_gate_ab_$(date +%Y%m%d_%H%M%S)}"

DIVISORS="${I4_DIVISORS:-4 2 8}"
TRIALS="${I4_TRIALS:-3}"
CELLS="2:32 8:32 2:512"           # rows:user_k at W=65536 (llk window 512, 128 chunks)
TELEM_SEEDS="${I4_TELEM_SEEDS:-0 1 2 3 4}"
W=65536

# ---------------------------------------------------------------- guards ---
[[ -f "$HDR" ]] || { echo "FATAL: header not found: $HDR"; exit 2; }
grep -q '^#define CHUNK_SKIP_GATE_DIVISOR 4$' "$HDR" || {
    echo "FATAL: CHUNK_SKIP_GATE_DIVISOR knob not in header (apply patch 0002) or not at default 4"; exit 2; }
grep -q '^// #define CHUNK_SKIP_TELEMETRY 1$' "$HDR" || {
    echo "FATAL: CHUNK_SKIP_TELEMETRY define not in header (apply patch 0001) or already enabled"; exit 2; }
if env | grep -qE '^TT_METAL_(DPRINT|WATCHER)'; then
    echo "FATAL: parent env carries TT_METAL_DPRINT*/TT_METAL_WATCHER* -- unset before profiling"; exit 2
fi
cd "$ROOT"
mkdir -p "$OUT"
echo "output: $OUT"

set_divisor() {
    sed -i -E "s/^#define CHUNK_SKIP_GATE_DIVISOR [0-9]+\$/#define CHUNK_SKIP_GATE_DIVISOR $1/" "$HDR"
    grep -q "^#define CHUNK_SKIP_GATE_DIVISOR $1\$" "$HDR" || { echo "FATAL: set_divisor $1 failed"; exit 3; }
}
telemetry_on()  { sed -i 's|^// #define CHUNK_SKIP_TELEMETRY 1$|#define CHUNK_SKIP_TELEMETRY 1|' "$HDR"; }
telemetry_off() { sed -i 's|^#define CHUNK_SKIP_TELEMETRY 1$|// #define CHUNK_SKIP_TELEMETRY 1|' "$HDR"; }
restore() { telemetry_off; set_divisor 4; echo "header restored (divisor=4, telemetry off)"; }
trap restore EXIT

clear_kernel_cache() {
    rm -rf "$HOME"/.cache/tt-metal-cache*/kernels "$HOME"/.cache/tt-metal-cache/*/kernels \
        "$ROOT/generated/kernels" 2>/dev/null || true
}

# Median DEVICE KERNEL DURATION [ns] of the newest Tracy report (warmup row
# dropped). Columns picked by HEADER NAME (schema-shift tolerant).
extract_median_ns() {
    python - "$ROOT" <<'PYEOF'
import csv, glob, os, statistics, sys
root = sys.argv[1]
reports = sorted(glob.glob(os.path.join(root, "generated/profiler/reports", "*")), key=os.path.getmtime)
assert reports, "no Tracy reports found"
csvs = glob.glob(os.path.join(reports[-1], "ops_perf_results_*.csv"))
assert csvs, f"no ops_perf_results CSV in {reports[-1]}"
with open(csvs[0]) as f:
    rd = csv.DictReader(f)
    opcol = next(c for c in rd.fieldnames if c.strip() == "OP CODE")
    dkcol = next(c for c in rd.fieldnames if "DEVICE KERNEL DURATION" in c)
    durs = [int(r[dkcol]) for r in rd if "topk" in r[opcol].lower() and r[dkcol]]
assert len(durs) >= 2, f"expected warmup+iters topk rows, got {len(durs)}"
print(int(statistics.median(durs[1:])))  # drop warmup
PYEOF
}

TIMING_CSV="$OUT/timing.csv"
echo "divisor,rows,user_k,trial,median_ns" > "$TIMING_CSV"

for d in $DIVISORS; do
    echo "=================== ARM divisor=$d ==================="
    set_divisor "$d"

    # ---------------- wall time (telemetry OFF, Tracy) ----------------
    if [[ "${I4_SKIP_TIMING:-0}" != "1" ]]; then
        grep -q '^// #define CHUNK_SKIP_TELEMETRY 1$' "$HDR" || { echo "FATAL: telemetry must be OFF for timing"; exit 3; }
        clear_kernel_cache
        for cell in $CELLS; do
            rows="${cell%%:*}"; k="${cell##*:}"
            for trial in $(seq 1 "$TRIALS"); do
                echo "--- timing d=$d rows=$rows k=$k trial=$trial"
                flock "$LOCK" timeout 600 env \
                    TOPK_ROWS="$rows" TOPK_K="$k" TOPK_W="$W" TOPK_ITERS=5 \
                    python -m tracy -r -v "$BENCH" \
                    2>&1 | tee "$OUT/tracy_d${d}_r${rows}_k${k}_t${trial}.log"
                grep -q "OK k=$k W=$W rows=$rows" "$OUT/tracy_d${d}_r${rows}_k${k}_t${trial}.log" || {
                    echo "FATAL: bench correctness line missing (d=$d rows=$rows k=$k t=$trial)"; exit 4; }
                ns=$(extract_median_ns)
                echo "$d,$rows,$k,$trial,$ns" >> "$TIMING_CSV"
                echo "    median = $ns ns"
            done
        done
    fi

    # ---------------- skip rate (telemetry ON, DPRINT) ----------------
    if [[ "${I4_SKIP_TELEM:-0}" != "1" ]]; then
        telemetry_on
        clear_kernel_cache
        for cell in $CELLS; do
            rows="${cell%%:*}"; k="${cell##*:}"
            for seed in $TELEM_SEEDS; do
                log="$OUT/telem_d${d}_r${rows}_k${k}_s${seed}.log"
                echo "--- telemetry d=$d rows=$rows k=$k seed=$seed"
                flock "$LOCK" timeout 300 env \
                    TT_METAL_DPRINT_CORES=worker TT_METAL_DPRINT_RISCVS=TR1 \
                    TT_METAL_DPRINT_FILE="$log" \
                    TOPK_ROWS="$rows" TOPK_K="$k" TOPK_W="$W" TOPK_ITERS=1 TOPK_SEED="$seed" \
                    python "$BENCH" > "$OUT/telem_d${d}_r${rows}_k${k}_s${seed}.out" 2>&1
                grep -q "OK k=$k W=$W rows=$rows" "$OUT/telem_d${d}_r${rows}_k${k}_s${seed}.out" || {
                    echo "FATAL: telemetry-on run failed correctness (d=$d rows=$rows k=$k s=$seed)"; exit 4; }
            done
            python "$PARSE" --user-k "$k" --csv "$OUT/skiprate_d${d}_r${rows}_k${k}.csv" \
                "$OUT"/telem_d${d}_r${rows}_k${k}_s*.log \
                > "$OUT/skiprate_d${d}_r${rows}_k${k}.txt"
            tail -2 "$OUT/skiprate_d${d}_r${rows}_k${k}.txt"
        done
        telemetry_off
    fi
done

# -------------------------------------------------------------- summary ---
python - "$OUT" <<'PYEOF'
import csv, glob, math, os, re, statistics, sys

out = sys.argv[1]
timing = {}  # (d, rows, k) -> [median_ns]
with open(os.path.join(out, "timing.csv")) as f:
    for r in csv.DictReader(f):
        timing.setdefault((int(r["divisor"]), int(r["rows"]), int(r["user_k"])), []).append(int(r["median_ns"]))

skip = {}  # (d, rows, k) -> (obs_skips_per_row, law_skips_per_row)
for p in glob.glob(os.path.join(out, "skiprate_d*_r*_k*.csv")):
    m = re.search(r"skiprate_d(\d+)_r(\d+)_k(\d+)\.csv", p)
    with open(p) as f:
        for row in csv.reader(f):
            if row and row[0] == "TOTAL":
                skip[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = (float(row[3]), float(row[4]))

cells = sorted({(rows, k) for (_, rows, k) in list(timing) + list(skip)})
divs = sorted({d for (d, _, _) in list(timing) + list(skip)})
base = 4
print("\n================= GATE A/B SUMMARY (W=65536, 128 chunks) =================")
print(f"{'cell':>14} {'div':>4} {'gate':>5} {'med_us':>9} {'spread%':>8} {'vs /4':>8} {'skips obs':>10} {'skips law':>10}")
for rows, k in cells:
    ref = statistics.median(timing[(base, rows, k)]) if (base, rows, k) in timing else None
    for d in divs:
        gate = max(2, k // d)
        t = timing.get((d, rows, k))
        med = statistics.median(t) if t else None
        spread = (max(t) - min(t)) / med * 100 if t and med else None
        delta = (med - ref) / ref * 100 if med and ref else None
        s = skip.get((d, rows, k))
        print(
            f"{f'rows={rows} k={k}':>14} {d:>4} {gate:>5} "
            f"{f'{med/1000:.2f}' if med else '-':>9} "
            f"{f'{spread:.1f}' if spread is not None else '-':>8} "
            f"{f'{delta:+.1f}%' if delta is not None else '-':>8} "
            f"{f'{s[0]:.2f}' if s else '-':>10} "
            f"{f'{s[1]:.2f}' if s else '-':>10}"
        )
print("""
Law: P(skip|c) = e^(-USER_K/(c+1)) over tested chunks c in [gate, 128).
DECISION RULE: change /4 ONLY if some arm improves a k=32 cell by >=5%
with spread < delta/3, AND k=512 regresses <=0.5%, AND the adversarial
battery is green on that divisor. Predicted outcome: ties -> KEEP /4 and
file this table as the paper's gate-sensitivity ablation.""")
PYEOF

echo "DONE. Artifacts in $OUT"
