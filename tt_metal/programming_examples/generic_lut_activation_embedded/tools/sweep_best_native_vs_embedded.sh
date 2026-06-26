#!/bin/bash
# =============================================================================
# sweep_best_native_vs_embedded.sh — Batch NATIVE-vs-OURS over a best*.csv
#
# Resolves every (activation, precision) row of a tt-polynomial-fitter best*.csv
# to its best-ULP coefficient CSV, then runs compare_native_vs_embedded.sh for
# each — producing a full ULP + runtime table of TTNN native vs our LUT kernel.
#
# Works with best.csv, best95.csv (cheapest within 95% of peak accuracy),
# best99.csv, etc. — the selection policy lives in the CSV, this just drives it.
#
# Resolution is by HEADER NAME (not fixed column indices) and uses the
# best_ulp_source_metric column for the filename suffix, so it stays correct
# across best.csv schema changes (the bug that broke pr_submission.sh).
#
# Usage:
#   ./sweep_best_native_vs_embedded.sh --best-csv $TT_POLY_FIT_DIR/best95.csv \
#       [--precision both|bf16|fp32] [--activations tanh,asin,gelu] [--tiles N] \
#       [--out results.txt] [--csv-out results.csv] [--dump-dir dumps] \
#       [--shard I --num-shards N]
#
# Output: a table to stdout + text/CSV result files. The CSV is resumable and
# shardable for multi-chip dispatch. Use one checkout per chip because run_csv.sh
# regenerates the checkout-local kernels/compute/adhoc/adhoc.cpp.
#
# Requires: python_env (ttnn), /usr/bin/python3 + numpy, TT_POLY_FIT_DIR,
#           the adhoc target built (see README "Setup & Build").
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${TT_POLY_FIT_DIR:-}" ]]; then
  for c in "$SCRIPT_DIR/../../../../../tt-polynomial-fitter" "$HOME/tt-polynomial-fitter" "$HOME/workspace/tt-polynomial-fitter" "/localdev/$USER/tt-polynomial-fitter"; do
    [[ -d "$c/data/coefficients" ]] && { TT_POLY_FIT_DIR="$(cd "$c" && pwd)"; break; }
  done
fi
TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/$USER/tt-polynomial-fitter}"
COEFF_DIR="$TT_POLY_FIT_DIR/data/coefficients"
COMPARE_SCRIPT="$SCRIPT_DIR/compare_native_vs_embedded.sh"

BEST_CSV=""; PREC_FILTER="both"; ACT_FILTER=""; TILES=256
OUT=""; CSV_OUT=""; DUMP_DIR=""; RUN_DIR=""; SHARD=0; NUM_SHARDS=1; CACHE=""; PER_CFG_TIMEOUT=900
while [[ $# -gt 0 ]]; do
  case "$1" in
    --best-csv|-b)    BEST_CSV="$2"; shift 2 ;;
    --precision|-p)   PREC_FILTER="$2"; shift 2 ;;
    --activations|-a) ACT_FILTER="$2"; shift 2 ;;
    --tiles|-t)       TILES="$2"; shift 2 ;;
    --out)            OUT="$2"; shift 2 ;;
    --csv-out)        CSV_OUT="$2"; shift 2 ;;
    --dump-dir)       DUMP_DIR="$2"; shift 2 ;;
    --run-dir)        RUN_DIR="$2"; shift 2 ;;
    --shard)          SHARD="$2"; shift 2 ;;
    --num-shards)     NUM_SHARDS="$2"; shift 2 ;;
    --cache)          CACHE="$2"; shift 2 ;;
    --timeout)        PER_CFG_TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[[ -z "$BEST_CSV" ]] && BEST_CSV="$TT_POLY_FIT_DIR/best.csv"
[[ ! -f "$BEST_CSV" ]] && { echo "Error: best csv not found: $BEST_CSV"; exit 1; }
[[ ! -x "$COMPARE_SCRIPT" ]] && { echo "Error: missing compare helper: $COMPARE_SCRIPT"; exit 1; }
if [[ "$NUM_SHARDS" -lt 1 || "$SHARD" -lt 0 || "$SHARD" -ge "$NUM_SHARDS" ]]; then
  echo "Error: invalid shard $SHARD/$NUM_SHARDS" >&2
  exit 1
fi
if [[ -n "$CACHE" ]]; then
  export TT_METAL_CACHE="$CACHE"
fi

stem="$(basename "${BEST_CSV%.csv}")"
WL="$(mktemp /tmp/sweep_wl_XXXX.txt)"
DEFAULT_RUN_DIR="$SCRIPT_DIR/../results/native_vs_embedded/${PREC_FILTER}"
if [[ -n "$RUN_DIR" ]]; then
  [[ -z "$OUT" ]] && OUT="$RUN_DIR/logs/worker/${stem}_chip${SHARD}.txt"
  [[ -z "$CSV_OUT" ]] && CSV_OUT="$RUN_DIR/data/csv/shards/chip${SHARD}/${stem}_chip${SHARD}.csv"
  [[ -z "$DUMP_DIR" ]] && DUMP_DIR="$RUN_DIR/data/dumps"
fi
[[ -z "$OUT" ]] && OUT="$DEFAULT_RUN_DIR/logs/worker/${stem}_chip${SHARD}.txt"
[[ -z "$CSV_OUT" ]] && CSV_OUT="$DEFAULT_RUN_DIR/data/csv/shards/chip${SHARD}/${stem}_chip${SHARD}.csv"
trap 'rm -f "$WL"' EXIT
mkdir -p "$(dirname "$OUT")" "$(dirname "$CSV_OUT")"
if [[ -n "$DUMP_DIR" ]]; then
  mkdir -p "$DUMP_DIR"
fi
if [[ ! -f "$CSV_OUT" ]]; then
  echo "activation,precision,config_csv,native_maxulp,native_meanulp,native_us,native_pure,native_ml,ours_maxulp,ours_meanulp,ours_us,ours_pure,ours_ml,range_min,range_max,tiles,status" > "$CSV_OUT"
fi

# Resolve best_ulp coeff filename per row, by header name + source_metric.
python3 - "$BEST_CSV" "$COEFF_DIR" "$PREC_FILTER" "$ACT_FILTER" > "$WL" <<'PYEOF'
import csv, os, sys
best_csv, coeff_dir, prec_filter, act_filter = sys.argv[1:5]
acts = set(a for a in act_filter.split(",") if a) if act_filter else None
rows = list(csv.reader(open(best_csv))); h = rows[0]; idx = {n: i for i, n in enumerate(h)}
need = ["best_ulp_degree","best_ulp_num_segments","best_ulp_segmentation","best_ulp_fitting","best_ulp_source_metric"]
if any(n not in idx for n in need):
    sys.stderr.write("best csv missing best_ulp_* columns (regenerate with best_all.sh)\n"); sys.exit(1)
hit = miss = 0
for r in rows[1:]:
    act, prec = r[0], r[1]
    if prec_filter != "both" and prec != prec_filter: continue
    if acts and act not in acts: continue
    deg = r[idx["best_ulp_degree"]]; segs = r[idx["best_ulp_num_segments"]]
    seg = r[idx["best_ulp_segmentation"]]; fit = r[idx["best_ulp_fitting"]]
    sm = r[idx["best_ulp_source_metric"]]
    approx = f"n{deg.replace('/','d')}" if "/" in deg else f"p{deg}"
    fname = f"{act}_{approx}_s{segs}_{seg}_{fit}_{sm}.csv"
    if os.path.exists(os.path.join(coeff_dir, fname)):
        print(f"{act},{prec},{fname}"); hit += 1
    else:
        sys.stderr.write(f"  MISS {act} {prec} -> {fname}\n"); miss += 1
sys.stderr.write(f"resolved {hit} / {hit+miss}\n")
PYEOF

echo "=== sweep $stem ($PREC_FILTER) :: $(grep -c . "$WL") configs ==="
echo "=== shard $SHARD/$NUM_SHARDS chip=${TT_VISIBLE_DEVICES:-unset} cache=${TT_METAL_CACHE:-unset} ==="
touch "$OUT"
n=0; ran=0; skipped=0; total=$(grep -c . "$WL")
while IFS=, read -r act prec csv; do
  [[ -z "$act" ]] && continue
  idx=$n
  n=$((n+1)); echo "[$n/$total] $act $prec" >&2
  [[ $(( idx % NUM_SHARDS )) -eq "$SHARD" ]] || continue
  if grep -q "^${act},${prec},${csv}," "$CSV_OUT" 2>/dev/null; then
    skipped=$((skipped + 1))
    continue
  fi
  set +e
  dump_args=()
  if [[ -n "$DUMP_DIR" ]]; then
    dump_args=(--dump-dir "$DUMP_DIR/shard${SHARD}")
  fi
  timeout "$PER_CFG_TIMEOUT" "$COMPARE_SCRIPT" \
     --activation "$act" --precision "$prec" --csv "$COEFF_DIR/$csv" \
     --tiles "$TILES" --csv-out "$CSV_OUT" "${dump_args[@]}" \
     2>/dev/null | grep -E '\| NATIVE' | tee -a "$OUT"
  rc=${PIPESTATUS[0]}
  set +e
  if [[ "$rc" -ne 0 ]]; then
    if [[ -n "$RUN_DIR" ]]; then
      fail_dir="$RUN_DIR/logs/failure/shard${SHARD}"
    else
      fail_dir="$(dirname "$CSV_OUT")/failure_logs/shard${SHARD}"
    fi
    mkdir -p "$fail_dir"
    echo "compare_native_vs_embedded failed rc=$rc activation=$act precision=$prec csv=$csv" \
      > "$fail_dir/${act}_${prec}_${csv}.log"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$act" "$prec" "$csv" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "" "" "$TILES" "script_fail_rc_${rc}" >> "$CSV_OUT"
  fi
  ran=$((ran + 1))
done < "$WL"
echo "=== done shard $SHARD/$NUM_SHARDS: ran=$ran skipped=$skipped -> $OUT ; $CSV_OUT ==="
