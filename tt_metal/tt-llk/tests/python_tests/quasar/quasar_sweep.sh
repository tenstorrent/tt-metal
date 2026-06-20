#!/usr/bin/env bash
# quasar_sweep.sh — Quasar backend DRIVER for the tt-polynomial-fitter.
#
# Sweeps the fitter's committed coefficients across activations, runs each on the
# craq-sim Quasar simulator (sim-qsr) via the tt-llk quasar generic-LUT tests, and
# tabulates PCC + ULP (+ per-run sim perf) per precision/format.
#
# This is the Quasar analogue of the BH/WH validate_craqsim.sh, but routed through
# the tt-llk pytest path instead of the embedded-binary run_csv.sh path.
#
# THE CONTRACT (the poly + rational quasar tests follow this):
#   each test reads env QUASAR_ACT=<activation> + QUASAR_LUT_CSV=<coeff csv>, runs on
#   sim-qsr, and prints lines `[<testname>] PCC = <number>` and (optionally)
#   `ULP = <number>` per precision/format. The pytest test-id (e.g.
#   `formats:Float32->Float32`) gives the precision for each PCC line.
#     poly test:     quasar/test_generic_lut_activation_quasar.py  (proven green)
#     rational test: quasar/test_generic_lut_rational_quasar.py    (may not exist yet)
#
# Coefficient-CSV resolution (from the fitter's best*.csv "best of metric" configs):
#   poly:     <act>_p<degree>_s<num_segments>_<segmentation>_<fitting>_<source_metric>.csv
#   rational: <act>_n<num>d<den>_s<num_segments>_<segmentation>_<fitting>_<source_metric>.csv
#   under $POLY/data/coefficients/. The fields come from best_polynomial.csv /
#   best_rational.csv columns best_<sel>_{num_segments,degree,segmentation,fitting,source_metric},
#   where <sel> is the metric selector (default: ulp).
#
# Usage:
#   ./quasar_sweep.sh --activations sigmoid,tanh [--approximation polynomial|rational|both]
#   ./quasar_sweep.sh --activations all          [--approximation both]
#   options:
#     --activations a,b,c | all      activations to sweep (all = union from best*.csv)
#     --approximation     poly|polynomial|rational|both   (default: polynomial)
#     --metric            ulp|mae|max     best-config selector       (default: ulp)
#     --precision         bf16|fp32       which best*.csv row to read (default: bf16)
#     --pcc-threshold     T               pass threshold             (default: 0.99)
#     --timeout           SECS            per-run timeout            (default: 300)
#     --results           FILE            results file               (default: /tmp/quasar_sweep_results.txt)
#
# Concurrency is capped at 1 (sim runs are heavy; ~1-3 min each under load).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTESTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"               # .../tests/python_tests
POLY="${TT_POLY_FIT_DIR:-/localdev/nkapre/tt-polynomial-fitter}"
COEFF_DIR="$POLY/data/coefficients"
TT_METAL_HOME="${TT_METAL_HOME:-/localdev/nkapre/tt-metal-nkapreTT}"
TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR:-/home/nkapre/sim-qsr/libttsim.so}"
VENV_PY="${VENV_PY:-$PYTESTS_DIR/../.venv/bin/python}"

POLY_TEST="quasar/test_generic_lut_activation_quasar.py"
RATIONAL_TEST="quasar/test_generic_lut_rational_quasar.py"

ACTS_ARG=""; APPROX="polynomial"; METRIC="ulp"; PRECROW="bf16"
PCC_THRESHOLD=0.99; RUN_TIMEOUT=300
RESULTS="${RESULTS:-/tmp/quasar_sweep_results.txt}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --activations) ACTS_ARG="$2"; shift 2 ;;
        --approximation) APPROX="$2"; shift 2 ;;
        --metric) METRIC="$2"; shift 2 ;;
        --precision) PRECROW="$2"; shift 2 ;;
        --pcc-threshold) PCC_THRESHOLD="$2"; shift 2 ;;
        --timeout) RUN_TIMEOUT="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -z "$ACTS_ARG" ]] && { echo "usage: $0 --activations a,b,c|all [--approximation polynomial|rational|both]" >&2; exit 2; }

case "$APPROX" in
    poly|polynomial) APPROXES=(polynomial) ;;
    rat|rational)    APPROXES=(rational) ;;
    both)            APPROXES=(polynomial rational) ;;
    *) echo "bad --approximation: $APPROX" >&2; exit 2 ;;
esac

[[ -x "$VENV_PY" || -f "$VENV_PY" ]] || { echo "venv python not found: $VENV_PY" >&2; exit 2; }
[[ -f "$TT_METAL_SIMULATOR" ]] || { echo "simulator not found: $TT_METAL_SIMULATOR" >&2; exit 2; }

# Build the activation list.
if [[ "$ACTS_ARG" == "all" ]]; then
    mapfile -t ACTS < <(
        { tail -n +2 "$POLY/best_polynomial.csv" 2>/dev/null; tail -n +2 "$POLY/best_rational.csv" 2>/dev/null; } \
        | cut -d, -f1 | sort -u | grep -v '^$'
    )
else
    IFS=',' read -r -a ACTS <<< "$ACTS_ARG"
fi
[[ ${#ACTS[@]} -eq 0 ]] && { echo "no activations resolved" >&2; exit 2; }

# is_degenerate <coeff-csv> -> rc 0 if the config is a degenerate identity fit (y=x), else rc 1.
# A config is degenerate when it has a SINGLE segment whose only coefficients are c0~=0,c1~=1
# (the identity y=x). For piecewise/step/fast-growth activations (relu, relu_min, threshold,
# i1, ...) the fitter's best-ULP pick is exactly this identity: ULP ignores relu's x<0 half,
# so y=x scores ULP=0 and is rated "best" even though it does not represent the activation at
# all (kernel then computes y=x vs the true-activation golden -> PCC ~0.7-0.9). We reject such
# fits so the driver instead selects a config that actually represents the activation.
is_degenerate() {
    CSV="$1" "$VENV_PY" - <<'PY'
import csv, os, sys
path = os.environ["CSV"]
segs = []
with open(path) as f:
    for r in csv.DictReader(f):
        sid = (r.get("segment_id") or "").strip()
        if sid == "" or not sid.lstrip("-").isdigit():
            continue  # skip METADATA / blank rows
        segs.append(r)
if not segs:
    sys.exit(1)  # can't tell -> treat as non-degenerate, let it run
# Degenerate == single segment that is the identity y=x (all coeffs zero except c1~=1).
if len(segs) != 1:
    sys.exit(1)
def fv(s):
    try: return float(s)
    except (TypeError, ValueError): return 0.0
s = segs[0]
ident = abs(fv(s.get("c0")) - 0.0) < 1e-6 and abs(fv(s.get("c1")) - 1.0) < 1e-6
# any higher-order coeff present and non-trivial -> not the bare identity
for k, v in s.items():
    if k and k.startswith("c") and k[1:].isdigit() and int(k[1:]) >= 2:
        if abs(fv(v)) > 1e-12:
            ident = False
sys.exit(0 if ident else 1)
PY
}

# candidate_csvs <activation> <approx> <metric> -> echoes candidate coeff-csv filenames, one per
# line, in selection-preference order. Reads the best_<m>_* columns of best_<approx>.csv for the
# requested metric first, then the other metrics as fallbacks (so a degenerate best-<metric> pick
# can be replaced by a representative best-mae/best-max one).
candidate_csvs() {
    local act="$1" approx="$2" metric="$3" bestfile="$4"
    BEST="$bestfile" ACT="$act" PREC="$PRECROW" SEL="$metric" APPROX="$approx" "$VENV_PY" - <<'PY'
import csv, os
best, act, prec, sel, approx = (os.environ[k] for k in ("BEST","ACT","PREC","SEL","APPROX"))
row = None
with open(best) as f:
    for r in csv.DictReader(f):
        if r.get("activation") == act and r.get("precision") == prec:
            row = r; break
if row is None:
    raise SystemExit
# requested metric first, then the others as fallbacks (dedup, keep order)
order = [sel] + [m for m in ("mae", "max", "ulp") if m != sel]
seen = set()
for m in order:
    def g(field, _m=m):
        return row.get(f"best_{_m}_{field}", "")
    nseg = g("num_segments") or g("segments")
    deg  = g("degree")
    segm = g("segmentation") or "uniform"
    fit  = g("fitting") or ("rational" if approx == "rational" else "any")
    src  = g("source_metric") or m
    if not (nseg and deg):
        continue
    if approx == "rational":
        num, den = (deg.split("/", 1) if "/" in deg else (deg, deg))
        core = f"n{num.strip()}d{den.strip()}"
    else:
        core = f"p{deg.strip()}"
    fname = f"{act}_{core}_s{nseg.strip()}_{segm.strip()}_{fit.strip()}_{src.strip()}.csv"
    if fname not in seen:
        seen.add(fname); print(f"{m}\t{fname}")
PY
}

# resolve_csv <activation> <polynomial|rational> -> echoes coeff-csv path (or empty), rc 0/1.
# Reads best_<approx>.csv for (activation, $PRECROW) and walks the candidate configs in metric
# preference order ($METRIC first, then mae/max/ulp), skipping any DEGENERATE identity fit. The
# first existing, non-degenerate candidate wins. If every best-* candidate is degenerate it then
# scans committed multi-segment CSVs (highest segment count first) for a representative fit. The
# chosen file and the reason are logged to stderr for transparency.
resolve_csv() {
    local act="$1" approx="$2"
    local bestfile
    if [[ "$approx" == "polynomial" ]]; then bestfile="$POLY/best_polynomial.csv"; else bestfile="$POLY/best_rational.csv"; fi
    [[ -f "$bestfile" ]] || { return 1; }

    local requested_path="" line metric fname path
    while IFS=$'\t' read -r metric fname; do
        [[ -z "$fname" ]] && continue
        path="$COEFF_DIR/$fname"
        [[ -f "$path" ]] || continue
        [[ -z "$requested_path" ]] && requested_path="$path"   # remember the as-requested pick for logging
        if is_degenerate "$path"; then
            echo "  [resolve_csv] $act/$approx: skipping DEGENERATE identity fit (metric=$metric) $fname" >&2
            continue
        fi
        if [[ "$metric" == "$METRIC" ]]; then
            echo "  [resolve_csv] $act/$approx: selected $fname (metric=$metric, as requested)" >&2
        else
            echo "  [resolve_csv] $act/$approx: selected $fname (metric=$metric; requested '$METRIC' was degenerate)" >&2
        fi
        echo "$path"; return 0
    done < <(candidate_csvs "$act" "$approx" "$METRIC" "$bestfile")

    # Every best-* candidate was degenerate (or missing). Scan committed multi-segment CSVs,
    # highest segment count first, for any non-degenerate representative fit.
    local pat
    if [[ "$approx" == "rational" ]]; then pat="${act}_n*d*_s*_*.csv"; else pat="${act}_p*_s*_*.csv"; fi
    local cand
    while read -r cand; do
        [[ -n "$cand" && -f "$cand" ]] || continue
        if ! is_degenerate "$cand"; then
            echo "  [resolve_csv] $act/$approx: best-* picks all degenerate; fell back to multi-segment $(basename "$cand")" >&2
            echo "$cand"; return 0
        fi
    done < <(ls "$COEFF_DIR"/$pat 2>/dev/null | sort -t_ -k3.2 -n -r)

    # Nothing representative exists: the fitter only ever produced an identity fit for this
    # activation. Don't fake it -- flag the fitter-side gap and skip.
    if [[ -n "$requested_path" ]]; then
        echo "  [resolve_csv] $act/$approx: NO REPRESENTATIVE poly fit (only degenerate identity configs exist); needs re-fit (fitter gap)" >&2
    fi
    return 1
}

# Pick the test file for an approximation kind. Returns rc 1 if the test source is missing.
test_for() {
    case "$1" in
        polynomial) [[ -f "$PYTESTS_DIR/$POLY_TEST" ]]     && { echo "$POLY_TEST"; return 0; } ;;
        rational)   [[ -f "$PYTESTS_DIR/$RATIONAL_TEST" ]] && { echo "$RATIONAL_TEST"; return 0; } ;;
    esac
    return 1
}

mkdir -p "$(dirname "$RESULTS")"
: > "$RESULTS"

HDR=$(printf '%-14s %-11s %-12s %-12s %-10s %-9s %-7s' ACTIVATION APPROX PRECISION PCC ULP CYCLES STATUS)
SEP="------------------------------------------------------------------------------------"
echo "$HDR" | tee -a "$RESULTS"
echo "$SEP" | tee -a "$RESULTS"

pass_count=0; fail_count=0; total_rows=0

emit() {  # activation approx precision pcc ulp cycles status
    local line
    line=$(printf '%-14s %-11s %-12s %-12s %-10s %-9s %-7s' "$1" "$2" "$3" "$4" "$5" "$6" "$7")
    echo "$line" | tee -a "$RESULTS"
}

for act in "${ACTS[@]}"; do
    for approx in "${APPROXES[@]}"; do
        tf=$(test_for "$approx") || { emit "$act" "$approx" "-" "-" "-" "-" "SKIP(no-test)"; continue; }
        csv=$(resolve_csv "$act" "$approx") || { emit "$act" "$approx" "-" "-" "-" "-" "SKIP(no-csv)"; continue; }

        log="$(mktemp /tmp/qsweep_${act}_${approx}_XXXX.log)"
        timeout "$RUN_TIMEOUT" env \
            TT_METAL_HOME="$TT_METAL_HOME" \
            TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR" \
            CHIP_ARCH=quasar \
            QUASAR_ACT="$act" \
            QUASAR_LUT_CSV="$csv" \
            "$VENV_PY" -m pytest --run-simulator "$PYTESTS_DIR/$tf" -x -s -q \
            >"$log" 2>&1
        rc=$?

        if [[ $rc -eq 124 ]]; then
            emit "$act" "$approx" "-" "-" "-" "-" "TIMEOUT"
            fail_count=$((fail_count+1)); total_rows=$((total_rows+1)); continue
        fi

        # Per-run sim perf: craq-sim prints a final "[NNNN] X.X seconds (Y KHz)" line.
        # This covers the whole process (all variants in this test), so it is a
        # per-run aggregate, not strictly per-variant; reported in the CYCLES column
        # as "Ys/ZKHz".
        perf=$(grep -oE '\[[0-9]+\] [0-9.]+ seconds \([0-9.]+ KHz\)' "$log" | tail -1)
        if [[ -n "$perf" ]]; then
            secs=$(echo "$perf" | sed -E 's/.* ([0-9.]+) seconds.*/\1/')
            khz=$(echo "$perf" | sed -E 's/.*\(([0-9.]+) KHz\).*/\1/')
            perfcol="${secs}s/${khz}K"
        else
            perfcol="n/a"
        fi

        # Parse "[<testname>] PCC = <n>" lines, each preceded by a pytest test-id line
        # carrying "formats:<IN>-><OUT>" that gives the precision. Also parse optional
        # "ULP = <n>" lines (contract; may be absent if the test doesn't emit them yet).
        n_rows_before=$total_rows
        while IFS=$'\t' read -r prec pcc ulp; do
            [[ -z "$prec$pcc$ulp" ]] && continue
            status="FAIL"
            if [[ "$pcc" != "-" ]] && awk "BEGIN{exit !($pcc >= $PCC_THRESHOLD)}" 2>/dev/null; then
                status="pass"; pass_count=$((pass_count+1))
            else
                fail_count=$((fail_count+1))
            fi
            emit "$act" "$approx" "$prec" "$pcc" "${ulp:--}" "$perfcol" "$status"
            total_rows=$((total_rows+1))
        done < <(
            # Emit-pending parser: the contract allows the "ULP = " line to appear
            # either just before or just after its "[<test>] PCC = " line (same
            # precision block). We buffer one pending (precision, PCC) and flush it
            # when the NEXT PCC arrives or at END, attaching any ULP seen in between.
            awk '
                function flush() {
                    if (have) {
                        printf "%s\t%s\t%s\n", (prec=="" ? "-" : prec), pcc, (ulp=="" ? "-" : ulp)
                    }
                }
                /formats:[A-Za-z0-9_]+->/ {
                    if (match($0, /formats:[A-Za-z0-9_]+->[A-Za-z0-9_]+/)) {
                        nextprec = substr($0, RSTART+8, RLENGTH-8)
                    }
                }
                /PCC = / {
                    flush()
                    p = $0; sub(/.*PCC = /, "", p); sub(/[^0-9eE.+-].*$/, "", p)
                    pcc = p; prec = nextprec; ulp = ""; have = 1
                }
                /ULP = / {
                    u = $0; sub(/.*ULP = /, "", u); sub(/[^0-9eE.+-].*$/, "", u)
                    ulp = u
                }
                END { flush() }
            ' "$log"
        )

        # No PCC lines parsed at all -> run failed before printing, or crashed.
        if [[ $total_rows -eq $n_rows_before ]]; then
            emit "$act" "$approx" "-" "-" "-" "$perfcol" "FAIL(rc=$rc)"
            fail_count=$((fail_count+1)); total_rows=$((total_rows+1))
            echo "  (see $log)" | tee -a "$RESULTS"
        else
            rm -f "$log"
        fi
    done
done

echo "$SEP" | tee -a "$RESULTS"
SUMMARY=$(printf 'SUMMARY: %d row(s) | %d pass | %d fail | threshold PCC>=%s | results -> %s' \
    "$total_rows" "$pass_count" "$fail_count" "$PCC_THRESHOLD" "$RESULTS")
echo "$SUMMARY" | tee -a "$RESULTS"

[[ $fail_count -eq 0 ]] && exit 0 || exit 1
