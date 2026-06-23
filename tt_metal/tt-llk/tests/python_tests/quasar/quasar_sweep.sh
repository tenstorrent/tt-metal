#!/usr/bin/env bash
# quasar_sweep.sh — Quasar DEPLOYMENT validator for the tt-polynomial-fitter.
#
# Sweeps the fitter's ACTUALLY-DEPLOYED pick per activation, runs each on the
# craq-sim Quasar simulator (sim-qsr) via the tt-llk quasar generic-LUT tests, and
# tabulates PCC + ULP (+ per-run sim perf) per precision/format.
#
# This is the Quasar analogue of the BH/WH validate_craqsim.sh, but routed through
# the tt-llk pytest path instead of the embedded-binary run_csv.sh path.
#
# THE DEPLOYMENT TRUTH (best.csv): for each (activation, precision) the fitter records
# the TRUE deployed pick in columns best_<sel>_{fitting,degree,num_segments,...}
# (<sel> = the metric selector, default ulp). That pick is the good, shipping config:
#   best_ulp_fitting  = rational | any | fpminimax | ...
#   best_ulp_degree   = "8" (poly) | "6/6" (rational num/den)
#   best_ulp_num_segments, best_ulp_segmentation, best_ulp_source
# We test EXACTLY that pick -- no per-category guessing, no degeneracy/hopelessness
# guards, no oversized fallbacks. If the deployed pick is rational, we use the rational
# kind; if it has degree "n/m" (or fitting=="rational") it is rational, else polynomial.
# (identity's deployed pick is y=x (p1 s1) -- which is CORRECT for identity, PCC 1.0.)
#
# THE CONTRACT (the poly + rational quasar tests follow this):
#   each test reads env QUASAR_ACT=<activation> + QUASAR_LUT_CSV=<coeff csv>, runs on
#   sim-qsr, and prints lines `[<testname>] PCC = <number>` and (optionally)
#   `ULP = <number>` per precision/format. The pytest test-id (e.g.
#   `formats:Float32->Float32`) gives the precision for each PCC line.
#     poly test:     quasar/test_generic_lut_activation_quasar.py
#     rational test: quasar/test_generic_lut_rational_quasar.py
#
# Coefficient-CSV resolution is a TOLERANT GLOB (the fitter's recorded config does not
# always have an exactly-named CSV -- e.g. log records p8_s1 but only log_p8_s2_* exists).
# Given core ("p<deg>" or "n<num>d<den>") + segs we try, first hit wins:
#   (a) {act}_{core}_s{segs}_*.csv   exact
#   (b) {act}_{core}_s*_*.csv        same degree, any segment count (fewest first)
#   (c) {act}_{core}*_*.csv          same degree, loosest
# under $POLY/data/coefficients/. No exact-name reconstruction, no guards.
#
# Modes:
#   best  (DEFAULT)            -- DEPLOYMENT sweep. One config per activation from best.csv
#                                (rational or poly per the deployed pick). This is what ships.
#   polynomial|rational|both  -- opt-in COMPARISON. Reads best_polynomial.csv / best_rational.csv
#                                (per-CATEGORY pick), same tolerant glob. If a category's pick is
#                                weak it just reports its real PCC -- honest, not patched.
#
# Usage:
#   ./quasar_sweep.sh --activations all                         # deployment sweep (default = best)
#   ./quasar_sweep.sh --activations gelu,exp,tanh               # subset, deployment pick each
#   ./quasar_sweep.sh --activations all --approximation both    # opt-in poly+rational comparison
#   options:
#     --activations a,b,c | all      activations to sweep (all = union from best.csv)
#     --approximation best|poly|polynomial|rational|both   (default: best = deployment sweep)
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
LLK_TESTS_DIR="$(cd "$PYTESTS_DIR/.." && pwd)"            # .../tt-llk/tests
# Repo root derived from THIS script's path -- portable, no hardcoded machine dirs.
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$PYTESTS_DIR/../../../.." && pwd)}"
POLY="${TT_POLY_FIT_DIR:-/localdev/nkapre/tt-polynomial-fitter}"
COEFF_DIR="$POLY/data/coefficients"
# Pinned craq-sim Quasar sim (external build; override TT_METAL_SIMULATOR per machine).
TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR:-$(cd "$TT_METAL_HOME/.." 2>/dev/null && pwd)/craq-sim-quasar/src/_out/release_qsr/libttsim.so}"

# Python autodetect: canonical tt-llk tests/.venv first, then repo python_env, then python3.
# NO /tmp, no stray machine paths. Override with VENV_PY. Create the venv with:
#   bash tt_metal/tt-llk/tests/setup_external_testing_env.sh   (installs requirements.txt -> tt-exalens 0.3.20)
_py_ok() { [[ -x "$1" || "$1" == python3 ]] || return 1
    "$1" -c "import torch, numpy; from ttexalens.tt_exalens_lib import ParsedElfFile" >/dev/null 2>&1; }
VENV_PY="${VENV_PY:-}"
for _cand in "$VENV_PY" "$LLK_TESTS_DIR/.venv/bin/python" "$TT_METAL_HOME/python_env/bin/python" "python3"; do
    [[ -z "$_cand" ]] && continue
    if _py_ok "$_cand"; then VENV_PY="$_cand"; break; fi
done

POLY_TEST="quasar/test_generic_lut_activation_quasar.py"
RATIONAL_TEST="quasar/test_generic_lut_rational_quasar.py"

ACTS_ARG=""; APPROX="best"; METRIC="ulp"; PRECROW="bf16"
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
        -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -z "$ACTS_ARG" ]] && { echo "usage: $0 --activations a,b,c|all [--approximation best|polynomial|rational|both]" >&2; exit 2; }

# MODE selects which best*.csv drives resolution and which category column(s) we read.
#   best  -> best.csv, kind decided per-row from the deployed fitting/degree (poly or rational)
#   poly  -> best_polynomial.csv, forced polynomial kind
#   rat   -> best_rational.csv,   forced rational kind
#   both  -> poly then rational comparison passes
case "$APPROX" in
    best)            PASSES=(best) ;;
    poly|polynomial) PASSES=(polynomial) ;;
    rat|rational)    PASSES=(rational) ;;
    both)            PASSES=(polynomial rational) ;;
    *) echo "bad --approximation: $APPROX (best|polynomial|rational|both)" >&2; exit 2 ;;
esac

[[ -n "$VENV_PY" ]] && _py_ok "$VENV_PY" || { echo "ERROR: no python with the tt-llk harness deps (torch, numpy, ttexalens.ParsedElfFile)." >&2
    echo "       Create it:  bash $LLK_TESTS_DIR/setup_external_testing_env.sh   (-> tests/.venv, tt-exalens 0.3.20)" >&2
    echo "       or export VENV_PY=<interpreter>." >&2; exit 2; }
[[ -f "$TT_METAL_SIMULATOR" ]] || { echo "ERROR: simulator not found: $TT_METAL_SIMULATOR" >&2
    echo "       Build the pinned craq-sim Quasar sim or export TT_METAL_SIMULATOR=<libttsim.so>." >&2; exit 2; }

# Build the activation list (all = union from best.csv).
if [[ "$ACTS_ARG" == "all" ]]; then
    mapfile -t ACTS < <(tail -n +2 "$POLY/best.csv" 2>/dev/null | cut -d, -f1 | sort -u | grep -v '^$')
else
    IFS=',' read -r -a ACTS <<< "$ACTS_ARG"
fi
[[ ${#ACTS[@]} -eq 0 ]] && { echo "no activations resolved" >&2; exit 2; }

# resolve_deployed <activation> <bestfile> <forced-kind|""> -> echoes "<kind>\t<coeff-csv-path>", rc 0/1.
# Reads best_<METRIC>_{fitting,degree,num_segments} for (activation, $PRECROW) from <bestfile>.
#   kind: forced-kind if given (poly|rational, for the comparison modes); else inferred --
#         rational iff fitting=="rational" OR degree contains "/", else polynomial.
#   core: poly -> "p<deg>"; rational -> "n<num>d<den>" (split degree on "/").
# Resolves the coeff CSV by TOLERANT GLOB under $COEFF_DIR (exact segs, then same-degree any
# segs fewest-first, then loosest). First hit wins. No exact-name reconstruction, no guards:
# the deployed pick is the good config, so if it does not resolve we log + skip (rc 1) rather
# than fall back to an oversized/degenerate config.
resolve_deployed() {
    local act="$1" bestfile="$2" forced="$3"
    [[ -f "$bestfile" ]] || { echo "  [resolve] $act: best file not found: $bestfile" >&2; return 1; }

    # Pull deployed (kind, core, segs) from the best file.
    local spec
    spec="$(ACT="$act" BEST="$bestfile" PREC="$PRECROW" SEL="$METRIC" FORCED="$forced" "$VENV_PY" - <<'PY'
import csv, os
act, best, prec, sel, forced = (os.environ[k] for k in ("ACT","BEST","PREC","SEL","FORCED"))
row = None
with open(best) as f:
    for r in csv.DictReader(f):
        if r.get("activation") == act and r.get("precision") == prec:
            row = r; break
if row is None:
    raise SystemExit  # no row -> empty output -> caller skips
def g(field):
    return (row.get(f"best_{sel}_{field}") or "").strip()
deg  = g("degree")
nseg = g("num_segments") or g("segments")
fit  = g("fitting")
if not (deg and nseg):
    raise SystemExit
# Decide kind: forced (comparison modes) wins; else infer from the deployed pick.
if forced in ("poly", "polynomial"):
    kind = "polynomial"
elif forced == "rational":
    kind = "rational"
else:
    kind = "rational" if (fit == "rational" or "/" in deg) else "polynomial"
if kind == "rational":
    num, den = (deg.split("/", 1) if "/" in deg else (deg, deg))
    core = f"n{num.strip()}d{den.strip()}"
else:
    core = f"p{deg.strip()}"
print(f"{kind}\t{core}\t{nseg}")
PY
)"
    [[ -z "$spec" ]] && { echo "  [resolve] $act: no deployed $(basename "$bestfile") pick for ($act,$PRECROW,metric=$METRIC)" >&2; return 1; }

    local kind core segs
    IFS=$'\t' read -r kind core segs <<< "$spec"

    # Tolerant glob: exact segs, then same-degree any segs (fewest first), then loosest.
    local path=""
    local g
    # (a) exact
    for g in "$COEFF_DIR/${act}_${core}_s${segs}_"*.csv; do
        [[ -f "$g" ]] && { path="$g"; break; }
    done
    # (b) same degree, any segment count -- pick FEWEST segments first (closest to deployed
    #     intent, smallest LUT) so log p8_s1 -> log_p8_s2 (not log_p8_s16).
    if [[ -z "$path" ]]; then
        path="$(ls -1 "$COEFF_DIR/${act}_${core}_s"*_*.csv 2>/dev/null \
            | sed -E 's@.*/'"${act}_${core}"'_s([0-9]+)_.*@\1 &@' \
            | sort -n -k1,1 | head -1 | cut -d' ' -f2-)"
    fi
    # (c) loosest
    if [[ -z "$path" ]]; then
        for g in "$COEFF_DIR/${act}_${core}"*_*.csv; do
            [[ -f "$g" ]] && { path="$g"; break; }
        done
    fi

    if [[ -z "$path" || ! -f "$path" ]]; then
        echo "  [resolve] $act: no coeff CSV for deployed pick $core s$segs (kind=$kind) under $COEFF_DIR" >&2
        return 1
    fi
    echo "  [resolve] $act: $kind deployed pick $core s$segs -> $(basename "$path")" >&2
    printf '%s\t%s\n' "$kind" "$path"
    return 0
}

# Pick the test file for a kind. Returns rc 1 if the test source is missing.
test_for() {
    case "$1" in
        polynomial) [[ -f "$PYTESTS_DIR/$POLY_TEST" ]]     && { echo "$POLY_TEST"; return 0; } ;;
        rational)   [[ -f "$PYTESTS_DIR/$RATIONAL_TEST" ]] && { echo "$RATIONAL_TEST"; return 0; } ;;
    esac
    return 1
}

mkdir -p "$(dirname "$RESULTS")"
: > "$RESULTS"

HDR=$(printf '%-14s %-11s %-12s %-12s %-10s %-9s %-7s' ACTIVATION MODE PRECISION PCC ULP CYCLES STATUS)
SEP="------------------------------------------------------------------------------------"
echo "$HDR" | tee -a "$RESULTS"
echo "$SEP" | tee -a "$RESULTS"

pass_count=0; fail_count=0; total_rows=0

emit() {  # activation mode precision pcc ulp cycles status
    local line
    line=$(printf '%-14s %-11s %-12s %-12s %-10s %-9s %-7s' "$1" "$2" "$3" "$4" "$5" "$6" "$7")
    echo "$line" | tee -a "$RESULTS"
}

for act in "${ACTS[@]}"; do
    for pass in "${PASSES[@]}"; do
        # Choose the best file + forced kind for this pass.
        case "$pass" in
            best)       bestfile="$POLY/best.csv";            forced="" ;;
            polynomial) bestfile="$POLY/best_polynomial.csv"; forced="poly" ;;
            rational)   bestfile="$POLY/best_rational.csv";   forced="rational" ;;
        esac

        spec=$(resolve_deployed "$act" "$bestfile" "$forced") \
            || { emit "$act" "$pass" "-" "-" "-" "-" "SKIP(no-csv)"; continue; }
        IFS=$'\t' read -r kind csv <<< "$spec"

        tf=$(test_for "$kind") || { emit "$act" "$pass" "-" "-" "-" "-" "SKIP(no-test)"; continue; }

        log="$(mktemp /tmp/qsweep_${act}_${pass}_XXXX.log)"
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
            emit "$act" "$pass" "-" "-" "-" "-" "TIMEOUT"
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
            emit "$act" "$pass" "$prec" "$pcc" "${ulp:--}" "$perfcol" "$status"
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
            emit "$act" "$pass" "-" "-" "-" "$perfcol" "FAIL(rc=$rc)"
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
