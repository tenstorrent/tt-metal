#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# run_quasar.sh — the SINGLE entry point for the Quasar generic-LUT eltwise flow.
#
# For one (activation, eval_method) it:
#   1. builds the config  : maps eval_method -> the tt-llk Quasar test source +
#                           the per-method env-var contract (CSV var, act var),
#   2. (optional) compiles : SFPI compile-only sanity gate via compile_llk_quasar.sh,
#   3. runs under the sim  : pytest --run-simulator against the pinned craq-sim
#                           Quasar ttsim build, with TT_METAL_SLOW_DISPATCH_MODE=1,
#   4. compares to golden  : the test prints "[<test>] PCC = <n>" / "ULP = <n>"
#                           (the tt-polynomial-fitter ground_truth golden); this
#                           script extracts and pass/fail-gates them.
#
# The actual build+run is owned by the tt-llk pytest harness
# (tt_metal/tt-llk/tests/python_tests/helpers/test_config.py + conftest.py); this
# script is the thin, documented driver that wires the pinned sim + the per-method
# env contract. For a multi-activation SWEEP across the fitter's best configs, use
# tt_metal/tt-llk/tests/python_tests/quasar/quasar_sweep.sh instead.
#
# Usage:
#   ./run_quasar.sh [-m EVAL_METHOD] [-a ACTIVATION] [-c CSV] [--compile-only]
#                   [--no-compile] [-t PCC] [-- EXTRA_PYTEST_ARGS...]
#
#   -m EVAL_METHOD   polynomial | rational | parity | expalu | newton_root
#                    (default: polynomial)
#   -a ACTIVATION    activation name for the ground_truth golden
#                    (default: gelu; newton_root default: sqrt)
#   -c CSV           coefficient CSV (tt-polynomial-fitter). Optional: each test
#                    has a sensible built-in default / skip if omitted.
#   --compile-only   only run the SFPI compile gate, then stop.
#   --no-compile     skip the compile gate, go straight to the sim run.
#   -t PCC           pass threshold for PCC (default: 0.99).
#   -h, --help       show this help.
#
# Pinned environment (override by exporting before calling):
#   TT_METAL_HOME        (default: /localdev/nkapre/tt-metal)
#   TT_METAL_SIMULATOR   (default: /localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so)
#   TT_METAL_SLOW_DISPATCH_MODE=1, CHIP_ARCH=quasar  (forced)
#   VENV_PY              python with the tt-llk test deps (auto-detected)
#
# Examples:
#   ./run_quasar.sh -m polynomial -a gelu
#   ./run_quasar.sh -m rational   -a atanh -c /path/atanh_n8d8_s3_..._rational.csv
#   ./run_quasar.sh -m newton_root -a sqrt
#   ./run_quasar.sh -m parity -a tanh --compile-only

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives at $TT_METAL_HOME/tt_metal/tt-llk/tests/quasar_eltwise/ —
# derive the repo root from it so the flow is self-contained in the tt-llk tree.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# --- Pinned env (overridable) ---------------------------------------------
TT_METAL_HOME="${TT_METAL_HOME:-$REPO_ROOT}"
TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR:-/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so}"
LLK_TESTS_DIR="$TT_METAL_HOME/tt_metal/tt-llk/tests"
PYTESTS_DIR="$LLK_TESTS_DIR/python_tests"

# Python autodetect. The tt-llk harness needs a ttexalens new enough to expose
# ParsedElfFile (older envs like tt-metal/python_env fail conftest import). Pick
# the first candidate that imports the harness deps cleanly. Override with VENV_PY.
_py_ok() {  # $1 = python interpreter
    [[ -x "$1" || "$1" == python3 ]] || return 1
    "$1" -c "import torch, numpy; from ttexalens.tt_exalens_lib import ParsedElfFile" \
        >/dev/null 2>&1
}
PY=""
for cand in \
    "${VENV_PY:-}" \
    "$PYTESTS_DIR/../.venv/bin/python" \
    "/home/$USER/.local/bin/python" \
    "$TT_METAL_HOME/python_env/bin/python" \
    "python3"; do
    [[ -z "$cand" ]] && continue
    if _py_ok "$cand"; then PY="$cand"; break; fi
done
if [[ -z "$PY" ]]; then
    echo "ERROR: no python with the tt-llk harness deps (torch, numpy, ttexalens" >&2
    echo "       with ParsedElfFile) found. Set VENV_PY to a suitable interpreter." >&2
    exit 1
fi

# --- Args ------------------------------------------------------------------
EVAL_METHOD="polynomial"
ACTIVATION=""
CSV=""
DO_COMPILE=1
DO_RUN=1
PCC_THRESHOLD="0.99"
EXTRA_PYTEST=()

usage() { sed -n '2,55p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--method)      EVAL_METHOD="$2"; shift 2 ;;
        -a|--activation)  ACTIVATION="$2";  shift 2 ;;
        -c|--csv)         CSV="$2";         shift 2 ;;
        --compile-only)   DO_COMPILE=1; DO_RUN=0; shift ;;
        --no-compile)     DO_COMPILE=0; shift ;;
        -t|--threshold)   PCC_THRESHOLD="$2"; shift 2 ;;
        -h|--help)        usage ;;
        --)               shift; EXTRA_PYTEST+=("$@"); break ;;
        *)                echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# --- Map eval_method -> (test file, CSV env var, default activation) -------
case "$EVAL_METHOD" in
    poly|polynomial) EVAL_METHOD="polynomial"
        TEST="quasar/test_generic_lut_activation_quasar.py"; CSV_VAR="QUASAR_LUT_CSV"; DEF_ACT="gelu" ;;
    rat|rational)    EVAL_METHOD="rational"
        TEST="quasar/test_generic_lut_rational_quasar.py";   CSV_VAR="QUASAR_LUT_CSV"; DEF_ACT="atanh" ;;
    parity)
        TEST="quasar/test_generic_lut_parity_quasar.py";     CSV_VAR="QUASAR_LUT_CSV"; DEF_ACT="tanh" ;;
    expalu|exponent_alu) EVAL_METHOD="expalu"
        TEST="quasar/test_generic_lut_expalu_quasar.py";     CSV_VAR="QUASAR_LUT_CSV"; DEF_ACT="sigmoid" ;;
    newton_root|newton|root) EVAL_METHOD="newton_root"
        TEST="quasar/test_generic_lut_newton_root_quasar.py"; CSV_VAR="QUASAR_NR_CSV"; DEF_ACT="sqrt" ;;
    *) echo "ERROR: unknown eval_method '$EVAL_METHOD'" >&2
       echo "       expected: polynomial | rational | parity | expalu | newton_root" >&2
       exit 2 ;;
esac
[[ -z "$ACTIVATION" ]] && ACTIVATION="$DEF_ACT"

echo "=============================================="
echo " Quasar generic-LUT flow — single entry point"
echo "=============================================="
echo "  eval_method : $EVAL_METHOD"
echo "  activation  : $ACTIVATION"
echo "  test        : $TEST"
echo "  csv         : ${CSV:-<built-in default / skip>}  (env: $CSV_VAR)"
echo "  simulator   : $TT_METAL_SIMULATOR"
echo "  python      : $PY"
echo ""

if [[ ! -f "$PYTESTS_DIR/$TEST" ]]; then
    echo "ERROR: test source not found: $PYTESTS_DIR/$TEST" >&2
    exit 1
fi

# --- 1+2. Build config + SFPI compile gate ---------------------------------
if [[ "$DO_COMPILE" -eq 1 ]]; then
    echo ">>> SFPI compile gate ($EVAL_METHOD)"
    if ! TT_METAL_HOME="$TT_METAL_HOME" bash "$SCRIPT_DIR/compile_llk_quasar.sh" "$EVAL_METHOD" "$ACTIVATION"; then
        echo "ERROR: compile gate failed for $EVAL_METHOD" >&2
        exit 1
    fi
    echo ""
fi

[[ "$DO_RUN" -eq 0 ]] && { echo "--compile-only: stopping after compile gate."; exit 0; }

# --- 3. Run under the pinned sim -------------------------------------------
if [[ ! -f "$TT_METAL_SIMULATOR" ]]; then
    echo "ERROR: simulator not found: $TT_METAL_SIMULATOR" >&2
    echo "       export TT_METAL_SIMULATOR=<path to libttsim.so> and retry." >&2
    exit 1
fi

CSV_ENV=()
[[ -n "$CSV" ]] && CSV_ENV=("$CSV_VAR=$CSV")

LOG="$(mktemp /tmp/run_quasar_${EVAL_METHOD}_${ACTIVATION}_XXXX.log)"
echo ">>> Running on sim (pytest --run-simulator)"
echo "    log: $LOG"
echo ""

( cd "$PYTESTS_DIR" && env \
    TT_METAL_HOME="$TT_METAL_HOME" \
    TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR" \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    CHIP_ARCH=quasar \
    QUASAR_ACT="$ACTIVATION" \
    "${CSV_ENV[@]}" \
    "$PY" -m pytest --run-simulator "$TEST" -x -s -q "${EXTRA_PYTEST[@]+"${EXTRA_PYTEST[@]}"}" \
) 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

# --- 4. Compare to golden (parse the contract PCC / ULP lines) -------------
echo ""
echo "----------------------------------------------"
echo " Results ($EVAL_METHOD / $ACTIVATION)"
echo "----------------------------------------------"
worst=""
any=0
fail=0
while IFS= read -r pcc; do
    [[ -z "$pcc" ]] && continue
    any=1
    status="pass"
    if ! awk "BEGIN{exit !($pcc >= $PCC_THRESHOLD)}" 2>/dev/null; then
        status="FAIL"; fail=1
    fi
    printf "  PCC = %-12s  [%s, threshold %s]\n" "$pcc" "$status" "$PCC_THRESHOLD"
# Match both contract forms: "PCC = <n>" (poly/rational/parity/expalu) and
# "PCC(sim_vs_model) = <n>" (newton_root).
done < <(grep -oE 'PCC(\([A-Za-z_]+\))? = [0-9.eE+-]+' "$LOG" | sed -E 's/.* = //')

ulp=$(grep -oE 'ULP(\([A-Za-z_]+\))? = [0-9.eE+-]+' "$LOG" | sed -E 's/.* = //' | tail -1)
[[ -n "$ulp" ]] && echo "  ULP (last) = $ulp"

if [[ "$any" -eq 0 ]]; then
    echo "  (no PCC lines parsed — test skipped or crashed; see $LOG)"
    [[ $rc -eq 0 ]] && rc=1
fi

echo ""
if [[ $rc -eq 0 && $fail -eq 0 ]]; then
    echo "STATUS: PASS  (log: $LOG)"
    exit 0
else
    echo "STATUS: FAIL (pytest rc=$rc, pcc_fail=$fail; log: $LOG)"
    exit 1
fi
