#!/usr/bin/env bash
# Run section 4 LLK craq-sim sweeps (weekly + nightly WH) on Galaxy compute.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO_ROOT}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/craq-parity-results/llk-run-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LOG="$RESULTS_DIR/run.log"
SUMMARY="$RESULTS_DIR/summary.tsv"

mkdir -p "$RESULTS_DIR"
: >"$SUMMARY"
: >"$RUN_LOG"
echo -e "suite\tstatus\texit_code\tduration_sec\tcommand" >>"$SUMMARY"

export TT_METAL_HOME="$REPO_ROOT"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:$PYTHONPATH}"
export ARCH_NAME="${ARCH_NAME:-blackhole}"

log() { echo "[llk-sweep] $*" | tee -a "$RUN_LOG"; }

run_cmd() {
    local suite="$1"
    shift
    local cmd=("$@")
    local logfile="$RESULTS_DIR/$(echo "$suite" | tr '/ ' '__').log"
    local start end dur rc status

    log "RUN [$suite]: ${cmd[*]}"
    start=$(date +%s)
    set +e
    bash -lc "cd '$REPO_ROOT' && ${cmd[*]}" >"$logfile" 2>&1
    rc=$?
    set -e
    end=$(date +%s)
    dur=$((end - start))

    case "$rc" in
        0) status=PASS ;;
        124) status=TIMEOUT ;;
        *) status=FAIL ;;
    esac

    log "DONE [$suite] status=$status rc=$rc duration=${dur}s"
    echo -e "${suite}\t${status}\t${rc}\t${dur}\t${cmd[*]}" >>"$SUMMARY"
}

"$REPO_ROOT/scripts/setup-llk-ttsim-env.sh" | tee -a "$RUN_LOG"

if [ ! -x "$CRAQ_SIM/scripts/llk-pytest-sweep.sh" ]; then
    log "ERROR: missing $CRAQ_SIM/scripts/llk-pytest-sweep.sh"
    exit 127
fi

run_cmd "4.llk/weekly_wh" \
    "$CRAQ_SIM/scripts/llk-pytest-sweep.sh weekly wh --timeout 120 --workers 2 --run-root $RESULTS_DIR/llk-weekly-wh"
run_cmd "4.llk/nightly_wh" \
    "$CRAQ_SIM/scripts/llk-pytest-sweep.sh nightly wh --timeout 300 --workers 2 --run-root $RESULTS_DIR/llk-nightly-wh"

log "Summary: $SUMMARY"
python3 "$REPO_ROOT/scripts/generate-parity-error-summary.py" "$RESULTS_DIR" || true
