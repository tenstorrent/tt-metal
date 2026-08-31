#!/usr/bin/env bash
# Quasar pool harness runner (craq-sim) — one entry point for all qpool test modes:
#
#   ./run_qpool.sh              debug: ONE hand-picked trace (test_qpool_debug.py CONFIG block);
#                               300s timeout doubles as the hang detector
#   ./run_qpool.sh sweep        channel-width sweep (test_qpool_sweep.py), per-case verdicts
#   ./run_qpool.sh span         span(T) = prologue + marginal*T fit at the tree's current
#                               num_threads (test_qpool_span.py + qpool_span_report.py)
#   ./run_qpool.sh span-ab      full threads A/B: measure at num_threads=4, rebuild at 1,
#                               measure, restore 4, rebuild, report marginal/span gains + T*.
#                               Each leg DPRINT-verifies its live thread count and the
#                               num_threads toggle aborts loudly on a pattern miss.
#                               Do NOT commit pool_utils.cpp while span-ab runs.
#
# Extra args after the mode are passed to pytest.
set -euo pipefail

# =============================== CONFIG — edit me ===============================
SIM_SO="$HOME/sim/qsr/libttsim.so"
TIMEOUT_DEBUG_S=300
TIMEOUT_SWEEP_S=2400
TIMEOUT_SPAN_S=900
OUT_DIR=""  # span trace output dir; empty = generated/qpool_span
# =================================================================================

MODE="${1:-debug}"
[[ $# -gt 0 ]] && shift

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
OUT_DIR="${OUT_DIR:-$REPO_ROOT/generated/qpool_span}"
PU="$REPO_ROOT/ttnn/cpp/ttnn/operations/pool/pool_utils.cpp"

if [[ ! -f "$SIM_SO" ]]; then
    echo "run_qpool: no sim library at $SIM_SO — build craq-sim branch wransom/qsr-csr-timeout-count:" >&2
    echo "  ./make.py --env TTSIM_MARCH=-march=x86-64-v3 src/_out/release_qsr/libttsim.so   (no TTSIM_LTO=0)" >&2
    exit 1
fi
if [[ ! -f "$(dirname "$SIM_SO")/soc_descriptor.yaml" ]]; then
    cp "$REPO_ROOT/tt_metal/soc_descriptors/quasar_32_arch.yaml" "$(dirname "$SIM_SO")/soc_descriptor.yaml"
fi

sim_env() {
    export TT_METAL_SIMULATOR="$SIM_SO"
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_FORCE_JIT_COMPILE=1
    unset TT_METAL_LLK_ASSERTS TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS 2>/dev/null || true
}

span_trace_env() {
    export TTSIM_PERF_TRACE=1
    export TTSIM_PERF_TRACE_PER_DISPATCH=1
    export TTSIM_PERF_TRACE_NODEID_COLUMN=1
    export TTSIM_PERF_TRACE_OUT="$OUT_DIR"
    export TTSIM_PERF_TRACE_NODEID=boot
}

run_pytest() {  # $1 = test file, $2 = timeout
    set +e
    timeout --foreground "$2" pytest -q -s "$SCRIPT_DIR/$1" "$@"
    local rc=$?
    set -e
    return $rc
}

span_leg() {  # $1 = output tsv name, $2 = expected get_num_threads() (empty = skip verification)
    if [[ -n "${2:-}" ]]; then
        # Live thread-count verification (mandatory for A/B): a silent sed/build miss otherwise
        # produces a byte-identical A/B that LOOKS like a valid null result.
        local seen
        seen=$( (sim_env; TT_METAL_DPRINT_CORES='(0,0)' \
            timeout --foreground "$TIMEOUT_SPAN_S" pytest -q -s "$SCRIPT_DIR/test_qpool_debug.py" 2>&1) |
            grep -m1 -oE "qpool num_threads: [0-9]+" | grep -oE "[0-9]+$" || true)
        if [[ "$seen" != "$2" ]]; then
            echo "run_qpool: FATAL — leg expected num_threads=$2 but kernel reports '${seen:-none}'" >&2
            exit 1
        fi
    fi
    rm -f "$OUT_DIR/ttsim_perf_trace.tsv"
    (sim_env; span_trace_env; timeout --foreground "$TIMEOUT_SPAN_S" pytest -q -s "$SCRIPT_DIR/test_qpool_span.py")
    mv "$OUT_DIR/ttsim_perf_trace.tsv" "$OUT_DIR/$1"
}

set_threads() {  # $1 = from, $2 = to — with loud verification (sed exits 0 on no match!)
    sed -i "s/? $1 : 1;/? $2 : 1;/" "$PU"
    grep -q "? $2 : 1;" "$PU" || { echo "run_qpool: FATAL — thread toggle $1->$2 did not match in $PU" >&2; exit 1; }
}

cd "$REPO_ROOT"
case "$MODE" in
debug)
    sim_env
    set +e
    timeout --foreground "$TIMEOUT_DEBUG_S" pytest -q -s "$SCRIPT_DIR/test_qpool_debug.py" "$@"
    rc=$?
    set -e
    if [[ $rc -eq 124 ]]; then
        echo "run_qpool: TIMED OUT after ${TIMEOUT_DEBUG_S}s — treat as a device-side HANG (the sim spins silently on hangs)."
    fi
    exit $rc
    ;;
sweep)
    sim_env
    set +e
    timeout --foreground "$TIMEOUT_SWEEP_S" pytest -q -s "$SCRIPT_DIR/test_qpool_sweep.py" "$@"
    rc=$?
    set -e
    if [[ $rc -eq 124 ]]; then
        echo "run_qpool: TIMED OUT after ${TIMEOUT_SWEEP_S}s — the last 'QPOOL-SWEEP: C=...' banner names the hung case."
    fi
    exit $rc
    ;;
span)
    mkdir -p "$OUT_DIR"
    span_leg span_current.tsv ""
    python3 "$SCRIPT_DIR/qpool_span_report.py" "$OUT_DIR/span_current.tsv"
    echo "run_qpool: trace at $OUT_DIR/span_current.tsv"
    ;;
span-ab)
    mkdir -p "$OUT_DIR"
    restore() { sed -i 's/? 1 : 1;/? 4 : 1;/' "$PU"; }
    trap restore EXIT
    echo "run_qpool: span-ab leg 1/2 — num_threads=4"
    ./build_metal.sh > "$OUT_DIR/build_T4.log" 2>&1
    span_leg span_T4.tsv 4
    echo "run_qpool: span-ab leg 2/2 — num_threads=1"
    set_threads 4 1
    ./build_metal.sh > "$OUT_DIR/build_T1.log" 2>&1
    span_leg span_T1.tsv 1
    restore
    trap - EXIT
    ./build_metal.sh > "$OUT_DIR/build_restore.log" 2>&1
    python3 "$SCRIPT_DIR/qpool_span_report.py" "$OUT_DIR/span_T1.tsv" "$OUT_DIR/span_T4.tsv" threads=1 threads=4
    echo "run_qpool: traces at $OUT_DIR/span_T{1,4}.tsv (tree restored to num_threads=4 and rebuilt)"
    ;;
*)
    echo "run_qpool: unknown mode '$MODE' (debug|sweep|span|span-ab)" >&2
    exit 2
    ;;
esac
