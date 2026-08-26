#!/usr/bin/env bash
# Quasar pool PERF runner — craq-sim per-dispatch clock measurement for num_threads A/B.
#
# Usage:  ./run_qpool_perf.sh
# Configure the trace (shape/kernel/cores/iterations) in test_qpool_perf.py's CONFIG block.
#
# Flow: one warmup iteration (JIT + program cache), then MEASURED_ITERS labeled iterations,
# then qpool_perf_report.py averages the pool program's `clocks` column from the sim's
# per-dispatch trace. Numbers are SIM CLOCKS from a functional simulator — meaningful only
# as a RELATIVE comparison on the same sim build (e.g. compute num_threads 1 vs 2 vs 4),
# never as silicon performance.
#
# Requires the craq-sim build from branch wransom/qsr-csr-timeout-count (adds the `clocks`
# trace column).
set -euo pipefail

# =============================== CONFIG — edit me ===============================
SIM_SO="$HOME/sim/qsr/libttsim.so"          # Quasar craq-sim library
TIMEOUT_S=600                               # kill the run after this many seconds
OUT_DIR=""                                  # trace output dir; empty = generated/qpool_perf
# =================================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
OUT_DIR="${OUT_DIR:-$REPO_ROOT/generated/qpool_perf}"

if [[ ! -f "$SIM_SO" ]]; then
    echo "QPOOL-PERF: no sim library at $SIM_SO — see run_qpool_sim.sh for build instructions." >&2
    exit 1
fi
if [[ ! -f "$(dirname "$SIM_SO")/soc_descriptor.yaml" ]]; then
    cp "$REPO_ROOT/tt_metal/soc_descriptors/quasar_32_arch.yaml" "$(dirname "$SIM_SO")/soc_descriptor.yaml"
fi

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/ttsim_perf_trace.tsv"

export TT_METAL_SIMULATOR="$SIM_SO"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_FORCE_JIT_COMPILE=1
unset TT_METAL_LLK_ASSERTS TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS 2>/dev/null || true
# Per-dispatch sim perf trace, with the python test labeling phases via TTSIM_PERF_TRACE_NODEID.
export TTSIM_PERF_TRACE=1
export TTSIM_PERF_TRACE_PER_DISPATCH=1
export TTSIM_PERF_TRACE_NODEID_COLUMN=1
export TTSIM_PERF_TRACE_OUT="$OUT_DIR"
export TTSIM_PERF_TRACE_NODEID="boot"

cd "$REPO_ROOT"
set +e
timeout --foreground "$TIMEOUT_S" pytest -q -s "$SCRIPT_DIR/test_qpool_perf.py" "$@"
rc=$?
set -e
if [[ $rc -eq 124 ]]; then
    echo "QPOOL-PERF: TIMED OUT after ${TIMEOUT_S}s — treat as a device-side HANG."
    exit $rc
fi
if [[ $rc -ne 0 ]]; then
    exit $rc
fi

echo
python3 "$SCRIPT_DIR/qpool_perf_report.py" "$OUT_DIR/ttsim_perf_trace.tsv"
echo "QPOOL-PERF: raw trace at $OUT_DIR/ttsim_perf_trace.tsv"
