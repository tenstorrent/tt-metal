#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrate a single workload run that captures BOTH the device profiler
# log AND the ttnvtop recorder CSV simultaneously, then runs compare.py.
#
# Usage (default: 1-layer Llama-3.2-1B batch-32 perf):
#   bash tt_metal/tools/ttnvtop/scripts/run_and_compare.sh
#
# Override workload via WORKLOAD_CMD env var:
#   WORKLOAD_CMD='pytest -xvs my_test.py' bash .../run_and_compare.sh
#
# All artifacts go under runs/<timestamp>/.

set -euo pipefail

# ───────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────
TT_METAL_HOME="${TT_METAL_HOME:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COLLECTOR_BIN="${COLLECTOR_BIN:-${TT_METAL_HOME}/build_Release/tools/ttnvtop-collector}"
RECORD_PY="${RECORD_PY:-${SCRIPT_DIR}/record.py}"
COMPARE_PY="${COMPARE_PY:-${SCRIPT_DIR}/compare.py}"

# Profiler output (fixed by tt-metal)
PROFILER_LOG="${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"

# Per-run output directory
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${TT_METAL_HOME}/runs/${TS}"
mkdir -p "$RUN_DIR"

# Sampling rates (collector publish must >= recorder hz)
SAMPLE_HZ="${SAMPLE_HZ:-300}"
PUBLISH_HZ="${PUBLISH_HZ:-100}"
RECORD_HZ="${RECORD_HZ:-100}"
AICLK_MHZ="${AICLK_MHZ:-1000}"

# Default workload: 1-layer Llama-3.2-1B batch-32 perf
WORKLOAD_CMD="${WORKLOAD_CMD:-pytest -xvs models/tt_transformers/demo/simple_text_demo.py -k 'performance and batch-32 and not log' --num_layers 1}"

HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"

# ───────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ───────────────────────────────────────────────────────────────────────
[[ -x "$COLLECTOR_BIN" ]] || { echo "FATAL: collector binary not found: $COLLECTOR_BIN"; exit 1; }
[[ -f "$RECORD_PY" ]]    || { echo "FATAL: record.py not found: $RECORD_PY"; exit 1; }
[[ -f "$COMPARE_PY" ]]   || { echo "FATAL: compare.py not found: $COMPARE_PY"; exit 1; }

# Verify libtt_metal has profiler instrumentation. Without ENABLE_PROFILER=ON,
# TT_METAL_DEVICE_PROFILER=1 is silently ignored and we'd dump only firmware-
# boot zones (~22 lines).
if ! strings "${TT_METAL_HOME}/build_Release/tt_metal/libtt_metal.so" 2>/dev/null \
        | grep -q "ProfilerOptionalMetadata"; then
    echo "WARNING: libtt_metal.so was not built with ENABLE_PROFILER=ON."
    echo "         The profiler dump will only contain firmware-boot rows."
    echo "         To rebuild with profiler:"
    echo "             cmake -B build_Release -DENABLE_PROFILER=ON ..."
    echo "             cmake --build build_Release -j8"
    echo "         Continuing anyway in case the symbol check is wrong."
    echo ""
fi

# Clean any stale collector/recorder
pkill -f "ttnvtop-collector" 2>/dev/null || true
pkill -f "record.py"          2>/dev/null || true
sleep 0.3

# Reset profiler log so we only capture this run
rm -f "$PROFILER_LOG"

# Reset registry so name lookups start fresh
rm -f /dev/shm/tt_program_registry

# ───────────────────────────────────────────────────────────────────────
# Cleanup trap
# ───────────────────────────────────────────────────────────────────────
COLL_PID=""
REC_PID=""

cleanup() {
    set +e
    if [[ -n "$REC_PID" ]] && kill -0 "$REC_PID" 2>/dev/null; then
        kill -INT "$REC_PID" 2>/dev/null
        wait "$REC_PID" 2>/dev/null
    fi
    if [[ -n "$COLL_PID" ]] && kill -0 "$COLL_PID" 2>/dev/null; then
        kill -INT "$COLL_PID" 2>/dev/null
        wait "$COLL_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# ───────────────────────────────────────────────────────────────────────
# Stage 1: collector
# ───────────────────────────────────────────────────────────────────────
echo "── stage 1: collector ($SAMPLE_HZ Hz sample / $PUBLISH_HZ Hz publish) ──"
"$COLLECTOR_BIN" \
    --sample-hz "$SAMPLE_HZ" \
    --publish-hz "$PUBLISH_HZ" \
    --log-file "$RUN_DIR/collector.log" &
COLL_PID=$!
sleep 1.5  # let it discover chips and create the SHM files

# Verify SHM appeared
if ! ls /dev/shm/tt_device_*_util >/dev/null 2>&1; then
    echo "FATAL: collector did not create /dev/shm/tt_device_*_util — see $RUN_DIR/collector.log"
    exit 1
fi

# ───────────────────────────────────────────────────────────────────────
# Stage 2: ttnvtop recorder
# ───────────────────────────────────────────────────────────────────────
echo "── stage 2: ttnvtop recorder ($RECORD_HZ Hz, output $RUN_DIR/ttnvtop.csv) ──"
python3 "$RECORD_PY" --hz "$RECORD_HZ" --out "$RUN_DIR/ttnvtop.csv" \
    > "$RUN_DIR/recorder.log" 2>&1 &
REC_PID=$!
sleep 0.5

# ───────────────────────────────────────────────────────────────────────
# Stage 3: workload (with both profiler and ttnvtop registrar enabled)
# ───────────────────────────────────────────────────────────────────────
echo "── stage 3: workload ──"
echo "  CMD:  $WORKLOAD_CMD"
echo "  HF:   $HF_MODEL"
echo "  log:  $RUN_DIR/workload.log"
echo ""

set +e
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TTNVTOP_REGISTER_PROGRAMS=1 \
HF_MODEL="$HF_MODEL" \
    bash -c "$WORKLOAD_CMD" 2>&1 | tee "$RUN_DIR/workload.log"
WORKLOAD_RC=${PIPESTATUS[0]}
set -e

echo ""
echo "── workload exited with rc=$WORKLOAD_RC ──"

# Stop recorder + collector cleanly so they flush.
if [[ -n "$REC_PID" ]] && kill -0 "$REC_PID" 2>/dev/null; then
    kill -INT "$REC_PID" 2>/dev/null
    wait "$REC_PID" 2>/dev/null || true
fi
if [[ -n "$COLL_PID" ]] && kill -0 "$COLL_PID" 2>/dev/null; then
    kill -INT "$COLL_PID" 2>/dev/null
    wait "$COLL_PID" 2>/dev/null || true
fi
COLL_PID=""
REC_PID=""

# ───────────────────────────────────────────────────────────────────────
# Stage 4: snapshot artifacts
# ───────────────────────────────────────────────────────────────────────
echo "── stage 4: snapshot artifacts ──"
if [[ -f "$PROFILER_LOG" ]]; then
    cp "$PROFILER_LOG" "$RUN_DIR/profile_log_device.csv"
    PROF_LINES=$(wc -l < "$RUN_DIR/profile_log_device.csv")
    echo "  profiler:  $RUN_DIR/profile_log_device.csv  ($PROF_LINES lines)"
else
    echo "  WARNING: profiler log missing at $PROFILER_LOG"
fi

if [[ -f "$RUN_DIR/ttnvtop.csv" ]]; then
    TT_LINES=$(wc -l < "$RUN_DIR/ttnvtop.csv")
    echo "  ttnvtop:   $RUN_DIR/ttnvtop.csv  ($TT_LINES lines)"
else
    echo "  WARNING: ttnvtop record missing"
fi

if [[ -f /dev/shm/tt_program_registry ]]; then
    cp /dev/shm/tt_program_registry "$RUN_DIR/tt_program_registry.bin"
    echo "  registry:  $RUN_DIR/tt_program_registry.bin"
fi

# ───────────────────────────────────────────────────────────────────────
# Stage 5: comparison
# ───────────────────────────────────────────────────────────────────────
if [[ -f "$RUN_DIR/profile_log_device.csv" && -f "$RUN_DIR/ttnvtop.csv" ]]; then
    echo ""
    echo "── stage 5: comparison ──"
    python3 "$COMPARE_PY" \
        --profiler "$RUN_DIR/profile_log_device.csv" \
        --ttnvtop  "$RUN_DIR/ttnvtop.csv" \
        --registry "$RUN_DIR/tt_program_registry.bin" \
        --aiclk-mhz "$AICLK_MHZ" 2>&1 | tee "$RUN_DIR/compare.txt"
    CMP_RC=${PIPESTATUS[0]}
else
    echo ""
    echo "── stage 5: skipping comparison (one or both inputs missing) ──"
    CMP_RC=2
fi

echo ""
echo "── all artifacts: $RUN_DIR ──"
ls -lh "$RUN_DIR"

# Propagate workload failure first; otherwise the compare verdict
exit $(( WORKLOAD_RC != 0 ? WORKLOAD_RC : CMP_RC ))
