#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run validate_sweep_pipeline.py for all operations in a model trace.
#
# Usage:
#   bash tests/sweep_framework/run_all_sweeps.sh \
#       --model-trace model_tracer/traced_operations/ttnn_operations_master_v2_reconstructed.json \
#       [--mesh-shape 4x8] [--arch-name wormhole_b0] [--dry-run] [--skip-existing]
#
# Requires: python3, jq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_TRACE=""
MESH_SHAPE="4x8"
ARCH_NAME="wormhole_b0"
SUITE="model_traced"
DRY_RUN=""
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-trace)  MODEL_TRACE="$2"; shift 2 ;;
        --mesh-shape)   MESH_SHAPE="$2"; shift 2 ;;
        --arch-name)    ARCH_NAME="$2"; shift 2 ;;
        --suite)        SUITE="$2"; shift 2 ;;
        --dry-run)      DRY_RUN="--dry-run"; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_TRACE" ]]; then
    echo "ERROR: --model-trace is required" >&2
    exit 1
fi

if [[ ! -f "$MODEL_TRACE" ]]; then
    echo "ERROR: Model trace not found: $MODEL_TRACE" >&2
    exit 1
fi

TRACED_OPS_DIR="$REPO_ROOT/model_tracer/traced_operations"
SWEEPS_MODULE_DIR="$REPO_ROOT/tests/sweep_framework/sweeps/model_traced"

op_to_module() {
    local op_name="$1"
    # Strip ttnn.experimental. or ttnn. prefix
    local base="${op_name#ttnn.experimental.}"
    base="${base#ttnn.}"
    echo "model_traced.${base}_model_traced"
}

echo "========================================================================"
echo "Run All Sweeps"
echo "========================================================================"
echo "  Model trace: $MODEL_TRACE"
echo "  Mesh shape:  $MESH_SHAPE"
echo "  Arch:        $ARCH_NAME"
echo "  Suite:       $SUITE"
echo "  Dry run:     ${DRY_RUN:-no}"
echo "  Skip existing: $SKIP_EXISTING"
echo ""

# Extract operation names from the model trace JSON
OP_NAMES=$(python3 -c "
import json, sys
with open('$MODEL_TRACE') as f:
    data = json.load(f)
for op in sorted(data.get('operations', {}).keys()):
    print(op)
")

TOTAL=0
SKIPPED=0
RAN=0
FAILED=0
FAILED_OPS=""

for OP in $OP_NAMES; do
    TOTAL=$((TOTAL + 1))
    MODULE=$(op_to_module "$OP")
    BASENAME="${MODULE##*.}"

    # Derive the module file path
    MODULE_FILE="$SWEEPS_MODULE_DIR/${BASENAME}.py"
    if [[ ! -f "$MODULE_FILE" ]]; then
        echo "[$TOTAL] SKIP $OP -- no sweep module: $MODULE_FILE"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    SWEEP_TRACE_OUTPUT="$TRACED_OPS_DIR/sweep_trace_${BASENAME}.json"
    SWEEP_TRACE_SPLIT="${SWEEP_TRACE_OUTPUT%.json}_split"

    if [[ "$SKIP_EXISTING" == true ]] && [[ -d "$SWEEP_TRACE_SPLIT" ]] && [[ -n "$(ls -A "$SWEEP_TRACE_SPLIT" 2>/dev/null)" ]]; then
        echo "[$TOTAL] SKIP $OP -- sweep trace already exists: $SWEEP_TRACE_SPLIT"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "========================================================================"
    echo "[$TOTAL] Running: $OP  ->  $MODULE"
    echo "========================================================================"

    if python3 "$SCRIPT_DIR/validate_sweep_pipeline.py" \
        --model-trace "$MODEL_TRACE" \
        --module-name "$MODULE" \
        --suite "$SUITE" \
        --mesh-shape "$MESH_SHAPE" \
        --arch-name "$ARCH_NAME" \
        $DRY_RUN; then
        RAN=$((RAN + 1))
    else
        echo "ERROR: Pipeline failed for $OP" >&2
        FAILED=$((FAILED + 1))
        FAILED_OPS="$FAILED_OPS $OP"
    fi

    echo ""
done

echo "========================================================================"
echo "All Sweeps Complete"
echo "========================================================================"
echo "  Total ops:    $TOTAL"
echo "  Ran:          $RAN"
echo "  Skipped:      $SKIPPED"
echo "  Failed:       $FAILED"
if [[ -n "$FAILED_OPS" ]]; then
    echo "  Failed ops:  $FAILED_OPS"
fi
echo ""
echo "Next step: run validation"
echo "  python3 tests/sweep_framework/validate_all_traces.py \\"
echo "      --model-trace-split ${MODEL_TRACE%.json}_split \\"
echo "      --sweep-traces-dir $TRACED_OPS_DIR \\"
echo "      --output-report validation_report.txt"
