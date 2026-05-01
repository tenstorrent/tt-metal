#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Profile all Molmo2 sub-blocks with Tracy (one run per block) and build Excel.
#
# Usage:
#   cd /path/to/tt-metal
#   bash models/demos/molmo2/tests/run_block_profiles.sh [--seq-len 128] [--output molmo2_profile.xlsx]
#
# Requires: openpyxl  (pip install openpyxl)

set -euo pipefail
cd "$(dirname "$0")/../../.."   # go to tt-metal root

SEQ=128
OUTPUT="molmo2_block_profile.xlsx"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seq-len) SEQ="$2"; shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

OUT_DIR="generated/profiler/block_csvs"
mkdir -p "$OUT_DIR"

BLOCKS=(text_attention text_mlp decoder_block vit_encoder image_pooling image_projector)

for BLOCK in "${BLOCKS[@]}"; do
    echo "========================================"
    echo "  Profiling: $BLOCK (seq_len=$SEQ)"
    echo "========================================"

    REPORT_DIR="generated/profiler/block_reports/${BLOCK}"
    mkdir -p "$REPORT_DIR"

    python -m tracy -p -v -r \
        -n "${BLOCK}" \
        -o "${REPORT_DIR}" \
        models/demos/molmo2/tests/profile_single_block.py \
        --block "${BLOCK}" \
        --seq-len "${SEQ}" \
        2>&1 | grep -E "Profiling|complete|INFO.*csv|ERROR|FATAL"

    # Tracy saves to a nested path: <REPORT_DIR>/reports/<BLOCK>/<timestamp>/ops_perf_results_*.csv
    LATEST_CSV=$(find "${REPORT_DIR}" -name "ops_perf_results_*.csv" | sort | tail -1)
    if [[ -n "$LATEST_CSV" ]]; then
        cp "$LATEST_CSV" "${OUT_DIR}/${BLOCK}_ops.csv"
        N=$(wc -l < "${OUT_DIR}/${BLOCK}_ops.csv")
        echo "  -> Saved: ${OUT_DIR}/${BLOCK}_ops.csv  (${N} rows)"
    else
        echo "  WARNING: no CSV generated for ${BLOCK}"
    fi
done

echo ""
echo "========================================"
echo "  Building Excel: $OUTPUT"
echo "========================================"

python models/demos/molmo2/tests/make_profile_xlsx.py \
    --csv-dir "$OUT_DIR" \
    --output "$OUTPUT"

echo ""
echo "Done: $OUTPUT"
