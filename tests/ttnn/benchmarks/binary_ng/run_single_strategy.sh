#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Run benchmarks for a single grid selection strategy
# Usage: ./run_single_strategy.sh <strategy>
# Example: ./run_single_strategy.sh current

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <strategy>"
    echo "Strategies: current, a_first, b_first, full_grid, half_grid, max_ab, max_abc, min_ab, new_grid"
    exit 1
fi

STRATEGY=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "====================================================================="
echo " Running benchmarks with strategy: ${STRATEGY}"
echo "====================================================================="
echo ""

export TT_METAL_BINARY_NG_GRID_STRATEGY="${STRATEGY}"

# Run pytest (profiler must be enabled externally via export TT_METAL_DEVICE_PROFILER=1)
pytest "${SCRIPT_DIR}/benchmark_grid_selection.py" \
    -v \
    --tb=short \
    -m "not slow"

echo ""
echo "====================================================================="
echo " Benchmark complete for strategy: ${STRATEGY}"
echo "====================================================================="
echo "Configuration saved to: ${RESULTS_DIR}/benchmark_results_${STRATEGY}.csv"
echo ""
echo "If profiler was enabled, check:"
echo "  /workspace/generated/profiler/reports/<timestamp>/ops_perf_results_*.csv"
echo ""
echo "To merge profiler data with configurations:"
echo "  python merge_profiler_data.py --results-dir ${RESULTS_DIR} --strategy ${STRATEGY}"
