#!/usr/bin/env bash
# Zero-compute chart points for extra core counts (e.g. 70, 90) around the BW peak.
#
# Usage:
#   source python_env/bin/activate
#   export TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_DISPATCH=1
#   ./run_extra_cores_chart.sh
#
# Env:
#   CORE_COUNTS  - default "48 60 70 72 84 90 100"
#   Then run rebuild_batch2_results.py with EXTRA_CORES merged (see script footer).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_DEVICE_PROFILER=1
# dispatch-core profiling off (workers only); official op2op uses KERNEL zones, no DISP markers needed
# export TT_METAL_DEVICE_PROFILER_DISPATCH=1

CORE_COUNTS="${CORE_COUNTS:-48 60 70 72 84 90 100}" \
NUM_RUNS="${NUM_RUNS:-3}" \
INPUT_DEPTHS="${INPUT_DEPTHS:-8,12,16,24,32,48,64}" \
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-2,4,8,16,32,64}" \
BW_PAGES="${BW_PAGES:-256}" \
COMPUTE_NOPS=0 \
CHART_TAG=batch2_extra \
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}" \
USE_DBUF_TRID=1 \
BUILD="${BUILD:-1}" \
bash "${SCRIPT_DIR}/run_op_to_op_chart_sweep.sh" \
  2>&1 | tee "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch2/extra_cores_sweep.log"

echo
echo "Extra chart CSV: ${TT_METAL_HOME}/generated/profiler/op_to_op_runs/chart_sweep/batch2_extra/chart_data.csv"
echo "Re-merge: python_env/bin/python3 ${SCRIPT_DIR}/rebuild_batch2_results.py"
