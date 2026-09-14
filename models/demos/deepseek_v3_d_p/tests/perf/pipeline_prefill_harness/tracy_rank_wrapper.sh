#!/usr/bin/env bash
# Per-MPI-rank Tracy wrapper for the prefill runner.
#
# WHY: the device profiler alone is not enough. `tracy --process-logs-only -r` hard-requires the Tracy
# HOST capture (tracy_profile_log_host.tracy) -- it runs tracy-csvexport on it to get op names/times and
# merges those with the device data. Capturing device-only produces a multi-GB profile_log_device.csv
# that no tool here can turn into ops_perf_results_*.csv. So every rank must run under the tracy wrapper.
#
# The reason that is not simply `ttrun -- python -m tracy ...` is that each wrapper starts its own
# capture daemon: four ranks would race for one port, and -o would point all four at one output tree.
# This script derives BOTH from the MPI rank, so each rank captures independently.
set -u
R="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}"
: "${PROF_ROOT:?PROF_ROOT must be set (forwarded with -x)}"
: "${PROF_NAME:=m4}"
PORT=$(( ${PROF_PORT_BASE:-8600} + R ))
OUT="$PROF_ROOT/rank$R"
mkdir -p "$OUT"
echo "[tracy-wrapper] rank=$R port=$PORT out=$OUT"
exec python3 -m tracy -r -p -o "$OUT" -n "${PROF_NAME}_rank${R}" -t "$PORT" \
  -m models.demos.common.prefill.runners.prefill_runner
