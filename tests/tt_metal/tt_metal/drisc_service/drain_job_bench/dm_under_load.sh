#!/bin/bash
# Data movement bandwidth tests under copy loads (SteadyLoad local / d2h on K banks), measured with the legacy device
# profiler (legacy) or with the streaming profiler (stream), whose drain jobs are a load themselves. Each run leaves
# <label>.log, <label>.device.csv (a legacy-format device log) and, under copies, <label>.load. Then run
# dm_reparse.py and dm_table.py on <outdir>. Run from the tt-metal root with python_env active.
# Usage: dm_under_load.sh <outdir> <reps>
set -u
OUT=$1; REPS=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export TT_METAL_HOME=$PWD
LOG=$PWD/generated/profiler/.logs/profile_log_device.csv
LOAD_BIN=./build_Release/test/tt_metal/unit_tests_drisc_service
DM_BIN=./build_Release/test/tt_metal/unit_tests_data_movement
F="*.PCIeReadBandwidth:*.PCIeReadBandwidthSweep:*.PCIeWriteBandwidthSweep:*.PCIeHostReadBandwidthSweep"
F+=":*.PCIeHostWriteBandwidthSweep:*.TensixDataMovementDramNeighbourDirectedIdeal"
F+=":*.TensixDataMovementMultiInterleavedReadDirectedIdeal:*.TensixDataMovementMultiInterleavedWriteDirectedIdeal"
F+=":*.TensixDataMovementDRAMShardedReadDirectedIdeal:*.TensixDataMovementOneToOneDirectedIdeal"
F+=":*.TensixDataMovementOneFromOneDirectedIdeal:*.TensixDataMovementAllToAllDirectedIdeal"
F+=":*.TensixDataMovementAllFromAllDirectedIdeal"
run() {  # <label> <legacy|stream> <none|local|d2h> <banks>
  local L=$OUT/$1 BACKEND=$2 COPIES=$3 BANKS=$4 LP=""
  if [ "$COPIES" != none ]; then
    DRISC_SERVICE_LOAD_SECONDS=2400 DRISC_SERVICE_LOAD_OP=$COPIES DRISC_SERVICE_LOAD_BANKS=$BANKS \
      DRISC_SERVICE_LOAD_REPORT=$L.load $LOAD_BIN --gtest_filter=DriscService.SteadyLoad > "$L.load.log" 2>&1 &
    LP=$!
    sleep 6
  fi
  python -c "from tracy.common import clear_profiler_runtime_artifacts as c; c()"
  if [ "$BACKEND" = legacy ]; then
    TT_METAL_DM_HOST_BW_ITERS=200 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=1333 \
      timeout 1200 $DM_BIN --gtest_filter="$F" > "$L.log" 2>&1
    cp "$LOG" "$L.device.csv"
  else
    TT_METAL_DM_HOST_BW_ITERS=200 TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_ZONE_CSV=$L.zones.csv \
      timeout 1200 $DM_BIN --gtest_filter="$F" > "$L.log" 2>&1
    # Same columns as the legacy device log, minus the two logical-coordinate ones.
    { sed -n 1p "$L.zones.csv"; tail -n +2 "$L.zones.csv" | sed "s/, /,/g" | cut -d, -f1-15; } > "$L.device.csv"
  fi
  local RC=$?
  if [ -n "$LP" ]; then kill "$LP" 2>/dev/null; wait "$LP" 2>/dev/null; fi
  echo "$1 rc=$RC $(grep -oE "PASSED  \] [0-9]+" "$L.log") failed=$(grep -c "^\[  FAILED  \]" "$L.log")"
}
for R in $(seq 1 "$REPS"); do
  run r${R}_L_none legacy none 0
  run r${R}_L_local legacy local 99
  run r${R}_L_d2h1 legacy d2h 1
  run r${R}_L_d2h2 legacy d2h 2
  run r${R}_L_d2h7 legacy d2h 99
  run r${R}_S_none stream none 0
  run r${R}_S_local stream local 99
  run r${R}_S_d2h1 stream d2h 1
  run r${R}_S_d2h7 stream d2h 99
done
