#!/usr/bin/env bash
# Steady-state op-to-op validation: trace warmup + later program transitions.
#
# - trace warmup replays (2 untimed) then 1 measured replay
# - num_programs=8 so we can analyze transitions >= 3→4
# - longer kernel (~10-20us): 16 tiles/core + compute nops
# - export with --min-prog-id 3 (skip early transitions)
#
# Usage:
#   ./run_op_to_op_validation.sh [NUM_RUNS] [NUM_ACTIVE_CORES]
#
set -euo pipefail

NUM_RUNS="${1:-5}"
NUM_CORES="${2:-110}"

CONFIG_LABEL="validation_cores${NUM_CORES}" \
MIN_PROG_ID=3 \
TILES_PER_CORE=16 \
INPUT_CB_DEPTH=2 \
READER_PUSH=2 \
EXTRA_ARGS="--use-trace --trace-warmup-replays 2 --num-programs 8 --compute-nops 2000 --use-device-profiler --use-realtime-profiler --reader-dbuf-trid --input-cb-depth-tiles 2 --output-cb-depth-tiles 4 --num-pages-per-core 16 --num-active-cores ${NUM_CORES}" \
bash "$(dirname "$0")/run_op_to_op_multi.sh" "${NUM_RUNS}"
