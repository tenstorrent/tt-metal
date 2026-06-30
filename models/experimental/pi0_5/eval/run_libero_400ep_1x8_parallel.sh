#!/bin/bash
# Run the 400-ep LIBERO benchmark across 4 parallel 1×8 mesh instances.
# Each process owns 8 chips and runs 1 LIBERO suite (10 tasks × 10 init states = 100 eps).
#
# Total: 4 procs × 100 eps = 400 eps. Wall-clock: ~5-10 min.
#
# Each process streams progress to stdout prefixed with [P<id>-<suite>]; the
# full per-process logs are written to /tmp/libero_400ep_<timestamp>/<suite>.log.

set -uo pipefail
cd "$(dirname "$0")/../../../.."
ROOT=$PWD
export PYTHONPATH=$ROOT:/home/tt-admin/pi05_cache/libero_repo
export TT_METAL_HOME=$ROOT
export LIBERO_REPO_PATH=/home/tt-admin/pi05_cache/libero_repo
export PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model
export TT_METAL_CACHE=$ROOT/.tt_metal_cache
export MUJOCO_GL=osmesa

# Note: libero_rollout.py auto-sources pi05_production.env via no mechanism today
# (it's not the same as the perf tests); rely on the production env vars below.
# These match _bench_runs/pi05_production.env.
set -a
source "$ROOT/_bench_runs/pi05_production.env"
set +a

# Use upstream LIBERO suite names. The 4th suite is "libero_10" (long horizon) —
# the existing 400-ep sweep at _bench_runs/libero_400ep_sweep.sh uses this name.
SUITES=(libero_spatial libero_object libero_goal libero_10)
LOG_DIR="/tmp/libero_400ep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "=== 400-ep LIBERO on 1×8 mesh (4 parallel processes) ==="
echo "Logs dir: $LOG_DIR"
echo ""

declare -a PIDS=()
for proc_id in 0 1 2 3; do
    chip_lo=$((proc_id * 8))
    chip_hi=$((chip_lo + 7))
    devices=$(seq -s, $chip_lo $chip_hi)
    suite="${SUITES[$proc_id]}"
    logfile="$LOG_DIR/${suite}.log"

    echo "[P$proc_id] suite=$suite chips=$devices -> $logfile"

    (TT_VISIBLE_DEVICES="$devices" \
     python_env/bin/python models/experimental/pi0_5/eval/libero_rollout.py \
        --backend ttnn_1x8 \
        --checkpoint /home/tt-admin/pi05_cache/pi05_libero_upstream \
        --suites "$suite" \
        --task-range 0 9 \
        --num-episodes 10 \
        --steps-sweep 5 \
        2>&1 | tee "$logfile" | sed -u "s/^/[P$proc_id-$suite] /") &
    PIDS+=($!)
done

echo ""
echo "Spawned ${#PIDS[@]} processes; PIDs: ${PIDS[*]}"
echo "Tail individual logs with: tail -f $LOG_DIR/*.log"
echo "Waiting for all 4 processes to finish..."
echo ""

wait "${PIDS[@]}"

echo ""
echo "=== ALL DONE — per-suite summary ==="
for suite in "${SUITES[@]}"; do
    echo "--- $suite ---"
    grep -E "TOTAL:|task .*:" "$LOG_DIR/$suite.log" | tail -15
done

echo ""
echo "Logs: $LOG_DIR"
