#!/usr/bin/env bash
# Single-galaxy worst-AllGather: FABRIC_1D or 2D, no peer. $1 = 1d|2d
set -u
MODE="${1:?usage: run_worst_ag_single.sh <1d|2d>}"
WK=/home/ppopovic/tt-metal/models/demos/common/prefill/fabric_debug
TAG="worst_ag_1gal_${MODE}"; OUT=/data/ppopovic/prof_out/$TAG; LOG=$WK/logs/${TAG}.log
rm -rf "$OUT"; mkdir -p "$OUT"; cd /home/ppopovic/tt-metal
rm -f "$WK"/__pycache__/worst_allgather_test*.pyc
ps -u ppopovic -o pid,command|grep -F 'worst_allgather_test'|grep -v grep|awk '{print $1}'|xargs -r kill -9 2>/dev/null
timeout 160 tt-smi -glx_reset >"$WK/logs/rst_${TAG}.txt" 2>&1; echo "reset: $(grep -c 'Re-initialized 32 boards' $WK/logs/rst_${TAG}.txt)" | tee "$LOG"
echo "=== worst-AG single galaxy FABRIC_${MODE} $(date) ===" | tee -a "$LOG"
PREFILL_FABRIC_MODE=$MODE PREFILL_MODEL=kimi_k2_6 WORST_AG_REPLAYS=100 WORST_AG_NUM_LINKS=2 \
  TT_METAL_DEVICE_PROFILER=1 PYTHONPATH=/home/ppopovic/tt-metal:$WK \
  timeout --signal=KILL 900 python_env/bin/python -m tracy -r -o "$OUT" -m worst_allgather_test >>"$LOG" 2>&1
echo "=== exit rc=$? $(date) ===" | tee -a "$LOG"
find "$OUT" -name 'ops_perf_results*.csv' | tee -a "$LOG"
