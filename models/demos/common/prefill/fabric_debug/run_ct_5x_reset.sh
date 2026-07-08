#!/usr/bin/env bash
# 5x { fresh PARALLEL dual glx_reset (retry peer if it flakes) -> try connected capture }.
# On the first attempt where BOTH per-host yamls capture, run the APPLY pass + report trimmed AG.
set -u
WK=/home/ppopovic/tt-metal/models/demos/common/prefill/fabric_debug
PEER=bh-glx-110-d09u02; LOCAL=bh-glx-110-d09u08
LOG=$WK/logs/ct_5x.log; CTDIR=/data/ppopovic/prof_out/ct_connected
PY=/home/ppopovic/tt-metal/python_env/bin/python
export WORST_AG_OUT=/data/ppopovic/prof_out/worst_ag_pipe_trim WORST_AG_CT_DIR=$CTDIR
TT_METAL_HOME=/home/ppopovic/tt-metal; export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME:$WK"
source /home/ppopovic/tt-metal/python_env/bin/activate; cd "$TT_METAL_HOME"
: > "$LOG"
MPI="--host ${LOCAL}:1,${PEER}:1 --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x TT_METAL_HOME -x WORST_AG_OUT -x WORST_AG_CT_DIR -x WORST_AG_CT_MODE"
killall_both(){ for pid in $(ps -u ppopovic -o pid,command|grep -E 'ttrun\.py|prterun|prted|worst_allgather_test'|grep -v grep|awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
  timeout 20 ssh $PEER "for p in ttrun prted prterun worst_allgather_test; do pkill -9 -u ppopovic -f \$p; done" 2>/dev/null; }
dual_reset(){ killall_both; sleep 3
  timeout 340 tt-smi -glx_reset_auto >$WK/logs/ct5x_rl.txt 2>&1 & local A=$!
  timeout 360 ssh $PEER 'timeout 340 tt-smi -glx_reset_auto' >$WK/logs/ct5x_rp.txt 2>&1 & local B=$!
  wait $A; wait $B
  local rl=$(grep -c 'Re-initialized 32 boards' $WK/logs/ct5x_rl.txt) rp=$(grep -c 'Re-initialized 32 boards' $WK/logs/ct5x_rp.txt)
  if [ "$rp" = 0 ]; then echo "    peer reset flaked -> retry peer" | tee -a "$LOG"; timeout 360 ssh $PEER 'timeout 340 tt-smi -glx_reset_auto' >$WK/logs/ct5x_rp.txt 2>&1; rp=$(grep -c 'Re-initialized 32 boards' $WK/logs/ct5x_rp.txt); fi
  if [ "$rl" = 0 ]; then timeout 340 tt-smi -glx_reset_auto >$WK/logs/ct5x_rl.txt 2>&1; rl=$(grep -c 'Re-initialized 32 boards' $WK/logs/ct5x_rl.txt); fi
  echo "    reset local=$rl peer=$rp" | tee -a "$LOG"; sleep 6; }
capture_try(){ rm -rf "$CTDIR"; mkdir -p "$CTDIR"
  WORST_AG_CT_MODE=capture timeout --signal=KILL 700 python3 ttnn/ttnn/distributed/ttrun.py \
    --tcp-interface ens5f0np0 --rank-binding $WK/bindings/worst_ag_2galaxy.yaml --mpi-args "$MPI" \
    "$PY" -m worst_allgather_test >>"$LOG" 2>&1; }
for i in $(seq 1 5); do
  echo "=== DUAL RESET + TRY $i/5 $(date) ===" | tee -a "$LOG"
  dual_reset
  capture_try
  r0=$(find $CTDIR/rank0 -name channel_trimming_capture.yaml 2>/dev/null|head -1|xargs wc -l 2>/dev/null|awk '{print $1+0}')
  r1=$(find $CTDIR/rank1 -name channel_trimming_capture.yaml 2>/dev/null|head -1|xargs wc -l 2>/dev/null|awk '{print $1+0}')
  echo "  TRY $i: rank0=${r0:-0} rank1=${r1:-0}" | tee -a "$LOG"
  if [ "${r0:-0}" -gt 100 ] && [ "${r1:-0}" -gt 100 ]; then
    echo "  *** BOTH CAPTURED on try $i -> running APPLY ***" | tee -a "$LOG"
    rm -rf "$WORST_AG_OUT"; mkdir -p "$WORST_AG_OUT"; dual_reset
    WORST_AG_CT_MODE=apply timeout --signal=KILL 900 python3 ttnn/ttnn/distributed/ttrun.py \
      --tcp-interface ens5f0np0 --rank-binding $WK/bindings/worst_ag_2galaxy.yaml --mpi-args "$MPI" \
      bash "$WK/worst_ag_rank_wrapper.sh" >>"$LOG" 2>&1
    echo "  APPLY done; CSV: $(find $WORST_AG_OUT -name 'ops_perf_results*.csv'|head -1)" | tee -a "$LOG"; break
  fi
done
echo "=== 5x DONE $(date) ===" | tee -a "$LOG"; killall_both
