#!/usr/bin/env bash
# Connected 2-galaxy channel-trimming with per-pass retry (flaky inter-galaxy fabric).
# CAPTURE (per-host yamls, plain python) then APPLY (proven wrapper, rank0 tracy). Trimming env set
# inside the python test (keyed on WORST_AG_CT_MODE + rank). Retries each pass w/ fresh reset until it maps.
set -u
WK=/home/ppopovic/tt-metal/models/demos/common/prefill/fabric_debug
PEER=bh-glx-110-d09u02; LOCAL=bh-glx-110-d09u08
LOG=$WK/logs/worst_ag_pipe_ct.log; CTDIR=/data/ppopovic/prof_out/ct_connected
PY=/home/ppopovic/tt-metal/python_env/bin/python; TRIES=6
export WORST_AG_OUT=/data/ppopovic/prof_out/worst_ag_pipe_trim WORST_AG_CT_DIR=$CTDIR
TT_METAL_HOME=/home/ppopovic/tt-metal; export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME:$WK"
source /home/ppopovic/tt-metal/python_env/bin/activate; cd "$TT_METAL_HOME"
rm -rf "$CTDIR" "$WORST_AG_OUT"; mkdir -p "$CTDIR" "$WORST_AG_OUT"
MPI="--host ${LOCAL}:1,${PEER}:1 --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x TT_METAL_HOME -x WORST_AG_OUT -x WORST_AG_CT_DIR -x WORST_AG_CT_MODE"
killall_both(){ for pid in $(ps -u ppopovic -o pid,command|grep -E 'ttrun\.py|prterun|prted|worst_allgather_test'|grep -v grep|awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
  timeout 20 ssh $PEER "for p in ttrun prted prterun worst_allgather_test; do pkill -9 -u ppopovic -f \$p; done" 2>/dev/null; }
reset_both(){ killall_both; sleep 3
  timeout 340 tt-smi -glx_reset_auto >$WK/logs/rst_ct_local.txt 2>&1 & local A=$!
  timeout 360 ssh $PEER 'timeout 340 tt-smi -glx_reset_auto' >$WK/logs/rst_ct_peer.txt 2>&1 & local B=$!
  wait $A; wait $B; sleep 6
  echo "  reset local=$(grep -c 'Re-initialized 32 boards' $WK/logs/rst_ct_local.txt) peer=$(grep -c 'Re-initialized 32 boards' $WK/logs/rst_ct_peer.txt)"; }
echo "=== CT connected (per-pass retry) $(date) ===" | tee "$LOG"
# ---- CAPTURE (retry until both per-host yamls exist) ----
for t in $(seq 1 $TRIES); do
  echo "=== CAPTURE attempt $t $(date) ===" | tee -a "$LOG"; reset_both | tee -a "$LOG"
  WORST_AG_CT_MODE=capture timeout --signal=KILL 900 python3 ttnn/ttnn/distributed/ttrun.py \
    --tcp-interface ens5f0np0 --rank-binding $WK/bindings/worst_ag_2galaxy.yaml --mpi-args "$MPI" \
    "$PY" -m worst_allgather_test >>"$LOG" 2>&1
  r0=$(find $CTDIR/rank0 -name channel_trimming_capture.yaml 2>/dev/null|head -1|xargs wc -l 2>/dev/null|awk '{print $1+0}')
  r1=$(find $CTDIR/rank1 -name channel_trimming_capture.yaml 2>/dev/null|head -1|xargs wc -l 2>/dev/null|awk '{print $1+0}')
  echo "  capture attempt $t: rank0=$r0 lines rank1=$r1 lines" | tee -a "$LOG"
  [ "${r0:-0}" -gt 100 ] && [ "${r1:-0}" -gt 100 ] && { echo "  CAPTURE OK" | tee -a "$LOG"; break; }
done
# ---- APPLY (retry until ops CSV exists) ----
for t in $(seq 1 $TRIES); do
  echo "=== APPLY attempt $t $(date) ===" | tee -a "$LOG"; reset_both | tee -a "$LOG"
  WORST_AG_CT_MODE=apply timeout --signal=KILL 900 python3 ttnn/ttnn/distributed/ttrun.py \
    --tcp-interface ens5f0np0 --rank-binding $WK/bindings/worst_ag_2galaxy.yaml --mpi-args "$MPI" \
    bash "$WK/worst_ag_rank_wrapper.sh" >>"$LOG" 2>&1
  F=$(find "$WORST_AG_OUT" -name 'ops_perf_results*.csv' 2>/dev/null|head -1)
  [ -n "$F" ] && [ "$(stat -c %s "$F" 2>/dev/null||echo 0)" -gt 5000 ] && { echo "  APPLY OK -> $F" | tee -a "$LOG"; break; }
  echo "  apply attempt $t: no CSV yet" | tee -a "$LOG"
done
echo "=== DONE $(date) ===" | tee -a "$LOG"
find "$WORST_AG_OUT" -name 'ops_perf_results*.csv' | tee -a "$LOG"
killall_both
