#!/usr/bin/env bash
# Connected 2-galaxy worst-AllGather: this host (rank0) runs the AllGather, PEER idles holding the fabric.
set -u
WK=/home/ppopovic/tt-metal/models/demos/common/prefill/fabric_debug
PEER=bh-glx-110-d09u02; LOCAL=bh-glx-110-d09u08; TAG=worst_ag_pipe
LOG=$WK/logs/${TAG}.log; export WORST_AG_OUT=/data/ppopovic/prof_out/$TAG
TT_METAL_HOME=/home/ppopovic/tt-metal; export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME:$WK"
source /home/ppopovic/tt-metal/python_env/bin/activate; cd "$TT_METAL_HOME"
rm -rf "$WORST_AG_OUT"; mkdir -p "$WORST_AG_OUT"
rm -f "$WK"/__pycache__/worst_allgather_test*.pyc
killall_both(){ for pid in $(ps -u ppopovic -o pid,command|grep -E 'ttrun\.py|prterun|prted|worst_ag_rank_wrapper|worst_allgather_test'|grep -v grep|awk '{print $1}'); do kill -9 $pid 2>/dev/null; done
  timeout 20 ssh $PEER "for pid in \$(ps -u ppopovic -o pid,command|grep -E 'ttrun|prted|prterun|fabric_erisc|worst_allgather_test'|grep -v grep|awk '{print \$1}'); do kill -9 \$pid 2>/dev/null; done" 2>/dev/null; }
echo "=== worst-AG connected 2-galaxy clean+reset $(date) ===" | tee "$LOG"
killall_both; sleep 3
timeout 340 tt-smi -glx_reset_auto >$WK/logs/rst_${TAG}_local.txt 2>&1 & A=$!
timeout 360 ssh $PEER 'timeout 340 tt-smi -glx_reset_auto' >$WK/logs/rst_${TAG}_peer.txt 2>&1 & B=$!
wait $A; wait $B; sleep 6
echo "  reset local=$(grep -c 'Re-initialized 32 boards' $WK/logs/rst_${TAG}_local.txt) peer=$(grep -c 'Re-initialized 32 boards' $WK/logs/rst_${TAG}_peer.txt)" | tee -a "$LOG"
echo "=== launch ttrun (rank0=$LOCAL, rank1=$PEER) $(date) ===" | tee -a "$LOG"
timeout --signal=KILL 1200 python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface ens5f0np0 --rank-binding $WK/bindings/worst_ag_2galaxy.yaml \
  --mpi-args "--host ${LOCAL}:1,${PEER}:1 --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x TT_METAL_HOME -x WORST_AG_OUT" \
  bash "$WK/worst_ag_rank_wrapper.sh" >>"$LOG" 2>&1
echo "=== exit rc=$? $(date) ===" | tee -a "$LOG"
find "$WORST_AG_OUT" -name 'ops_perf_results*.csv' | tee -a "$LOG"
killall_both
