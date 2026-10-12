#!/bin/bash
# t430: one broker job on blx01, one old/new A/B pair for the #427 cross-fade rewrite:
# test_decode_stage yuv420 5 s (MINIMAX_H3_VAE_PROFILE unset, seed 0 latents) on
#   old = $F/t423/b (5a8c59583a9, same VAE code as b0f45278d72^) and new = $F/t427/b (b0f45278d72),
# run old, new, old, new (one pytest process each, same box and clock).
# The script re-execs once in its own process group (broker reaping); pytest runs under
# 'timeout --foreground -k 10', never setsid under timeout.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t430; VENV=$F/t48/python_env
if [ -z "$T430_INNER" ]; then
  T430_INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
K="decode_stage and yuv420 and 5s and not 15s and not neighbor"
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 88 ] || { echo "[t430] / at $use%"; exit 5; }
gb=$(timeout 120 du -sxBG $F 2>/dev/null | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 450 ] || { echo "[t430] footprint ${gb}G"; exit 5; }
OUT=$T/out_ab; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
export MINIMAX_H3_MODEL_PATH=$F/models/MiniMax-H3
[ -e $MINIMAX_H3_MODEL_PATH/READ_OK ] || { echo "[t430] no READ_OK in $MINIMAX_H3_MODEL_PATH"; exit 4; }
r=$(readlink -f $MINIMAX_H3_MODEL_PATH); fs=$(findmnt -n -o FSTYPE -T "$r" 2>/dev/null)
case "$r" in /mnt/*) echo "[t430] weights under /mnt"; exit 4;; esac
case "$fs" in nfs*|cifs|smb*|fuse*|sshfs|lustre|ceph*|9p|wekafs|"") echo "[t430] weights on fs '${fs:-unknown}'"; exit 4;; esac
unset MINIMAX_H3_VAE_PROFILE
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
source $VENV/bin/activate || exit 3
[ "$(git -C $F/t423/b rev-parse --short=11 HEAD)" = 5a8c59583a9 ] || { echo "[t430] t423/b moved"; exit 3; }
[ "$(git -C $F/t427/b rev-parse --short=11 HEAD)" = b0f45278d72 ] || { echo "[t430] t427/b moved"; exit 3; }
SUM=$OUT/summary.log; : > $SUM
START=$(date +%s); rc=0; n=0
for tree in t423 t427 t423 t427; do
  n=$((n + 1)); W=$F/$tree/b; LOG=$OUT/run_${n}_$tree.log
  left=$(( 520 - ($(date +%s) - START) ))
  [ $left -ge 120 ] || { echo "[t430] arm $n $tree skipped: ${left}s left" | tee -a $SUM; break; }
  [ $left -le 400 ] || left=400
  cd $W || exit 3
  echo "[t430] arm=$n tree=$tree commit=$(git rev-parse --short=11 HEAD) host=$(hostname) load=$(cut -d' ' -f1-3 /proc/loadavg) $(date -u '+%F %T') UTC" | tee -a $SUM > $LOG
  TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools timeout --foreground -k 10 $left python -u -m pytest -sv -p no:cacheprovider \
    models/tt_dit/tests/models/minimax_h3/test_performance_vae_minimax_h3.py -k "$K" >> $LOG 2>&1 &
  PY=$!
  while kill -0 $PY 2>/dev/null; do
    sleep 5
    # CPU affinity of the pytest threads (tt-metal pins only with TT_METAL_NUMA_BASED_AFFINITY).
    for c in $(cat /proc/$PY/task/$PY/children 2>/dev/null); do
      grep -h Cpus_allowed_list /proc/$c/task/*/status 2>/dev/null | sort | uniq -c | tr -s ' ' > $OUT/aff_$n.txt
    done
    idle=$(( $(date +%s) - $(stat -c %Y $LOG) ))
    if [ $idle -gt 90 ]; then echo "[t430] watchdog: arm $n no output for ${idle}s, stopping" | tee -a $SUM $LOG; kill -TERM $PY; sleep 12; kill -KILL $PY 2>/dev/null; break; fi
  done
  wait $PY; arc=$?
  { grep -E "DECODE_STAGE|DECODE_SPLIT|passed|failed|Error" $LOG | tail -6; echo "affinity: $(tr '\n' ';' < $OUT/aff_$n.txt 2>/dev/null)"; echo "[t430] arm=$n exit=$arc"; } | tee -a $SUM
  [ $arc -eq 0 ] || { rc=$arc; [ $n -ge 2 ] || break; }
done
echo "T430_EXIT=$rc" | tee -a $SUM
exit $rc
