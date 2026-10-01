#!/bin/bash
# Runs the eval pack on blx03, one config at a time through submit.sh (one project job at a time),
# and stops at the first failed config so a drop is never followed by another submission.
# Launch detached on blx03:
#   ALLOW_4X8=1 setsid nohup bash /home/smarton/fasth3/t86/tmp/t86/eval_pack.sh > /var/tmp/fasth3/eval86/pack.log 2>&1 &
# Done marker: /var/tmp/fasth3/eval86/PACK_DONE (contents: ok, or the config that failed).
# ONLY=<cfg,cfg> runs a subset. DRY_RUN=1 runs run_eval.sh locally in dry mode (no broker, no device).
# Every config is a full 32-chip (4x8) run: ALLOW_4X8=1 is required and may be set only once the user
# allows 4x8 runs again.
W=${W:-/home/smarton/fasth3/t86}
# A dry run writes elsewhere by default so it can never mark a real config done.
OUT=${OUT:-$([ -n "$DRY_RUN" ] && echo /tmp/eval86_dry || echo /var/tmp/fasth3/eval86)}
SUBMIT=${SUBMIT:-/home/smarton/fasth3/tt-metal/tmp/blx03/submit.sh}
JOB_TIMEOUT=${JOB_TIMEOUT:-2700}
mkdir -p $OUT
rm -f $OUT/PACK_DONE
if [ -z "$DRY_RUN" ] && [ "$ALLOW_4X8" != 1 ]; then
  echo "refusing: 4x8 runs are not allowed (set ALLOW_4X8=1 only after the user lifts the ban)"
  echo "refused" > $OUT/PACK_DONE
  exit 2
fi
active() { tt-device-mcp status -j "$1" | grep -qiE 'Status: +(running|queued|pending)'; }
# fd 3 keeps the config list away from commands that read stdin.
while read -r cfg flags <&3; do
  if [ -n "$ONLY" ] && ! [[ ",$ONLY," == *",$cfg,"* ]]; then continue; fi
  if [ -s $OUT/$cfg/run.log ] && grep -q "RUN_EXIT\[$cfg\]=0" $OUT/$cfg/run.log; then
    echo "[pack] $cfg already done, skipping"; continue
  fi
  echo "[pack] $(date +%T) $cfg $flags"
  if [ -n "$DRY_RUN" ]; then
    mkdir -p $OUT/$cfg
    DRY_RUN=1 W=$W OUT=$OUT bash $W/tmp/t86/run_eval.sh $cfg $flags > $OUT/$cfg/run.log 2>&1
    continue
  fi
  while true; do
    sub=$($SUBMIT $JOB_TIMEOUT bash $W/tmp/t86/run_eval.sh $cfg $flags 2>&1); rc=$?
    [ $rc -ne 75 ] && break
    sleep 120  # another project job is active; wait our turn
  done
  id=$(echo "$sub" | grep -oE 'Job [0-9]+' | awk '{print $2}' | head -1)
  if [ -z "$id" ]; then echo "[pack] submit failed: $sub"; echo "submit:$cfg" > $OUT/PACK_DONE; exit 1; fi
  echo "[pack] $cfg job $id" | tee -a $OUT/jobs.txt
  while active $id; do sleep 60; done
  if ! grep -q "RUN_EXIT\[$cfg\]=0" $OUT/$cfg/run.log 2>/dev/null; then
    tt-device-mcp status -j $id | head -6
    echo "[pack] $cfg failed (job $id): stopping the pack; check the broker log for a chip drop before anything else"
    echo "failed:$cfg job=$id" > $OUT/PACK_DONE
    exit 1
  fi
done 3< <(grep -vE '^\s*(#|$)' $W/tmp/t86/configs.txt)
[ -e $OUT/PACK_DONE ] || echo ok > $OUT/PACK_DONE
echo "[pack] $(date +%T) $(cat $OUT/PACK_DONE)"
