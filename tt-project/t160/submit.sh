#!/bin/bash
# submit.sh: run on exabox-login. Re-checks the dit idle >= 2 h rule (qualify.sh) right before taking a node,
# submits job.sbatch there as one self-ending job, cancels it if still PENDING after 60 s (node taken meanwhile),
# and logs to $F/t160/alloc.log. One project job at a time: refuses if any smarton-fasth3 job is queued or running.
# usage: MODE=build|e2e bash submit.sh [extra sbatch args]
# MODE=e2e (device): --exclusive, 10 min limit (600 s reservation cap; never raise it, split the job instead).
# MODE=build (no device): whole node, 45 min limit.
# Our own jobs reset a node's LastBusyTime, so consecutive jobs land on different qualifying nodes.
F=/data/smarton/fasth3; L=$F/t160/alloc.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
[ -z "$(squeue -h -u $USER -n smarton-fasth3-ltx25 -o %i)" ] || { log "a smarton-fasth3 job is already queued/running"; exit 3; }
MODE=${MODE:-e2e}
case $MODE in
  e2e) TLIM=${TLIM:-00:10:00}; X="--exclusive"
       [ $(( 10#${TLIM:3:2} + 60*10#${TLIM:0:2} )) -le 10 ] || { log "refused: e2e TLIM $TLIM over the 10 min cap"; exit 5; } ;;
  build) TLIM=${TLIM:-00:45:00}; X="--exclusive" ;;  # Slurm lists these nodes as CPUTot=1 RealMemory=1
  *) log "unknown MODE=$MODE"; exit 5 ;;
esac
c=$(bash $F/qualify.sh) || { log "no qualifying dit node"; exit 1; }
read -r m n lb <<<"$(head -1 <<<"$c")"
proof=$(scontrol show node $n | grep -E 'NodeName|State=|LastBusyTime|Partitions')
p=$(grep -oP 'Partitions=\K[^, ]+' <<<"$proof")
j=$(sbatch --parsable -p $p -w $n --time=$TLIM $X --export=ALL,MODE=$MODE "$@" $F/job.sbatch) || { log "sbatch failed on $n"; exit 2; }
log "submitted job $j node $n partition $p idle_min=$m LastBusyTime=$lb mode=$MODE tlim=$TLIM"
sleep 60
s=$(squeue -h -j $j -o %T)
if [ "$s" = PENDING ]; then scancel $j; log "job $j PENDING after 60 s: cancelled our own job"; exit 4; fi
log "job $j state after 60 s: ${s:-gone}"
echo $j
