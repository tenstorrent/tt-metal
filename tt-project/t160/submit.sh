#!/bin/bash
# submit.sh: run on exabox-login. Re-checks the dit idle >= 2 h rule (qualify.sh) right before taking a node,
# submits job.sbatch there as one self-ending job, cancels it if still PENDING after 60 s (node taken meanwhile),
# and logs to $F/t160/alloc.log. One project job at a time: refuses if any smarton-fasth3 job is queued or running.
# usage: TLIM=03:00:00 bash submit.sh [extra sbatch args]
F=/data/smarton/fasth3; L=$F/t160/alloc.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
[ -z "$(squeue -h -u $USER -n smarton-fasth3-ltx25 -o %i)" ] || { log "a smarton-fasth3 job is already queued/running"; exit 3; }
c=$(bash $F/qualify.sh) || { log "no qualifying dit node"; exit 1; }
read -r m n lb <<<"$(head -1 <<<"$c")"
proof=$(scontrol show node $n | grep -E 'NodeName|State=|LastBusyTime|Partitions')
p=$(grep -oP 'Partitions=\K[^, ]+' <<<"$proof")
j=$(sbatch --parsable -p $p -w $n --time=${TLIM:-03:00:00} "$@" $F/job.sbatch) || { log "sbatch failed on $n"; exit 2; }
log "submitted job $j node $n partition $p idle_min=$m LastBusyTime=$lb tlim=${TLIM:-03:00:00}"
sleep 60
s=$(squeue -h -j $j -o %T)
if [ "$s" = PENDING ]; then scancel $j; log "job $j PENDING after 60 s: cancelled our own job"; exit 4; fi
log "job $j state after 60 s: ${s:-gone}"
echo $j
