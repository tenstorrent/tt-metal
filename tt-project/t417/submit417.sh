#!/bin/bash
# t417 submit on exabox-login: same rules as ../submit.sh (dit node idle >= 2 h via qualify.sh, one self-ending job,
# cancel if still PENDING after 60 s), but for a t417 sbatch. usage: TLIM=hh:mm:ss bash submit417.sh <sbatch> [args]
F=/data/smarton/fasth3; L=$F/t417/alloc.log; SB=$1; shift
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
[ -z "$(squeue -h -u $USER -o %i | grep .)" ] || { log "a job of ours is already queued/running"; exit 3; }
[ -n "$TLIM" ] || { log "TLIM required"; exit 5; }
c=$(bash $F/qualify.sh) || { log "no qualifying dit node"; exit 1; }
read -r m n lb <<<"$(head -1 <<<"$c")"
p=$(scontrol show node $n | grep -oP 'Partitions=\K[^, ]+')
j=$(sbatch --parsable -p $p -w $n --time=$TLIM --exclusive "$@" $SB) || { log "sbatch failed on $n"; exit 2; }
log "submitted job $j sb=$(basename $SB) node $n partition $p idle_min=$m LastBusyTime=$lb tlim=$TLIM"
sleep 60
s=$(squeue -h -j $j -o %T)
if [ "$s" = PENDING ]; then scancel $j; log "job $j PENDING after 60 s: cancelled our own job"; exit 4; fi
log "job $j state after 60 s: ${s:-gone}"
echo $j
