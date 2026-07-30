#!/usr/bin/env bash
# PARTITIONED-PARALLEL proof-of-concept on the 128 hardcap CNF (cadical-single = TIMEOUT>400s; gimsatul-t32 ~67s).
# Partition axis = stage-0 placement (vars 1..144, the first exactly-one group). For a sample of buckets, pin that
# var (append unit clause) and solve with PLAIN cadical, all buckets in PARALLEL. Measures: per-bucket verdict+time,
# skew (how many SAT), and parallel wall = max over buckets. Tests whether partitioned pure-cadical rivals gimsatul.
set -u
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
CAD="$SC/cadical_src/build/cadical"; RAW="$SC/2x4_128_hardcap.cnf"
RES="$SC/partition_results.txt"; : > "$RES"
WORK="$SC/partition_work"; rm -rf "$WORK"; mkdir -p "$WORK"
read -r _ _ NV NC < <(head -1 "$RAW")   # p cnf NV NC
NC2=$((NC+1))
# sample a spread of stage-0 buckets across the 144 slots (every 8th -> 18 buckets, ~fits cores)
BUCKETS=$(seq 1 8 144)
echo "buckets sampled: $(echo $BUCKETS | wc -w) (of 144); raw NV=$NV NC=$NC" | tee -a "$RES"
declare -A PID START
launch() { local g=$1; local f="$WORK/b_$g.cnf"
  # copy CNF, bump clause count, append unit clause 'g 0' pinning stage-0 -> slot g
  { printf "p cnf %s %s\n" "$NV" "$NC2"; tail -n +2 "$RAW"; printf "%s 0\n" "$g"; } > "$f"
  ( s=$(date +%s%3N); r=$(timeout 400 "$CAD" -q "$f" 2>/dev/null | grep -aoE "^s [A-Z]+" | head -1); e=$(date +%s%3N)
    echo "$g $(( (e-s)/1000 )) ${r:-TIMEOUT}" > "$WORK/r_$g.txt" ) &
  PID[$g]=$!
}
t0=$(date +%s)
for g in $BUCKETS; do launch "$g"; done
for g in $BUCKETS; do wait "${PID[$g]}" 2>/dev/null; done
wall=$(( $(date +%s) - t0 ))
echo "--- per-bucket (slot time_s verdict) ---" | tee -a "$RES"
sat=0; tmax=0; firstsat=999999
for g in $BUCKETS; do
  line=$(cat "$WORK/r_$g.txt" 2>/dev/null); echo "  $line" | tee -a "$RES"
  tt=$(echo "$line" | awk '{print $2}'); v=$(echo "$line" | awk '{print $3}')
  [ "$v" = "SATISFIABLE" ] && { sat=$((sat+1)); [ "$tt" -lt "$firstsat" ] && firstsat=$tt; }
  [ "${tt:-0}" -gt "$tmax" ] && tmax=$tt
done
echo "SUMMARY: buckets=$(echo $BUCKETS|wc -w) SAT=$sat  first_SAT=${firstsat}s  slowest_bucket=${tmax}s  parallel_wall=${wall}s" | tee -a "$RES"
echo "  (compare: cadical-single >400s TIMEOUT ; gimsatul-t32 ~67s)" | tee -a "$RES"
echo "[$(date -u +%T)] PARTITION TEST DONE" | tee -a "$RES"
