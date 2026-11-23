#!/bin/bash
# Starts runs 8c, 8d, 8e on the three bh-30 boards that run 8b (board 2) does not use:
#   8c -> board 1 (IRD job 164868, ssh 47032), 8d -> board 3 (164870, 47033), 8e -> board 0 (164867, 47031).
# Runs on the bh-30 HOST (setsid nohup) and reaches the containers with ssh -p <port> localhost, using the
# private known_hosts in this folder (ssh-keyscan 2026-11-21; same ED25519 key the Mac knows for [bh-30]:4703x).
# Waits until
#  (a) the agent of run 8b has finished (claude_exit_code in 8b logs/run_meta.txt, or no "claude -p" in the
#      board-2 container): 8b's ws/hal-settings.json was written before the 8c-8e run dirs existed, so it
#      does not deny them; starting after its agent has ended keeps 8b clean;
#  (b) nothing of mine runs in the three target containers (board 0 still had the asserts-off rebuild).
# Then runs launch_8x.sh in each container 30 s apart (RUN_ID has 1 s granularity; step 5/9 Permissions
# comes ~2 min after the start, so each run's settings deny the other two run dirs), then check_cross_deny_8x.sh.
# `touch tools/run8cde/STOP` before the start = nothing is launched. Progress: orchestrate_progress.txt
set -u
B=/proj_sw/user_dev/ctr-sbratic
S=$B/tools/run8cde
P=$S/orchestrate_progress.txt
R8B=$B/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5
SSHO="-n -o BatchMode=yes -o UserKnownHostsFile=$S/known_hosts -o StrictHostKeyChecking=yes -o ConnectTimeout=30"
PAT='[r]un_hal_gen.sh|[c]laude -p|[p]ytest|[r]un_full_gdb|[r]un_hal_gdb|[d]river_|[b]uild_counts|[t]t-smi'
RUNS="8c 1 47032
8d 3 47033
8e 0 47031"
now() { date -u +%FT%TZ; }
agent8b_done() {
  grep -q '^claude_exit_code=' $R8B/logs/run_meta.txt && return 0
  local out; out=$(ssh $SSHO -p 47030 localhost 'ps -eo args') || return 1
  ! grep -q '[c]laude -p' <<< "$out"
}
idle() { local out; out=$(ssh $SSHO -p $1 localhost 'ps -eo args') || return 1; ! grep -qE "$PAT" <<< "$out"; }

echo "orchestrator start $(now), pid $$, host $(hostname)" >> $P
N=0
while :; do
  [[ -e $S/STOP ]] && { echo "STOP file found, nothing launched $(now)" >> $P; exit 1; }
  A=no; agent8b_done && A=yes
  I=""
  while read -r n b p; do if idle $p; then I+="$n/board$b:idle "; else I+="$n/board$b:busy "; fi; done <<< "$RUNS"
  [[ $A == yes && $I != *busy* ]] && break
  (( N % 15 == 0 )) && echo "waiting $(now): 8b agent finished=$A; target containers: $I" >> $P
  N=$((N+1)); sleep 60
done
echo "gate open $(now): 8b agent finished=$A; target containers: $I" >> $P

FIRST=1
while read -r n b p; do
  (( FIRST )) || sleep 30
  FIRST=0
  ssh $SSHO -p $p localhost "bash $S/launch_8x.sh $n $b" > $S/launch_$n.out 2>&1 < /dev/null
  echo "$n (board $b, ssh $p) launch exit $? $(now)" >> $P
done <<< "$RUNS"

bash $S/check_cross_deny_8x.sh $R8B 8c 8d 8e > $S/cross_deny.txt 2>&1 < /dev/null
echo "cross-deny / config check exit $? $(now) (cross_deny.txt)" >> $P
echo "orchestrator end $(now)" >> $P
