#!/usr/bin/env bash
# laneMK submit loop: ONE Slurm job = ONE galaxy = ONE op. For every op still lacking a
# VERDICT, grab one idle glx and run the op to completion on it; the job frees its node on
# exit. Repeat until every op has a VERDICT. Slurm is the refill: a dead job just leaves no
# verdict and is resubmitted next pass. No claims, no work-stealing, no held-idle nodes.
#
# Config via env (also exported to run_op): OPS_TSV IDMAP BUILD VENV LLK_HOME PYDIR OUT.
#   LANEMK_NODE_EXCLUDE (regex) drops known-poisoned racks (default the glx-110-c rack).
set -uo pipefail
: "${OPS_TSV:?} ${IDMAP:?} ${BUILD:?} ${VENV:?} ${LLK_HOME:?} ${PYDIR:?} ${OUT:?}"
export OPS_TSV IDMAP BUILD VENV LLK_HOME PYDIR OUT
RUN_OP="$(dirname "$0")/lanemk_run_op.sh"
EXCL="${LANEMK_NODE_EXCLUDE:-glx-110-c}"

remaining() { awk -F'\t' 'NF{print $1}' "$OPS_TSV" | while read -r op; do
  [ -s "$OUT/$op/$op-VERDICT.txt" ] || echo "$op"; done; }

pass=0
while :; do
  ops=$(remaining)
  [ -n "$ops" ] || { echo "ALL OPS HAVE VERDICTS"; break; }
  pass=$((pass + 1))
  echo "=== pass $pass: $(echo "$ops" | wc -l) ops without a verdict ==="
  # Pre-fetch a DISTINCT list of idle glx nodes for this pass (sinfo state lags a fresh
  # salloc, so re-querying per op would hand the same node to two ops). Zip ops to nodes.
  mapfile -t nodes < <(sinfo -h -t idle -N -o "%N %P" |
    awk -v x="$EXCL" '/glx/ && $1 !~ x {if(!s[$1]++) print $1" "$2}')
  [ "${#nodes[@]}" -ge 1 ] || { echo "no idle glx this pass; retrying"; sleep 60; continue; }
  pids=(); ni=0
  for op in $ops; do
    [ "$ni" -lt "${#nodes[@]}" ] || break        # more ops than idle nodes: next pass refills
    node="${nodes[$ni]%% *}"; part="${nodes[$ni]##* }"; ni=$((ni + 1))
    jid=$(salloc --no-shell --immediate=10 -J lanemk_op -w "$node" -p "$part" 2>&1 |
      grep -oE 'allocation [0-9]+' | grep -oE '[0-9]+')
    [ -n "$jid" ] || { echo "salloc failed on $node for $op"; continue; }
    echo "submit $op -> $node (job $jid)"
    ( srun --jobid="$jid" --overlap -w "$node" bash "$RUN_OP" "$op" > "$OUT/$op.log" 2>&1
      scancel "$jid" 2>/dev/null ) &            # frees the galaxy the instant the op finishes
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done       # let this pass's galaxies finish, then refill
done
