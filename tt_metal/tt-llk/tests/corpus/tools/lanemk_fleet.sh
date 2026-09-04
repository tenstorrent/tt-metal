#!/usr/bin/env bash
# laneMK exabox fleet launcher — runs on the exabox LOGIN node. Grabs idle glx hosts
# (one salloc each, --immediate), starts one work-stealing worker per host, supervises
# until every op has a VERDICT, then reaps ONLY its own allocations. Etiquette: idle glx
# only, only as many as there are ops, never touches drain/reserved/other jobs.
#
#   lanemk_fleet.sh <ops_tsv> <root> <build> <venv> <llk_home> <pydir> <worker.sh> <idmap> [max_hosts] [band_bits]
set -uo pipefail
OPS="${1:?ops_tsv}"; ROOT="${2:?root(work dir on /data)}"; BUILD="${3:?build}"; VENV="${4:?venv}"
LLKH="${5:?llk_home}"; PYDIR="${6:?pydir}"; WORKER="${7:?worker.sh}"; IDMAP="${8:?idmap}"; MAXH="${9:-999}"; BB="${10:-28}"
CLAIMS="$ROOT/claims"; OUTD="$ROOT/out"; mkdir -p "$CLAIMS" "$OUTD"
NOPS=$(grep -cve '^[[:space:]]*$' "$OPS")
JOB=lanemk_fleet
echo "=== fleet: $NOPS ops, root=$ROOT, band_bits=$BB ==="

# idle glx nodes, one per line, deduped (customer deployment is live -> re-check at grab time).
# NODE_EXCLUDE (regex) drops known-poisoned racks whose salloc times out (e.g. bh_sc36_5 =
# the glx-110-c rack). Default excludes it; override with LANEMK_NODE_EXCLUDE.
EXCL="${LANEMK_NODE_EXCLUDE:-glx-110-c}"
mapfile -t NODES < <(sinfo -h -t idle -N -o "%N %P" | awk -v x="$EXCL" '/glx/ && $1 !~ x{sub(/\*/,"",$2); if(!s[$1]++) print $1" "$2}' | head -n "$MAXH")
echo "idle glx available: ${#NODES[@]}"
[ "${#NODES[@]}" -ge 1 ] || { echo "NO_IDLE_GLX -- abort"; exit 2; }

JIDS=()
for entry in "${NODES[@]}"; do
  node="${entry%% *}"; part="${entry##* }"
  out=$(salloc --no-shell --immediate=10 -J "$JOB" -w "$node" -p "$part" 2>&1)
  jid=$(echo "$out" | grep -oE 'Granted job allocation [0-9]+' | grep -oE '[0-9]+')
  if [ -n "$jid" ]; then
    JIDS+=("$jid")
    echo "grabbed $node (job $jid, part $part)"
    srun --jobid="$jid" --overlap -w "$node" bash -c \
      "ulimit -u \$(ulimit -Hu) 2>/dev/null; bash '$WORKER' '$OPS' '$CLAIMS' '$OUTD' '$BUILD' '$VENV' '$LLKH' '$PYDIR' '$BB' '$IDMAP'" \
      > "$OUTD/worker-$node.log" 2>&1 &
  else
    echo "skip $node (not grantable: $(echo "$out" | tail -1))"
  fi
done
echo "workers launched on ${#JIDS[@]} hosts"

reap(){ for j in "${JIDS[@]}"; do scancel "$j" 2>/dev/null; done; echo "reaped ${#JIDS[@]} allocations"; }
trap reap EXIT

# supervise until every op has a VERDICT (or all workers exited)
while :; do
  done_n=$(find "$OUTD" -name VERDICT 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%SZ)] verdicts $done_n/$NOPS ; running workers $(jobs -r | wc -l)"
  [ "$done_n" -ge "$NOPS" ] && { echo "ALL VERDICTS IN"; break; }
  [ "$(jobs -r | wc -l)" -eq 0 ] && { echo "ALL WORKERS EXITED (verdicts $done_n/$NOPS)"; break; }
  sleep 60
done
echo "=== verdicts ==="; for v in "$OUTD"/*/VERDICT; do [ -f "$v" ] && cat "$v"; done
