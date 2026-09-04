#!/usr/bin/env bash
# laneMK exabox fleet launcher — runs on the exabox LOGIN node. Grabs idle glx hosts
# (one salloc each, --immediate), starts one work-stealing worker per host, and ELASTICALLY
# re-harvests newly-idle glx during the run (up to max_hosts) so nodes that free up mid-sweep
# join the same queue. Supervises until every op has a VERDICT, then reaps ONLY its own
# allocations. Etiquette: idle glx only, never touches drain/reserved/other jobs.
#
#   lanemk_fleet.sh <ops_tsv> <root> <build> <venv> <llk_home> <pydir> <worker.sh> <idmap> [max_hosts] [band_bits]
set -uo pipefail
OPS="${1:?ops_tsv}"; ROOT="${2:?root(work dir on /data)}"; BUILD="${3:?build}"; VENV="${4:?venv}"
LLKH="${5:?llk_home}"; PYDIR="${6:?pydir}"; WORKER="${7:?worker.sh}"; IDMAP="${8:?idmap}"; MAXH="${9:-999}"; BB="${10:-28}"
CLAIMS="$ROOT/claims"; OUTD="$ROOT/out"; mkdir -p "$CLAIMS" "$OUTD"
NOPS=$(grep -cve '^[[:space:]]*$' "$OPS")
JOB=lanemk_fleet
# NODE_EXCLUDE (regex) drops known-poisoned racks whose salloc times out (e.g. bh_sc36_5 =
# the glx-110-c rack). Default excludes it; override with LANEMK_NODE_EXCLUDE.
EXCL="${LANEMK_NODE_EXCLUDE:-glx-110-c}"
echo "=== fleet: $NOPS ops, root=$ROOT, band_bits=$BB, max_hosts=$MAXH ==="

JIDS=(); declare -A HELD   # HELD[node]=1 once we own it, so re-harvest never double-grabs a node

# idle glx nodes (customer deployment is live -> re-scan every time); one "node part" per line, deduped.
scan_idle(){ sinfo -h -t idle -N -o "%N %P" | awk -v x="$EXCL" '/glx/ && $1 !~ x{sub(/\*/,"",$2); if(!s[$1]++) print $1" "$2}'; }

# grab one node + launch its worker; no-op if already held or at the max_hosts cap.
grab_and_launch(){
  local node="$1" part="$2" out jid
  [ -n "${HELD[$node]:-}" ] && return 0
  [ "${#JIDS[@]}" -ge "$MAXH" ] && return 0
  out=$(salloc --no-shell --immediate=10 -J "$JOB" -w "$node" -p "$part" 2>&1)
  jid=$(echo "$out" | grep -oE 'Granted job allocation [0-9]+' | grep -oE '[0-9]+')
  [ -n "$jid" ] || { echo "skip $node (not grantable: $(echo "$out" | tail -1))"; return 0; }
  JIDS+=("$jid"); HELD[$node]=1
  echo "grabbed $node (job $jid, part $part) [hosts ${#JIDS[@]}/$MAXH]"
  srun --jobid="$jid" --overlap -w "$node" bash -c \
    "ulimit -u \$(ulimit -Hu) 2>/dev/null; bash '$WORKER' '$OPS' '$CLAIMS' '$OUTD' '$BUILD' '$VENV' '$LLKH' '$PYDIR' '$BB' '$IDMAP'" \
    > "$OUTD/worker-$node.log" 2>&1 &
}

# harvest: grab every currently-idle glx we don't already hold (up to max_hosts). Skipped once
# every op is claimed — no point grabbing a node that would find nothing to steal.
harvest(){
  [ "$(ls "$CLAIMS" 2>/dev/null | wc -l)" -lt "$NOPS" ] || return 0
  local node part; while read -r node part; do [ -n "$node" ] && grab_and_launch "$node" "$part"; done < <(scan_idle)
}

harvest
[ "${#JIDS[@]}" -ge 1 ] || { echo "NO_IDLE_GLX -- abort"; exit 2; }
echo "workers launched on ${#JIDS[@]} hosts"

reap(){ for j in "${JIDS[@]}"; do scancel "$j" 2>/dev/null; done; echo "reaped ${#JIDS[@]} allocations"; }
trap reap EXIT

# supervise until every op has a VERDICT (or all workers exited); re-harvest newly-idle glx each tick.
while :; do
  done_n=$(find "$OUTD" -name VERDICT 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%SZ)] verdicts $done_n/$NOPS ; hosts ${#JIDS[@]} ; running workers $(jobs -r | wc -l)"
  [ "$done_n" -ge "$NOPS" ] && { echo "ALL VERDICTS IN"; break; }
  [ "$(jobs -r | wc -l)" -eq 0 ] && { echo "ALL WORKERS EXITED (verdicts $done_n/$NOPS)"; break; }
  harvest   # elastic: pull in nodes that freed up mid-run into the same work-stealing queue
  sleep 60
done
echo "=== verdicts ==="; for v in "$OUTD"/*/VERDICT; do [ -f "$v" ] && cat "$v"; done
