#!/usr/bin/env bash
# Teardown sanity, multi-host — how a tt-metal job stops when the fault is on a REMOTE host (plan tt-metal #56902).
#
# run_teardown_multihost_tests.sh covers the launcher with a raw MPI probe. This one runs the tt-metal rank program
# (sanity_probe.py: ttnn, the DistributedContext, a mesh on the mock dual galaxy, no device) through tt-run, rank 0
# on the first host and rank 1 on the second, and checks every way a job ends (the table in
# tests/tt_metal/distributed/multiprocess/teardown_sanity/README.md): the job's exit code, the rank tt-run names,
# that every rank still alive closed its mesh (close_done), and that nothing is left on any host.
#
# Usage: run_teardown_multihost_context_tests.sh --hosts h1,h2[,...]
# Remote commands go over ssh (the SC4 lane), or srun --overlap with TEARDOWN_REMOTE=srun inside a Slurm allocation
# that holds the hosts (the compute nodes have no ssh keys). TEARDOWN_REMOTE_USER picks the ssh account.
set -uo pipefail

HOSTS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --hosts) HOSTS="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$HOSTS" ] || { echo "::error::--hosts is required"; exit 2; }

ROOT="${TT_METAL_HOME:-$(pwd)}"
TTRUN="$(command -v tt-run || echo tt-run)"
PY="${TT_SANITY_PYTHON:-python3}"
PROBE="$ROOT/tests/tt_metal/distributed/multiprocess/teardown_sanity/sanity_probe.py"
BINDING="$ROOT/tests/tt_metal/distributed/config/dual_bh_galaxy_rank_binding.yaml"
MAPPING="$ROOT/tt_metal/third_party/tt-cluster-descriptors/blackhole/dual_glx_2.5d_torus/dual_glx_2.5d_torus_cluster_desc_mapping.yaml"

# the mock dual galaxy is a 2-rank job; the first two hosts, so rank 1 is genuinely remote
HOSTS=$(tr ',' '\n' <<<"$HOSTS" | grep . | head -2 | paste -sd, -)
[ "$(tr ',' '\n' <<<"$HOSTS" | grep -c .)" -ge 2 ] || { echo "::error::need at least two hosts, got: $HOSTS"; exit 2; }
H0=$(cut -d, -f1 <<<"$HOSTS")
H1=$(cut -d, -f2 <<<"$HOSTS")

REMOTE_USER="${TEARDOWN_REMOTE_USER:-}"
remote_sh() {
  local host="$1"; shift
  if [ "${TEARDOWN_REMOTE:-ssh}" = srun ]; then srun --overlap -N1 -n1 -w "$host" bash -c "$*"
  else ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_USER:+$REMOTE_USER@}$host" "$*"; fi
}

# tt-run starts one app context per rank, so the ranks are placed with a rankfile: rank 0 on the first host, rank 1
# on the second. It sits under the working directory with a plain relative name, which is how tt-run hands it to
# PRRTE (--map-by rankfile:file=<path>; a ../ path is rejected). tt-run forwards only TT_* variables to remote ranks,
# so the probe's settings go through -x.
cd "$ROOT"
mkdir -p generated/teardown_sanity
RANKFILE="generated/teardown_sanity/rankfile_$$"
printf 'rank 0=%s slot=0\nrank 1=%s slot=0\n' "$H0" "$H1" > "$RANKFILE"
FORWARD="-x PROBE_SCENARIO -x PROBE_PHASE -x PROBE_COLL -x PROBE_VICTIM -x PROBE_PIDDIR -x PROBE_SLOW_S"
TCP_IF_ARGS=()
[ -n "${TT_SANITY_TCP_INTERFACE:-}" ] && TCP_IF_ARGS=(--tcp-interface "$TT_SANITY_TCP_INTERFACE")
PIDDIR="/tmp/teardown_ctx_$$"  # each rank writes its pid and phase markers on its own host

FAIL=0
LOG=""
# run <name> <action> [KEY=VAL ...]: action is - (none), INT, TERM, INT+INT (tt-run) or rank1:TERM (the remote rank)
run() {
  local name="$1" action="$2"; shift 2
  LOG="$(mktemp)"
  remote_sh "$H0" "rm -rf $PIDDIR; mkdir -p $PIDDIR" >/dev/null 2>&1
  remote_sh "$H1" "rm -rf $PIDDIR; mkdir -p $PIDDIR" >/dev/null 2>&1
  env PROBE_PIDDIR="$PIDDIR" "$@" timeout 240 "$TTRUN" \
    --rank-binding "$BINDING" --mock-cluster-rank-binding "$MAPPING" "${TCP_IF_ARGS[@]}" \
    --mpi-args "--allow-run-as-root --oversubscribe --rankfile $RANKFILE $FORWARD" \
    "$PY" "$PROBE" >"$LOG" 2>&1 &
  local tt=$!
  if [ "$action" != "-" ]; then
    for _ in $(seq 1 180); do  # both ranks past open
      remote_sh "$H0" "test -e $PIDDIR/rank_0.coll_ready" 2>/dev/null && remote_sh "$H1" "test -e $PIDDIR/rank_1.coll_ready" 2>/dev/null && break
      sleep 1
    done
    sleep 3
    case "$action" in
      INT|TERM) kill -"$action" "$tt" ;;
      INT+INT) kill -INT "$tt"; sleep 3; kill -INT "$tt" ;;
      rank1:TERM) remote_sh "$H1" "kill -TERM \$(cat $PIDDIR/rank_1.pid)" ;;
    esac
  fi
  wait "$tt"
  RC=$?
  sleep 2
}

leftovers() {
  local n=0 c
  for h in "$H0" "$H1"; do
    c=$(remote_sh "$h" "pgrep -cf 'sanity_prob[e].py' || true" 2>/dev/null)
    [[ "$c" =~ ^[0-9]+$ ]] || c=0
    n=$((n + c))
  done
  echo "$n"
}

# check <name> <exit code> <ranks that must have closed, e.g. "0 1"> [stop line regex] [extra log regex]
check() {
  local name="$1" want="$2" closed="$3" stop="${4:-}" extra="${5:-}" why="" left
  [ "$RC" -eq "$want" ] || why="exit $RC, expected $want"
  for r in $closed; do
    grep -aq "probe r=$r [^]]*\] close_done" "$LOG" || why="${why:+$why; }rank $r did not close its mesh"
  done
  if [ -n "$stop" ] && ! grep -aqE "\[tt-stop\] rank $stop" "$LOG"; then why="${why:+$why; }no [tt-stop] line 'rank $stop'"; fi
  if [ -n "$extra" ] && ! grep -aqE "$extra" "$LOG"; then why="${why:+$why; }log lacks /$extra/"; fi
  left=$(leftovers)
  [ "$left" -eq 0 ] || why="${why:+$why; }$left probe rank(s) left on the hosts"
  if [ -z "$why" ]; then
    echo "PASS  $name  (exit $RC; closed: ${closed:-none}; nothing left)"
  else
    echo "::error::FAIL  $name  ($why)"; FAIL=1
    grep -aE "\[tt-stop\]|\[tt-run\]|close_|Error" "$LOG" | tail -20 | sed 's/^/    /'
    for h in "$H0" "$H1"; do remote_sh "$h" "pkill -9 -f 'sanity_prob[e].py'" >/dev/null 2>&1 || true; done
  fi
  grep -aE "\[tt-stop\] rank|job end:" "$LOG" | sed -E 's/^.*(\[tt-stop\]|\[tt-run\] job end)/\1/' | sed 's/^/      /'
  rm -f "$LOG"
}

echo "=== rank 0 on $H0, rank 1 on $H1; tt-run=$TTRUN"

run clean -
check "clean" 0 "0 1" "" "outcome=clean exit=0"

run ctrl_c INT PROBE_COLL=send_no_recv
check "ctrl_c_at_ttrun" 130 "0 1" "" "outcome=stopped exit=130"

run sigterm_ttrun TERM PROBE_COLL=send_no_recv
check "sigterm_at_ttrun" 143 "0 1" "" "outcome=stopped exit=143"

run sigterm_remote_rank rank1:TERM PROBE_COLL=barrier_slow PROBE_SLOW_S=60
check "sigterm_to_remote_rank" 143 "0 1" "1 on $H1 .*received SIGTERM from pid"

run exit1_remote - PROBE_SCENARIO=exit1
check "sys_exit_1_on_remote_rank" 1 "0 1" "1 on $H1 .*sys.exit\(1\)"

run raise_remote - PROBE_SCENARIO=raise
check "exception_on_remote_rank" 1 "0 1" "1 on $H1 .*Python exception RuntimeError"

run sigkill_remote - PROBE_SCENARIO=sigkill
check "remote_rank_sigkilled" 137 "0" "1 on $H1 .*killed by signal 9"

run segv_remote - PROBE_SCENARIO=segv
check "remote_rank_segfault" 139 "0" "1 on $H1 .*killed by signal 11"

run stuck_remote INT+INT PROBE_SCENARIO=ignore_sigterm PROBE_COLL=barrier
check "remote_rank_stuck_second_ctrl_c" 130 "0" "" "job forced down.*second SIGINT"

# a peer hangs in a collective on the remote host: the operation timeout (P16) stops the job from rank 0
run hang_remote - PROBE_COLL=allgather_skip TT_METAL_OPERATION_TIMEOUT_SECONDS=15
check "remote_rank_hangs_operation_timeout" 1 "0 1" "0 on $H0 .*all_gather did not complete"

for h in "$H0" "$H1"; do remote_sh "$h" "rm -rf $PIDDIR" >/dev/null 2>&1 || true; done
rm -f "$RANKFILE"
[ "$FAIL" -eq 0 ] && echo "=== teardown context multihost: all cases passed" || echo "=== teardown context multihost: FAILURES above"
exit "$FAIL"
