#!/usr/bin/env bash
# Teardown sanity, multi-host — the tt-metal DistributedContext (plan tt-metal #56902, lane L4 across hosts).
# The launcher-level multi-host script (run_teardown_multihost_tests.sh, the base PR) proves PRRTE reaps a
# fault on a remote host. This one asks the next question: when the fault lands *inside* a DistributedContext
# collective on a REMOTE host, does the context's own teardown — the MPI_ERRORS_RETURN errhandler (P20) and the
# per-collective operation timeout (P22) — tear the whole job down cleanly, the same way it does on one host?
#
# It runs the DistributedContext probe (sanity_probe.py, the ttnn rank program) across the hosts through tt-run,
# one rank per host on the mock dual galaxy (no device), and asserts the teardown shape: the right exit code and
# no rank or launcher process left on ANY host. Needs more than one host, so it runs on SC4.
#
# The MPI job is genuinely multi-host: one rank per host, the DistributedContext collectives go over real TCP
# between the hosts, and the teardown (errhandler, per-collective timeout, abort) happens across them. Only the
# DEVICE is mocked (--mock-cluster-rank-binding, no device opened), which is fine here because the teardown logic is
# pure MPI and does not depend on the device; the mesh-graph descriptor + cluster mapping below only give the
# control plane a logical topology to parse. It is a Blackhole mock (tt-blaze targets Blackhole), the same dual
# BH galaxy the single-host DistributedContext rows use. It is NOT a negative test.
#
# Usage: run_teardown_multihost_context_tests.sh --hosts h1,h2,h3,h4
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
# The same Blackhole dual-galaxy mock the single-host DistributedContext rows use (two ranks of 8x4, no device);
# the only new variable here is that the two ranks land on two different hosts.
MGD="$ROOT/tt_metal/fabric/mesh_graph_descriptors/dual_bh_galaxy_torus_xy_graph_descriptor.textproto"
MAPPING="$ROOT/tt_metal/third_party/tt-cluster-descriptors/blackhole/dual_glx_2.5d_torus/dual_glx_2.5d_torus_cluster_desc_mapping.yaml"

# the mock dual galaxy is a 2-rank job; take the first two hosts so rank 1 is genuinely on a remote host.
HOSTS=$(tr ',' '\n' <<<"$HOSTS" | grep . | head -2 | paste -sd, -)
[ "$(tr ',' '\n' <<<"$HOSTS" | grep -c .)" -ge 2 ] || { echo "::error::need at least two hosts, got: $HOSTS"; exit 2; }
NP=2
VICTIM=1 # rank 1, on the second host (remote)
MPI_ARGS="--allow-run-as-root --oversubscribe --mca btl self,tcp"
FAIL=0

# tt-run in new mode: phase 1 solves the mock quad descriptor across --hosts, then launches the probe one rank
# per host. No PRRTE --timeout here on purpose: the hang case must be bounded by the context's own timeout, not
# the launcher wall clock.
run_probe() { # extra "KEY=VAL ..." env
  env "$@" timeout 180 "$TTRUN" \
    --hosts "$HOSTS" \
    --mesh-graph-descriptor "$MGD" \
    --mock-cluster-rank-binding "$MAPPING" \
    --mpi-args "$MPI_ARGS" \
    --force-rediscovery \
    "$PY" "$PROBE"
}

# how many probe ranks are still alive on any host (should be 0 after tt-run returns)
leftovers() {
  local n=0
  for h in $(tr ',' ' ' <<<"$HOSTS"); do
    n=$((n + $(ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "pgrep -cf sanity_probe.py" 2>/dev/null || echo 0)))
  done
  echo "$n"
}
reap() { for h in $(tr ',' ' ' <<<"$HOSTS"); do ssh -o BatchMode=yes "$h" "pkill -9 -f sanity_probe.py" 2>/dev/null || true; done; }

check() { # name  expect_nonzero(0|1)  actual_rc  [max_wall_s]
  local name="$1" want_nonzero="$2" rc="$3" max_wall="${4:-}" wall="${5:-}"
  sleep 2; local left; left=$(leftovers)
  local ok=1 why=""
  if [ "$want_nonzero" = "1" ] && [ "$rc" -eq 0 ]; then ok=0; why="expected non-zero, got 0"; fi
  if [ "$want_nonzero" = "0" ] && [ "$rc" -ne 0 ]; then ok=0; why="expected 0, got $rc"; fi
  if [ "$left" -ne 0 ]; then ok=0; why="$why; $left probe rank(s) left on the hosts"; fi
  if [ -n "$max_wall" ] && [ -n "$wall" ] && [ "$wall" -ge "$max_wall" ]; then
    ok=0; why="$why; ran ${wall}s >= ${max_wall}s (the context timeout did not bound it; the wall clock did)"; fi
  if [ "$ok" = "1" ]; then echo "PASS  $name  (exit $rc${wall:+, ${wall}s}, nothing left)";
  else echo "::error::FAIL  $name  ($why)"; FAIL=1; reap; fi
}

echo "=== hosts: $HOSTS  (np=$NP, victim=rank $VICTIM)  tt-run=$TTRUN"

# 1. clean DistributedContext job across hosts: init context, open the mock galaxy, a collective, close. exit 0.
run_probe PROBE_SCENARIO=clean; check "context_multihost_clean" 0 $?

# 2. a peer dies inside a collective on a remote host (P20): the errhandler turns the survivors' barrier into a
#    returned error, the context calls MPI_Abort, the whole job ends non-zero and leaves nothing on any host.
run_probe PROBE_COLL=barrier_dead_peer PROBE_VICTIM="$VICTIM"; check "context_multihost_remote_peer_death" 1 $?

# 3. a peer hangs inside a collective on a remote host (P22): the survivors' all_gather gives up at
#    TT_METAL_OPERATION_TIMEOUT_SECONDS (15 s), long before the victim's 30 s sleep, and tears the job down. The
#    job must end non-zero, leave nothing, and finish well under the 180 s launcher backstop — proving the
#    CONTEXT timeout bounded it, not the wall clock.
t0=$SECONDS
run_probe PROBE_COLL=allgather_skip PROBE_VICTIM="$VICTIM" TT_METAL_OPERATION_TIMEOUT_SECONDS=15; rc=$?
check "context_multihost_remote_hang_timeout" 1 "$rc" 60 $((SECONDS - t0))

reap
[ "$FAIL" -eq 0 ] && echo "=== teardown context multihost: all cases passed" || echo "=== teardown context multihost: FAILURES above"
exit "$FAIL"
