#!/usr/bin/env bash
# Teardown sanity, multi-host (lane L1c of the plan, tt-metal #56902): does tt-run / PRRTE tear a job down
# correctly when a fault lands on a REMOTE host? The single-host sanity (fabric CPU-only lane) cannot see these;
# they need more than one host, so they run here on SC4 (fabric-multihost-exabox, bh_sc4, 4 hosts).
#
# Usage: run_teardown_multihost_tests.sh --hosts h1,h2,h3,h4
# The C probe (sanity_mpi_probe.c) is compiled with the shipped mpicc and launched across the hosts with the
# production transport line (--mca btl self,tcp). Each case asserts a known teardown shape: the right exit code
# and no rank or daemon left on any host.
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
SRC="$ROOT/tests/tt_metal/distributed/multiprocess/teardown_sanity/sanity_mpi_probe.c"
MPIRUN="$(command -v mpirun-ulfm || command -v mpirun)"
MPICC="$(command -v mpicc)"
PROBE="$(mktemp -d)/teardown_mpi_probe"
"$MPICC" -O1 -o "$PROBE" "$SRC" || { echo "::error::could not compile the probe"; exit 1; }
# the probe binary must exist at the same path on every host (shared home on exabox); copy as a fallback.
for h in $(tr ',' ' ' <<<"$HOSTS"); do scp -q "$PROBE" "$h:$PROBE" 2>/dev/null || true; done

NP=$(($(tr ',' '\n' <<<"$HOSTS" | grep -c .)))
BASE=(--host "$HOSTS" -np "$NP" --map-by node --oversubscribe --bind-to none --mca btl self,tcp --tag-output)
FAIL=0

# how many probe processes are still alive on any host (should be 0 after the job)
leftovers() {
  local n=0
  for h in $(tr ',' ' ' <<<"$HOSTS"); do
    n=$((n + $(ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "pgrep -cf teardown_mpi_probe" 2>/dev/null || echo 0)))
  done
  echo "$n"
}
reap() { for h in $(tr ',' ' ' <<<"$HOSTS"); do ssh -o BatchMode=yes "$h" "pkill -9 -f teardown_mpi_probe" 2>/dev/null || true; done; }

check() { # name  expect_nonzero(0|1)  actual_rc
  local name="$1" want_nonzero="$2" rc="$3"
  sleep 2; local left; left=$(leftovers)
  local ok=1
  if [ "$want_nonzero" = "1" ] && [ "$rc" -eq 0 ]; then ok=0; fi
  if [ "$want_nonzero" = "0" ] && [ "$rc" -ne 0 ]; then ok=0; fi
  if [ "$left" -ne 0 ]; then ok=0; fi
  if [ "$ok" = "1" ]; then echo "PASS  $name  (exit $rc, nothing left)";
  else echo "::error::FAIL  $name  (exit $rc, $left process(es) left on the hosts)"; FAIL=1; reap; fi
}

echo "=== hosts: $HOSTS  (np=$NP)  mpirun=$MPIRUN"

# 1. clean multi-host job: tt-run/mpirun ends 0, nothing left (baseline)
PROBE_SCENARIO=clean PROBE_NP="$NP" timeout 120 "$MPIRUN" "${BASE[@]}" "$PROBE"; check "multihost_clean" 0 $?

# 2. a rank dies on a remote host (P20): the job must end with a non-zero code and leave nothing on any host
PROBE_SCENARIO=victim_sigkill PROBE_VICTIM=$((NP-1)) PROBE_NP="$NP" timeout 120 "$MPIRUN" "${BASE[@]}" "$PROBE"; check "multihost_remote_rank_death" 1 $?

# 3. a rank hangs on a remote host, bounded by the PRRTE wall clock (P22 backstop): exit 241, nothing left
PROBE_SCENARIO=victim_hang PROBE_VICTIM=$((NP-1)) PROBE_NP="$NP" timeout 120 "$MPIRUN" --timeout 30 --report-state-on-timeout "${BASE[@]}" "$PROBE"; check "multihost_remote_hang_timeout" 1 $?

reap
[ "$FAIL" -eq 0 ] && echo "=== teardown multihost: all cases passed" || echo "=== teardown multihost: FAILURES above"
exit "$FAIL"
