#!/usr/bin/env bash
# Teardown sanity, multi-host (lane L1c of the plan, tt-metal #56902).
#
# One question: when a fault lands on a REMOTE host, does tt-run / PRRTE tear the whole job down correctly across
# every host? The single-host sanity lane cannot see this, so these cases need more than one host and run on SC4.
# Each case launches the raw C++ probe (sanity_mpi_probe.cpp) across the hosts, one rank per host, on the production
# transport line (--mca btl self,tcp), then asserts the teardown shape: the right exit code AND no rank left alive
# on any host.
#
# Run it yourself:
#   ./run_teardown_multihost_tests.sh --hosts host1,host2[,host3,...]
# Needs: at least two hosts, passwordless ssh between them, and mpirun + mpicxx on PATH. The probe source is found
# under $TT_METAL_HOME (falls back to the current directory).
set -uo pipefail

# --------------------------------------------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------------------------------------------
hosts=""
while [ $# -gt 0 ]; do
  case "$1" in
    --hosts) hosts="$2"; shift 2;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown argument: $1 (try --help)"; exit 2;;
  esac
done

host_names() { tr ',' ' ' <<<"$hosts"; }  # "h1,h2" -> "h1 h2"
num_hosts=$(host_names | wc -w)
[ "${num_hosts:-0}" -ge 2 ] || { echo "::error::--hosts needs at least two hosts (got: '${hosts:-}')"; exit 2; }

last_host=$(host_names | awk '{print $NF}')  # the remote host we inject faults on
victim_rank=$((num_hosts - 1))               # one rank per host, so the last rank runs on last_host

# --------------------------------------------------------------------------------------------------------------
# build the probe and put it on every host (exabox shares home over NFS, so the copy is just a fallback)
# --------------------------------------------------------------------------------------------------------------
probe="$(mktemp -d)/teardown_mpi_probe"
probe_src="${TT_METAL_HOME:-$(pwd)}/tests/tt_metal/distributed/multiprocess/teardown_sanity/sanity_mpi_probe.cpp"
mpicxx -O1 -o "$probe" "$probe_src" || { echo "::error::the probe did not compile ($probe_src)"; exit 1; }
for host in $(host_names); do scp -q "$probe" "$host:$probe" 2>/dev/null || true; done

launcher="$(command -v 'mpirun-ulfm' || command -v 'mpirun')"

# the production launch line: one rank per host, tcp transport only (no shared-memory layer), tagged output
launch=(--host "$hosts" -np "$num_hosts" --map-by node --oversubscribe --bind-to none --mca btl self,tcp --tag-output)

# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------
# how many of this job's ranks are still alive on any host (should be 0 once the job has been torn down)
ranks_still_alive() {
  local total=0 host
  for host in $(host_names); do
    total=$((total + $(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "pgrep -cf teardown_mpi_probe" 2>/dev/null || echo 0)))
  done
  echo "$total"
}

kill_stragglers() {
  local host
  for host in $(host_names); do ssh -o BatchMode=yes "$host" "pkill -9 -f teardown_mpi_probe" 2>/dev/null || true; done
}

failures=0
# assert a teardown shape. usage: expect <clean|torn_down> <name> <exit-code>
#   clean      -> the job must exit 0
#   torn_down  -> the job must exit non-zero (a fault was injected)
# in both cases nothing of this job may be left alive on any host.
expect() {
  local shape="$1" name="$2" rc="$3"
  sleep 2  # let PRRTE finish tearing the job down before we count
  local alive problem=""
  alive=$(ranks_still_alive)
  if [ "$shape" = "clean" ] && [ "$rc" -ne 0 ]; then problem="expected exit 0, got $rc"; fi
  if [ "$shape" = "torn_down" ] && [ "$rc" -eq 0 ]; then problem="expected a non-zero exit, got 0"; fi
  if [ "$alive" -ne 0 ]; then problem="${problem:+$problem; }$alive rank(s) still alive on the hosts"; fi
  if [ -z "$problem" ]; then
    echo "  PASS  $name  (exit $rc, nothing left)"
  else
    echo "::error::  FAIL  $name  ($problem)"; failures=$((failures + 1)); kill_stragglers
  fi
}

echo "=== hosts: $hosts  (one rank per host, victim = rank $victim_rank on $last_host)  launcher: $launcher"

# --------------------------------------------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------------------------------------------

echo "--- 1. a clean job across hosts: it should just finish, exit 0, nothing left (baseline)"
PROBE_SCENARIO=clean PROBE_NP="$num_hosts" \
  timeout 120 "$launcher" "${launch[@]}" "$probe"
expect clean "multihost_clean" $?

echo "--- 2. a rank dies on the remote host (P20): PRRTE must end the whole job non-zero, nothing left on any host"
PROBE_SCENARIO=victim_sigkill PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts" \
  timeout 120 "$launcher" "${launch[@]}" "$probe"
expect torn_down "multihost_remote_rank_death" $?

echo "--- 3. a rank hangs on the remote host (P22): the PRRTE --timeout wall clock ends it (241), nothing left"
PROBE_SCENARIO=victim_hang PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts" \
  timeout 120 "$launcher" --timeout 30 --report-state-on-timeout "${launch[@]}" "$probe"
expect torn_down "multihost_remote_hang_timeout" $?

echo "--- 4. the daemon on the remote host is killed (P19): with the plain launcher PRRTE aborts the job instead of"
echo "       exiting 0 and orphaning the launch-host rank. We kill the remote prted while the ranks wait; the job"
echo "       must end non-zero with nothing left. (--timeout is only a backstop so the case cannot hang CI.)"
( sleep 10; ssh -o BatchMode=yes "$last_host" "pkill -9 -x prted || pkill -9 -f '[p]rted'" 2>/dev/null || true ) &
PROBE_SCENARIO=victim_hang PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts" \
  timeout 120 "$launcher" --timeout 60 --report-state-on-timeout "${launch[@]}" "$probe"
expect torn_down "multihost_remote_daemon_death" $?

# --------------------------------------------------------------------------------------------------------------
kill_stragglers
if [ "$failures" -eq 0 ]; then
  echo "=== teardown multihost: all cases passed"
else
  echo "=== teardown multihost: $failures case(s) FAILED (see ::error:: lines above)"
fi
exit "$failures"
