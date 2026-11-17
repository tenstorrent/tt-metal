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

# ssh/scp reach the peers as REMOTE_USER when it is set. The multihost CI (setup-multihost-job) provisions
# passwordless ssh for the 'user' account and syncs the workspace with `ssh -l user`, so that lane sets
# TEARDOWN_REMOTE_USER=user to reach the same account. Empty (the default) keeps a bare host, so a manual run
# uses your own account and keys.
REMOTE_USER="${TEARDOWN_REMOTE_USER:-}"
rtarget() { if [ -n "$REMOTE_USER" ]; then printf '%s@%s' "$REMOTE_USER" "$1"; else printf '%s' "$1"; fi; }

# --------------------------------------------------------------------------------------------------------------
# build the probe and stage it on every host. The compile dir is a local mktemp under /tmp, which SC4 does NOT
# share across hosts (only the workspace is synced), so the copy is required, not a fallback. scp will not create
# the destination directory, so make it first on each host, and verify the probe actually landed and is runnable
# there — otherwise a worker that never got the probe would fail the launch, not the staging.
# --------------------------------------------------------------------------------------------------------------
probe="$(mktemp -d)/teardown_mpi_probe"
probe_src="${TT_METAL_HOME:-$(pwd)}/tests/tt_metal/distributed/multiprocess/teardown_sanity/sanity_mpi_probe.cpp"
mpicxx -O1 -o "$probe" "$probe_src" || { echo "::error::the probe did not compile ($probe_src)"; exit 1; }
for host in $(host_names); do
  ssh -o BatchMode=yes "$(rtarget "$host")" "mkdir -p '$(dirname "$probe")'" \
    || { echo "::error::could not create the probe directory on $host"; exit 1; }
  scp -q "$probe" "$(rtarget "$host"):$probe" 2>/dev/null || true  # a no-op on the compile host (same file); verified next
  ssh -o BatchMode=yes "$(rtarget "$host")" "test -x '$probe'" \
    || { echo "::error::the probe is not staged/executable on $host ($probe)"; exit 1; }
done

# The pgrep/pkill pattern that matches ONLY this run's ranks: the probe path is a unique mktemp path, so a
# concurrent job's probe (under a different /tmp dir) is never counted or killed. The trailing character is
# bracketed (teardown_mpi_prob[e]) so the regex does not match the pgrep/pkill command's own wrapper shell, which
# carries the pattern in its cmdline — a bare pattern self-matches and would count/kill itself.
probe_pat="${probe%e}[e]"

# Select the SAME launcher tt-run uses: the PLAIN mpirun of the ULFM-capable install, never the mpirun-ulfm
# wrapper itself. mpirun-ulfm can inject --with-ft ulfm, which makes PRRTE treat the job as recoverable — a rank
# killed by a signal no longer aborts the job — so with that wrapper cases 2-4 would silently exercise the wrong
# mode. This mirrors get_mpi_launcher()/_plain_mpirun_beside() in ttnn/ttnn/distributed/ttrun.py: resolve
# mpirun-ulfm, take the mpirun beside it, and fall back to it (then to a bare mpirun) only when there is no sibling.
select_launcher() {
  local ulfm plain
  ulfm="$(command -v mpirun-ulfm 2>/dev/null || true)"
  if [ -n "$ulfm" ]; then
    plain="$(dirname "$(readlink -f "$ulfm")")/mpirun"
    if [ -x "$plain" ]; then printf '%s\n' "$plain"; return 0; fi
    printf '%s\n' "$ulfm"; return 0  # no sibling plain mpirun; use it directly (it may enable ULFM)
  fi
  command -v mpirun 2>/dev/null || return 1
}
launcher="$(select_launcher)" || { echo "::error::no mpirun-ulfm / mpirun on PATH"; exit 1; }

# the production launch line: one rank per host, tcp transport only (no shared-memory layer), tagged output
launch=(--host "$hosts" -np "$num_hosts" --map-by node --oversubscribe --bind-to none --mca btl self,tcp --tag-output)

# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------
# How many of this job's ranks are still alive on any host (should be 0 once the job has been torn down). Prints
# the count on success, or "ERROR" if any host could not be censused — `pgrep -c` prints 0 but EXITS 1 when there
# is no match, so the remote command ends in `|| true` to keep that a clean 0 (a bare `|| echo 0` would append a
# second line and break the arithmetic); ssh transport failure (e.g. rc 255) or a non-numeric count is treated as
# a census failure, not as zero survivors, so "nothing left" is never asserted from a census we could not take.
ranks_still_alive() {
  local total=0 host n
  for host in $(host_names); do
    n=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$(rtarget "$host")" "pgrep -cf '$probe_pat' || true" 2>/dev/null)
    if [ $? -ne 0 ] || ! [[ "$n" =~ ^[0-9]+$ ]]; then echo "ERROR"; return 1; fi
    total=$((total + n))
  done
  echo "$total"
}

kill_stragglers() {
  local host
  for host in $(host_names); do ssh -o BatchMode=yes "$(rtarget "$host")" "pkill -9 -f '$probe_pat'" 2>/dev/null || true; done
}

# Kill ONLY this job's prted on $1. The rank on that host runs under this job's prted, so we find this run's rank
# by its unique probe path (pgrep -f "$probe_pat") and walk up the process ancestry (ps -o ppid=) to the first
# prted — that is our daemon, killed by its one PID. This scopes the fault to this job (a concurrent job's prted
# on a shared host is never touched) and fails loudly if there is no rank or no prted ancestor to kill, so a
# no-op injection cannot masquerade as a teardown. $probe_pat is expanded here and handed to the remote shell as
# PROBE_PAT, which keeps the remote script free of nested quoting.
kill_this_jobs_prted() {
  local host="$1"
  ssh -o BatchMode=yes "$(rtarget "$host")" "PROBE_PAT='$probe_pat' bash -s" <<'REMOTE'
    rank=$(pgrep -f "$PROBE_PAT" | head -1)
    [ -n "$rank" ] || { echo "no rank matching the probe on this host"; exit 2; }
    pid=$rank
    while [ "${pid:-1}" -gt 1 ]; do
      if [ "$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')" = prted ]; then
        kill -9 "$pid" && { echo "killed this job's prted (pid $pid, ancestor of rank $rank)"; exit 0; }
        echo "found prted $pid but could not kill it"; exit 4
      fi
      pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    done
    echo "rank $rank has no prted ancestor"; exit 3
REMOTE
}

failures=0
# assert a teardown shape. usage: expect <clean|torn_down> <name> <exit-code> [exact-code]
#   clean      -> the job must exit 0
#   torn_down  -> the job must exit non-zero, but NOT 124. 124 is this harness's own `timeout 120` killing a
#                 launcher that hung: the launcher never tore the job down itself, which is the failure these
#                 cases exist to catch, not a teardown. With [exact-code] given the job must exit exactly that
#                 code (241, the PRRTE --timeout wall clock, for the hang case).
# in both cases nothing of this job may be left alive on any host.
expect() {
  local shape="$1" name="$2" rc="$3" exact="${4:-}"
  sleep 2  # let PRRTE finish tearing the job down before we count
  local alive problem=""
  alive=$(ranks_still_alive)
  if [ "$shape" = "clean" ] && [ "$rc" -ne 0 ]; then problem="expected exit 0, got $rc"; fi
  if [ "$shape" = "torn_down" ]; then
    if [ "$rc" -eq 0 ]; then
      problem="expected a non-zero exit, got 0"
    elif [ "$rc" -eq 124 ]; then
      problem="the launcher hung and this harness's timeout 120 killed it (rc 124); it did not tear the job down"
    elif [ -n "$exact" ] && [ "$rc" -ne "$exact" ]; then
      problem="expected exit $exact, got $rc"
    fi
  fi
  if ! [[ "$alive" =~ ^[0-9]+$ ]]; then
    problem="${problem:+$problem; }could not census the hosts (ssh/pgrep failed); cannot confirm the job was torn down"
  elif [ "$alive" -ne 0 ]; then
    problem="${problem:+$problem; }$alive rank(s) still alive on the hosts"
  fi
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
expect torn_down "multihost_remote_hang_timeout" $? 241  # PRRTE --timeout 30 must end it with 241, not a hang

echo "--- 4. the daemon on the remote host is killed (P19): with the plain launcher PRRTE must abort the job itself"
echo "       instead of exiting 0 and orphaning the launch-host rank. We wait for this job's rank to come up on the"
echo "       remote host, kill ONLY this job's prted there, and require PRRTE's own abort (a fast non-zero). There"
echo "       is no --timeout backstop on purpose: if PRRTE does not abort, the rank hangs, the outer 'timeout 120'"
echo "       ends it with 124, and expect rejects that as the failure — so the case tests the abort, not the clock."
( for _ in $(seq 1 30); do
    ssh -o BatchMode=yes "$(rtarget "$last_host")" "pgrep -f '$probe_pat' >/dev/null 2>&1" && break; sleep 1
  done
  kill_this_jobs_prted "$last_host" || echo "::error::daemon-death fault was not injected on $last_host" ) &
inject_pid=$!
PROBE_SCENARIO=victim_hang PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts" \
  timeout 120 "$launcher" "${launch[@]}" "$probe"
rc=$?
wait "$inject_pid"  # require the injector to finish and report what it did before we judge the teardown
expect torn_down "multihost_remote_daemon_death" $rc

# --------------------------------------------------------------------------------------------------------------
kill_stragglers
if [ "$failures" -eq 0 ]; then
  echo "=== teardown multihost: all cases passed"
else
  echo "=== teardown multihost: $failures case(s) FAILED (see ::error:: lines above)"
fi
exit "$failures"
