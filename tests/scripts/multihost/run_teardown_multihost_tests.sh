#!/usr/bin/env bash
# Teardown sanity, multi-host (lane L1c of the plan, tt-metal #56902).
#
# One question: when a fault lands on a REMOTE host, does tt-run tear the whole job down correctly across every
# host? The single-host sanity lane cannot see this, so these cases need more than one host and run on SC4. Each case
# launches the C++ probe (sanity_mpi_probe.cpp) through tt-run, as every real job is launched, one rank per host, then
# asserts the teardown shape: the right exit code AND no rank left alive on any host.
#
# Run it yourself:
#   ./run_teardown_multihost_tests.sh --hosts host1,host2[,host3,...]
# Needs: at least two hosts, passwordless ssh between them (or TEARDOWN_REMOTE=srun inside a Slurm allocation that
# holds them, where the compute nodes have no ssh keys), tt-run (TEARDOWN_TTRUN, else tt-run on PATH, else the
# ttnn module), and mpicxx on the hosts when the build has no probe. The probe source is found under $TT_METAL_HOME
# (falls back to the current directory). Run it from the repository root.
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
# The CI runner holds no host keys for the peers, so a strict check fails before anything runs; the same options as
# setup-multihost-job's own ssh to them.
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
# A command on a host: ssh, or srun --overlap inside a Slurm allocation (TEARDOWN_REMOTE=srun). remote_sh_stdin
# runs a script read from stdin there.
remote_sh() {
  local host="$1"; shift
  if [ "${TEARDOWN_REMOTE:-ssh}" = srun ]; then srun --overlap -N1 -n1 -w "$host" bash -c "$*"
  else ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "$(rtarget "$host")" "$*"; fi
}
remote_sh_stdin() {
  local host="$1"; shift
  if [ "${TEARDOWN_REMOTE:-ssh}" = srun ]; then srun --overlap -N1 -n1 -w "$host" env "$@" bash -s
  else ssh "${SSH_OPTS[@]}" "$(rtarget "$host")" "$* bash -s"; fi
}

# --------------------------------------------------------------------------------------------------------------
# build the probe and stage it on every host. The compile dir is a local mktemp under /tmp, which SC4 does NOT
# share across hosts (only the workspace is synced), so the copy is required, not a fallback. scp will not create
# the destination directory, so make it first on each host, and verify the probe actually landed and is runnable
# there — otherwise a worker that never got the probe would fail the launch, not the staging.
# --------------------------------------------------------------------------------------------------------------
# The probe the build made is preferred: it is built for the hosts, as every other rank of this lane is. A probe
# compiled here may not start there (the CI runner's glibc is newer than the hosts': "GLIBC_2.38 not found"), so
# each host also checks that its copy loads, and compiles its own from the source when it does not.
probe="$(mktemp -d)/teardown_mpi_probe"
probe_src="${TT_METAL_HOME:-$(pwd)}/tests/tt_metal/distributed/multiprocess/teardown_sanity/sanity_mpi_probe.cpp"
probe_built="${TT_METAL_HOME:-$(pwd)}/build/test/tt_metal/distributed/teardown_mpi_probe"
if [ -x "$probe_built" ]; then
  cp "$probe_built" "$probe"
else
  mpicxx -O1 -o "$probe" "$probe_src" || { echo "::error::the probe did not compile ($probe_src)"; exit 1; }
fi
probe_loads="ldd '$probe' 2>&1 | grep -q 'not found' && exit 1 || exit 0"
for host in $(host_names); do
  remote_sh "$host" "mkdir -p '$(dirname "$probe")'" \
    || { echo "::error::could not create the probe directory on $host"; exit 1; }
  for f in "$probe" "$probe_src"; do  # a no-op on the compile host (or a shared TMPDIR); verified next
    if [ "${TEARDOWN_REMOTE:-ssh}" = srun ]; then
      srun --overlap -N1 -n1 -w "$host" dd status=none of="$(dirname "$probe")/$(basename "$f")" <"$f" 2>/dev/null || true
      srun --overlap -N1 -n1 -w "$host" chmod +x "$(dirname "$probe")/$(basename "$f")" 2>/dev/null || true
    else
      scp -q "${SSH_OPTS[@]}" "$f" "$(rtarget "$host"):$(dirname "$probe")/" 2>/dev/null || true
    fi
  done
  if ! remote_sh "$host" "$probe_loads"; then
    remote_sh "$host" "mpicxx -O1 -o '$probe' '$(dirname "$probe")/$(basename "$probe_src")'" >/dev/null 2>&1 || true
  fi
  remote_sh "$host" "test -x '$probe' && { $probe_loads; }" \
    || { echo "::error::the probe is not staged or does not load on $host ($probe): ldd says"; remote_sh "$host" "ldd '$probe'" 2>&1 | grep 'not found'; exit 1; }
done

# The pgrep/pkill pattern that matches ONLY this run's ranks: the probe path is a unique mktemp path, so a
# concurrent job's probe (under a different /tmp dir) is never counted or killed. The trailing character is
# bracketed (teardown_mpi_prob[e]) so the regex does not match the pgrep/pkill command's own wrapper shell, which
# carries the pattern in its cmdline — a bare pattern self-matches and would count/kill itself.
probe_pat="${probe%e}[e]"

# tt-run: an override (the CI lane points it at the PR's ttrun.py), then tt-run on PATH, then the module.
if [ -n "${TEARDOWN_TTRUN:-}" ]; then read -r -a ttrun_cmd <<<"$TEARDOWN_TTRUN"
elif command -v tt-run >/dev/null 2>&1; then ttrun_cmd=(tt-run)
else ttrun_cmd=(python3 -m ttnn.distributed.ttrun); fi
"${ttrun_cmd[@]}" --help >/dev/null 2>&1 || { echo "::error::tt-run is not runnable here: ${ttrun_cmd[*]}"; exit 1; }
# A stop that has to be forced is forced within the harness's own `timeout 120`.
export TT_RUN_STOP_GRACE_S="${TT_RUN_STOP_GRACE_S:-30}"

# One rank per host. tt-run starts one app context per rank, so -np or --map-by node in --mpi-args would multiply
# the ranks or pile them on the first host; a rankfile places rank r on host r. It sits under the working directory
# with a plain relative name, which is how tt-run hands it to PRRTE (--map-by rankfile:file=<path>; a ../ path is
# rejected). -x forwards the fault settings to the ranks on the remote hosts.
binding="$(mktemp --suffix=.yaml)"
{ echo "rank_bindings:"
  for r in $(seq 0 $((num_hosts - 1))); do printf '  - rank: %d\n    mesh_id: 0\n' "$r"; done
  echo 'mesh_graph_desc_path: "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto"'
} >"$binding"
mkdir -p generated/teardown_sanity
rankfile="generated/teardown_sanity/rankfile_$$"
r=0; for host in $(host_names); do printf 'rank %d=%s slot=0\n' "$r" "$host"; r=$((r + 1)); done >"$rankfile"
trap 'rm -f "$binding" "$rankfile"' EXIT

# Launch the probe through tt-run with the PROBE_* settings of the caller, under the harness's `timeout 120` (124 if
# the job never ends). usage: ttrun_job [more --mpi-args]. With bound=() the job runs unbounded, for a case that signals
# tt-run's own pid (timeout passes on only one signal).
bound=(timeout 120)
ttrun_job() {
  local run=("${bound[@]}")
  [ -n "${exec_ttrun:-}" ] && run=(exec "${run[@]}")
  "${run[@]}" "${ttrun_cmd[@]}" --rank-binding "$binding" \
    --mpi-args "--rankfile $rankfile --oversubscribe --bind-to none --tag-output -x PROBE_SCENARIO -x PROBE_VICTIM -x PROBE_NP $*" \
    "$probe"
}

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
    n=$(remote_sh "$host" "pgrep -cf '$probe_pat' || true" 2>/dev/null)
    if [ $? -ne 0 ] || ! [[ "$n" =~ ^[0-9]+$ ]]; then echo "ERROR"; return 1; fi
    total=$((total + n))
  done
  echo "$total"
}

kill_stragglers() {
  local host
  for host in $(host_names); do remote_sh "$host" "pkill -9 -f '$probe_pat'" 2>/dev/null || true; done
}

# Kill ONLY this job's prted on $1. The rank on that host runs under this job's prted, so we find this run's rank
# by its unique probe path (pgrep -f "$probe_pat") and walk up the process ancestry (ps -o ppid=) to the first
# prted — that is our daemon, killed by its one PID. This scopes the fault to this job (a concurrent job's prted
# on a shared host is never touched) and fails loudly if there is no rank or no prted ancestor to kill, so a
# no-op injection cannot masquerade as a teardown. $probe_pat is expanded here and handed to the remote shell as
# PROBE_PAT, which keeps the remote script free of nested quoting.
kill_this_jobs_prted() {
  local host="$1"
  remote_sh_stdin "$host" "PROBE_PAT=$probe_pat" <<'REMOTE'
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
#                 code (137 for a rank killed by SIGKILL, 241 for the --timeout).
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

echo "=== hosts: $hosts  (one rank per host, victim = rank $victim_rank on $last_host)  tt-run: ${ttrun_cmd[*]}"

# --------------------------------------------------------------------------------------------------------------
# the cases, every one launched through tt-run
# --------------------------------------------------------------------------------------------------------------

echo "--- 1. a clean job across hosts: it should just finish, exit 0, nothing left (baseline)"
PROBE_SCENARIO=clean PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts" ttrun_job
expect clean "multihost_clean" $?

echo "--- 2. a rank dies on the remote host (P20): its tt-run-rank wrapper stops the job, tt-run returns the rank's"
echo "       137, nothing left on any host"
PROBE_SCENARIO=victim_sigkill PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts" ttrun_job
expect torn_down "multihost_remote_rank_death" $? 137

echo "--- 3. a rank hangs on the remote host (P22): at the --timeout tt-run stops the job (SIGTERM to every rank),"
echo "       241, nothing left"
PROBE_SCENARIO=victim_hang PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts" ttrun_job --timeout 30
expect torn_down "multihost_remote_hang_timeout" $? 241

echo "--- 4. the daemon on the remote host is killed (P19): the rank there must not outlive it. We wait for this"
echo "       job's rank to come up on the remote host and kill ONLY this job's prted there; the rank's tt-run-rank"
echo "       wrapper sees its daemon die and stops its rank, the job ends non-zero and nothing is left on any host."
echo "       There is no --timeout on purpose: if the teardown does not happen, the outer 'timeout 120' ends the job"
echo "       with 124, and expect rejects that as the failure."
( for _ in $(seq 1 30); do
    remote_sh "$last_host" "pgrep -f '$probe_pat' >/dev/null 2>&1" && break; sleep 1
  done
  kill_this_jobs_prted "$last_host" || echo "::error::daemon-death fault was not injected on $last_host" ) &
inject_pid=$!
PROBE_SCENARIO=victim_hang PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts" ttrun_job
rc=$?
wait "$inject_pid"  # require the injector to finish and report what it did before we judge the teardown
expect torn_down "multihost_remote_daemon_death" $rc

echo "--- 5. an interrupt at tt-run is forwarded to EVERY rank (P7), remote hosts included, as a catchable signal the"
echo "       rank handles — NOT a direct kill. On SIGINT tt-run sends SIGTERM to every rank through PRRTE job control;"
echo "       in all_hang each rank prints 'caught signal 15' from its handler and exits. We require that marker from"
echo "       all $num_hosts ranks (so a remote rank got it too) and nothing left. A killed rank never prints it."
sig_log="$(mktemp)"
bound=()
# exec in the background subshell, so $! is tt-run's own pid (a backgrounded function would be a subshell's)
( export PROBE_SCENARIO=all_hang PROBE_VICTIM="$victim_rank" PROBE_NP="$num_hosts"; exec_ttrun=1 ttrun_job ) >"$sig_log" 2>&1 &
ttrun_pid=$!
# all ranks up first (each rank counts twice under tt-run: its tt-run-rank wrapper carries the probe path too)
for _ in $(seq 1 60); do [ "$(ranks_still_alive)" -ge "$num_hosts" ] 2>/dev/null && break; sleep 1; done
kill -INT "$ttrun_pid" 2>/dev/null
for _ in $(seq 1 60); do kill -0 "$ttrun_pid" 2>/dev/null || break; sleep 1; done
kill -9 "$ttrun_pid" 2>/dev/null || true
wait "$ttrun_pid" 2>/dev/null; sig_rc=$?
caught=$(grep -c "caught signal 15" "$sig_log" 2>/dev/null)  # grep -c prints 0 itself (and exits 1) on no match
caught=${caught:-0}
sleep 2; sig_alive=$(ranks_still_alive)
grep "caught signal" "$sig_log" 2>/dev/null | sed 's/^/    /'
if [ "$caught" -eq "$num_hosts" ] && [ "$sig_alive" = 0 ] && [ "$sig_rc" -eq 130 ]; then
  echo "  PASS  multihost_interrupt_forwarded_to_ranks  ($caught/$num_hosts ranks caught SIGTERM, nothing left; exit $sig_rc)"
else
  echo "::error::  FAIL  multihost_interrupt_forwarded_to_ranks  (caught SIGTERM=$caught/$num_hosts, alive=$sig_alive, exit $sig_rc, expected 130 — a rank was killed without receiving SIGTERM, a leftover remains, or tt-run did not end the job)"
  echo "---- tt-run output ----"; tail -40 "$sig_log"
  failures=$((failures + 1)); kill_stragglers
fi
rm -f "$sig_log"

# --------------------------------------------------------------------------------------------------------------
kill_stragglers
if [ "$failures" -eq 0 ]; then
  echo "=== teardown multihost: all cases passed"
else
  echo "=== teardown multihost: $failures case(s) FAILED (see ::error:: lines above)"
fi
exit "$failures"
