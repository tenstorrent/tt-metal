#!/bin/bash
# Shared pieces for the prefill-matrix scripts (sourced, not executed).
# The scripts run with `set -uo pipefail` and check return codes by hand (no `set -e`): a failed row must not abort
# the matrix, and a failed shutdown must still fall through to the scoped kill.

# Pipeline constants the runner's binding, the producer and the shutdown helper must agree on.
MATRIX_CHUNK=5120           # PREFILL_CHUNK_SIZE
MATRIX_MAX_NEW=51200        # KV capacity per slot = cached + MATRIX_MAX_NEW (largest default new-token value)
MATRIX_FALLBACK_CAP=56320   # capacity assumed when a runner log carries no capacity= token (0 + MATRIX_MAX_NEW + one chunk)

# A Slurm step on a host that already runs pipeline ranks cannot even fork at the exabox default soft limits
# (512 processes/threads per user, 4096 files, 24 CPU-hours), so every step raises them to the hard limits first.
# Literal values on purpose: `ulimit -u $(ulimit -Hu)` needs a fork, which is exactly what fails. 2318132 is the
# exabox hard limit; the chain falls back for hosts with a lower one (a failed ulimit leaves the soft limit unchanged).
MATRIX_ULIMITS='{ ulimit -u 2318132 || ulimit -u 1000000 || ulimit -u 65536; } 2>/dev/null; ulimit -n 131072 2>/dev/null; ulimit -t unlimited 2>/dev/null; ulimit -s unlimited 2>/dev/null'

# Run one command string on one host of the allocation, ulimits raised, srun chatter filtered; returns srun's rc.
#   matrix_srun <job> <host> <command string for bash -c>
matrix_srun() {
  local job=$1 host=$2 cmd=$3
  srun --jobid="$job" --overlap -N1 -n1 -w "$host" bash -c "$MATRIX_ULIMITS; $cmd" 2>&1 | { grep -v '^srun: ' || true; }
}

# Shell-quote a KEY=VALUE assignment for use inside a `bash -c "..."` string (paths with spaces / `$` survive).
matrix_q() { printf '%q' "$1"; }

# Environment of the producer / shutdown helper (host-side H2D clients): the mesh shape, layer count, chunk size and
# KV capacity must agree with the runner's binding (SP=2 x TP=4 per [2,4] stage, 60 layers, 5120-token chunks);
# PREFILL_NUM_USERS is only read by prefill_producer's own main() and is irrelevant here.
#   matrix_producer_env <max_seq_len>  -> "KEY=VAL KEY=VAL ..." (each shell-quoted) for `env`
matrix_producer_env() {
  local e="PREFILL_MODEL=minimax_m3 PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_SP=2 PREFILL_TP=4 PREFILL_NUM_LAYERS=60"
  e="$e PREFILL_CHUNK_SIZE=$MATRIX_CHUNK PREFILL_NUM_USERS=1 PREFILL_MAX_SEQ_LEN=$1"
  e="$e $(matrix_q "TT_METAL_HOME=$TT_METAL_HOME") $(matrix_q "PYTHONPATH=$TT_METAL_HOME")"
  local v
  for v in HF_MODEL TT_CACHE_PATH PREFILL_TRACE_DIR; do
    [ -n "${!v:-}" ] && e="$e $(matrix_q "$v=${!v}")"
  done
  echo "$e"
}

# Kill ONLY the pipeline processes that belong to one runner launch, identified by the manifest path that launch
# wrote (every rank carries PREFILL_MANIFEST=<that path> in its environment, mpirun/prterun carries it on its command
# line as `-x PREFILL_MANIFEST=<path>`, and ttrun.py names the binding). Never a bare pkill of every
# prefill_runner/ttrun the account owns. Remote `prted` daemons are not matched: they exit when their prterun dies.
#   matrix_kill_runner <job> <manifest> <binding> <host>...
matrix_kill_runner() {
  local job=$1 manifest=$2 binding=$3 h; shift 3
  if [ -z "$manifest" ] || [ -z "$binding" ]; then
    echo "[kill] refusing to kill with an empty manifest/binding (would match every process)"; return 1
  fi
  for h in "$@"; do
    # Bracketed patterns so this step's own `bash -c` (whose command line contains them) does not match itself;
    # `$$` is skipped as well. Paths are matched as fixed strings.
    matrix_srun "$job" "$h" "n=0
      for p in \$(pgrep -u \"\$(id -un)\" -f 'prefill_runne[r]|ttrun\.p[y]|prteru[n]|prte[d]'); do
        [ \"\$p\" = \"\$\$\" ] && continue
        if tr '\\0' '\\n' < /proc/\$p/environ 2>/dev/null | grep -qxF $(matrix_q "PREFILL_MANIFEST=$manifest") \\
           || tr '\\0' ' ' < /proc/\$p/cmdline 2>/dev/null | grep -qF -e $(matrix_q "PREFILL_MANIFEST=$manifest") -e $(matrix_q "$binding"); then
          kill -9 \$p 2>/dev/null && n=\$((n+1))
        fi
      done
      echo \"[kill] \$(hostname -s): killed \$n process(es) of this runner\""
  done
}

# Shut down the runner recorded in <work>/last_runner_log if it is still live: send the SHUTDOWN sentinel from
# rank 0's host, wait up to 240 s for the launcher step to exit, else kill that runner's processes on every host.
# Returns 0 when there was nothing live or it exited on the sentinel, 1 when the kill path ran (reset advised).
#   matrix_shutdown_runner <job> <work> <rank0 host> <host>...
matrix_shutdown_runner() {
  local job=$1 work=$2 r0=$3 prev cap man bind; shift 3
  prev=$(cat "$work/last_runner_log" 2>/dev/null)
  [ -n "$prev" ] && [ -f "$prev" ] || return 0
  grep -q '^EXIT=' "$prev" && return 0
  cap=$(grep -o 'capacity=[0-9]*' "$prev" | head -1 | cut -d= -f2-)
  man=$(grep -o 'manifest=[^ ]*' "$prev" | head -1 | cut -d= -f2-)
  bind=$(grep -o 'binding=[^ ]*' "$prev" | head -1 | cut -d= -f2-)
  echo "[shutdown] $(date +%T) live runner ($prev): sending SHUTDOWN"
  matrix_srun "$job" "$r0" "cd $(matrix_q "$TT_METAL_HOME") && source python_env/bin/activate && env $(matrix_producer_env "${cap:-$MATRIX_FALLBACK_CAP}") timeout 300 python3 $(matrix_q "$PKG_DIR/matrix_shutdown.py")"
  for _ in $(seq 1 120); do grep -q '^EXIT=' "$prev" && break; sleep 2; done
  if grep -q '^EXIT=' "$prev"; then echo "[shutdown] runner exited: $(grep '^EXIT=' "$prev")"; return 0; fi
  echo "[shutdown] runner did not exit in 240 s; killing that runner's processes (manifest $man) on every host"
  matrix_kill_runner "$job" "$man" "$bind" "$r0" "$@"
  sleep 5
  return 1
}
