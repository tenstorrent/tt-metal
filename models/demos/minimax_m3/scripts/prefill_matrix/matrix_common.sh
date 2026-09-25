#!/bin/bash
# Shared pieces for the prefill-matrix scripts (sourced, not executed).

# A Slurm step on a host that already runs pipeline ranks cannot even fork at the exabox default soft limits
# (512 processes/threads per user, 4096 files, 24 CPU-hours), so every step raises them to the hard limits first.
# Literal values on purpose: `ulimit -u $(ulimit -Hu)` needs a fork, which is exactly what fails.
MATRIX_ULIMITS='ulimit -u 2318132 2>/dev/null; ulimit -n 131072 2>/dev/null; ulimit -t unlimited 2>/dev/null; ulimit -s unlimited 2>/dev/null'
matrix_ulimits() { eval "$MATRIX_ULIMITS"; }

# The pipeline configuration the producer / shutdown helper must agree on with the runner's binding
# (SP=2 x TP=4 per [2,4] stage, 60 layers, 5120-token chunks). One place, used by matrix_row.sh and run_matrix.sh.
#   matrix_producer_env <max_seq_len>  -> "KEY=VAL KEY=VAL ..." for `env`
matrix_producer_env() {
  local e="PREFILL_MODEL=minimax_m3 PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_SP=2 PREFILL_TP=4 PREFILL_NUM_LAYERS=60"
  e="$e PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 PREFILL_MAX_SEQ_LEN=$1 TT_METAL_HOME=$TT_METAL_HOME PYTHONPATH=$TT_METAL_HOME"
  [ -n "${HF_MODEL:-}" ] && e="$e HF_MODEL=$HF_MODEL"
  [ -n "${TT_CACHE_PATH:-}" ] && e="$e TT_CACHE_PATH=$TT_CACHE_PATH"
  [ -n "${PREFILL_TRACE_DIR:-}" ] && e="$e PREFILL_TRACE_DIR=$PREFILL_TRACE_DIR"
  echo "$e"
}

# Kill ONLY the pipeline processes that belong to one runner launch, identified by the manifest path that launch
# wrote (every rank carries PREFILL_MANIFEST=<that path> in its environment; the launcher's command line names
# the binding). Never a bare pkill of every prefill_runner/ttrun the account owns.
#   matrix_kill_runner <job> <manifest> <binding> <host>...
matrix_kill_runner() {
  local job=$1 manifest=$2 binding=$3; shift 3
  for h in "$@"; do
    srun --jobid="$job" --overlap -N1 -n1 -w "$h" bash -c "$MATRIX_ULIMITS
      n=0
      for p in \$(pgrep -u \$USER -f 'prefill_runner|ttrun.py|prterun|prted'); do
        if tr '\\0' '\\n' < /proc/\$p/environ 2>/dev/null | grep -qx 'PREFILL_MANIFEST=$manifest' || tr '\\0' ' ' < /proc/\$p/cmdline 2>/dev/null | grep -q -- '$binding'; then
          kill -9 \$p 2>/dev/null && n=\$((n+1))
        fi
      done
      echo \"[kill] \$(hostname -s): killed \$n process(es) of this runner\"" 2>&1 | grep -v '^srun: '
  done
}
