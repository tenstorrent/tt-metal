#!/bin/bash
# Run the local passthrough-pipeline framework smoke test on the 16-rank
# single-pod QUAD_BH cluster.
#
# The local test (../test_passthrough_pipeline.py) is a thin wrapper around
# deepseek's create_passthrough_pipeline_configuration helper that passes
# payload=ACTIVATION_W_TOKEN_META — the upstream
# test_passthrough_pipeline_block currently fails the socket-FIFO-size
# handshake on main because PR #43389 changed EmbeddingStage's downstream
# socket size without updating the helper's default payload.
#
# This script calls bootstrap_pipeline_dir.sh (with BH Galaxy revision
# detection) to set up the rank-binding and rankfile, then tt-runs pytest
# against the local test.

case "${1:-}" in
  -h|--help)
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Invokes the local passthrough-pipeline smoke test
  ../test_passthrough_pipeline.py::test_passthrough_pipeline_block

on the 16-rank single-pod QUAD_BH cluster (slow dispatch).

The local test wraps deepseek's create_passthrough_pipeline_configuration
helper with the payload=ACTIVATION_W_TOKEN_META workaround for the
FIFO-size-mismatch bug introduced upstream by PR #43389.

Required environment:
  TT_METAL_HOME    Repo root.

  HOSTS            Space- or comma-separated 4-host list. NO DEFAULT —
                   set per-shell, e.g.
                     export HOSTS="hostA hostB hostC hostD"

Optional environment:
  PYTEST_TIMEOUT   Per-test timeout (seconds). Default: 900.
EOF
    exit 0
    ;;
  "") ;;
  *)
    echo "[error] unexpected argument: $1" >&2
    echo "Run with --help for usage." >&2
    exit 2
    ;;
esac

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

. "$SCRIPT_DIR/_hosts.sh"

: "${TT_METAL_HOME:?set TT_METAL_HOME first}"
PIPELINE_DIR="${SINGLE_POD_PIPELINE_DIR:-${TT_METAL_HOME}/generated/single_pod_pipeline_dir}"
EXPECTED_RB="$PIPELINE_DIR/blitz_decode_pipeline_rank_binding_single_pod_ci.yaml"
EXPECTED_RF="$PIPELINE_DIR/blitz_decode_pipeline_rank_file_single_pod_ci"
BOOTSTRAP="$SCRIPT_DIR/bootstrap_pipeline_dir.sh"

# Auto-bootstrap if missing/incomplete.
if [ ! -d "$PIPELINE_DIR" ] || [ ! -f "$EXPECTED_RB" ] || [ ! -f "$EXPECTED_RF" ]; then
    echo "[run] $PIPELINE_DIR missing or incomplete — auto-bootstrapping..." >&2
    SINGLE_POD_PIPELINE_DIR="$PIPELINE_DIR" "$BOOTSTRAP"
fi

# PIPELINE_DIR scaffolding must exist on every remote host (each rank cd's into it
# before launching pytest). Mirror the symlink scaffold the way _run_common.sh does.
ensure_pipeline_dir_on_host() {
  local h="$1"
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "[ -d $PIPELINE_DIR ]" 2>/dev/null; then
    return 0
  fi
  echo "[run] $h: $PIPELINE_DIR missing — recreating symlink scaffolding..."
  ssh -o BatchMode=yes "$h" "
    mkdir -p $PIPELINE_DIR
    cd $PIPELINE_DIR && mkdir -p .benchmarks generated/{fabric,inspector,test_reports,watcher}
    for sub in build models python_env runtime tests tt_metal ttnn; do
      [ -e \$sub ] || ln -s $TT_METAL_HOME/\$sub \$sub
    done
  "
}
for h in $HOSTS; do ensure_pipeline_dir_on_host "$h"; done

HOSTS_WITH_SLOTS="$(echo "$HOSTS" | tr ' ' '\n' | sed 's/$/:4/' | paste -sd,)"
# OpenMPI 5.0.7 rejects rankfile paths containing '-'; bootstrap_pipeline_dir.sh
# pre-copies the rankfile to /var/tmp/single_pod_rankfile (hyphen-free).
RANKFILE_FOR_MPI="${SINGLE_POD_RANKFILE_PATH:-/var/tmp/single_pod_rankfile}"
if [ ! -f "$RANKFILE_FOR_MPI" ]; then
    echo "[run] ERROR: mpirun-friendly rankfile not found at $RANKFILE_FOR_MPI" >&2
    exit 1
fi

PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-900}"
# Local fork of the upstream test_passthrough_pipeline_block, with the
# payload=ACTIVATION_W_TOKEN_META fix for the FIFO-mismatch bug introduced
# by PR #43389. See ../test_passthrough_pipeline.py for details.
TEST_PATH="$TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/deepseek_pipeline_tests/test_passthrough_pipeline.py::test_passthrough_pipeline_block"
LOG_FILE="/tmp/deepseek_passthrough_$(date +%Y%m%d_%H%M%S).log"

cat <<EOF
[run] ─────────── deepseek test_passthrough_pipeline_block ───────────
[run] target:    $TEST_PATH
[run] hosts:     $HOSTS
[run] dispatch:  slow (TT_METAL_SLOW_DISPATCH_MODE=1)
[run] LOG_FILE=$LOG_FILE
[run] running... (multiplexed per-rank output; ~5-10 min)
[run] ─────────────────────────────────────────────────────────────────
EOF

export PATH=/opt/openmpi-v5.0.7-ulfm/bin:$PATH
ulimit -n 65536
cd "$PIPELINE_DIR"

tt-run --skip-executable-check \
  --rank-binding "$EXPECTED_RB" \
  --mpi-args "--map-by rankfile:file=$RANKFILE_FOR_MPI --bind-to hwt:overload-allowed --host $HOSTS_WITH_SLOTS --tag-output --mca btl_tcp_if_exclude docker0,lo --prtemca plm_ssh_no_tree_spawn 1" \
  bash -c "ulimit -n 65536; cd $PIPELINE_DIR && source $TT_METAL_HOME/python_env/bin/activate && \
    MESH_DEVICE=QUAD_BH TT_METAL_HOME=$TT_METAL_HOME TT_METAL_SLOW_DISPATCH_MODE=1 \
    pytest -svv --timeout=$PYTEST_TIMEOUT $TEST_PATH" \
  > "$LOG_FILE" 2>&1
RC=$?

echo "[run] exit=$RC"

# Per-rank summary if the run produced verdicts.
NUM_RANKS=$(echo "$HOSTS" | wc -w)
NUM_RANKS=$((NUM_RANKS * 4))
STRIPPED=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG_FILE")
extract_ranks() {
  echo "$STRIPPED" | grep -aE "$1" | sed -E 's/^\[1,([0-9]+)\].*/\1/' | sort -un
}
PASSED_RANKS=$(extract_ranks '^\[1,[0-9]+\]<stdout>: PASSED ' || true)
FAILED_RANKS=$(extract_ranks '^\[1,[0-9]+\]<stdout>: FAILED ' || true)
SKIPPED_RANKS=$(extract_ranks '^\[1,[0-9]+\]<stdout>: SKIPPED ' || true)
ERROR_RANKS=$(extract_ranks '^\[1,[0-9]+\]<stdout>: ERROR ' || true)
N_PASSED=$(echo $PASSED_RANKS | wc -w)
N_FAILED=$(echo $FAILED_RANKS | wc -w)
N_SKIPPED=$(echo $SKIPPED_RANKS | wc -w)
N_ERROR=$(echo $ERROR_RANKS | wc -w)

if [ $RC -ne 0 ]; then
    echo "[run] (non-zero exit — tail of log:)"
    echo "$STRIPPED" | tail -40
fi

if [ $N_FAILED -gt 0 ] || [ $N_ERROR -gt 0 ] || [ $RC -ne 0 ]; then
  VERDICT="FAILED"; COLOR='\033[1;31m'
elif [ $N_PASSED -eq $NUM_RANKS ] && [ $N_SKIPPED -eq 0 ]; then
  VERDICT="PASSED"; COLOR='\033[1;32m'
elif [ $((N_PASSED + N_SKIPPED)) -eq $NUM_RANKS ]; then
  VERDICT="PASSED (some skipped)"; COLOR='\033[1;33m'
else
  VERDICT="INCOMPLETE"; COLOR='\033[1;33m'
fi
RESET='\033[0m'
[ -t 1 ] || { COLOR=''; RESET=''; }

echo
printf "${COLOR}═══════════════════════════════════════════════════════════════${RESET}\n"
printf "${COLOR}[run] %s — PASSED=%d/%d  FAILED=%d  SKIPPED=%d  ERROR=%d${RESET}\n" \
       "$VERDICT" "$N_PASSED" "$NUM_RANKS" "$N_FAILED" "$N_SKIPPED" "$N_ERROR"
[ -n "$FAILED_RANKS" ]  && echo "[run]   failed ranks:  $(echo $FAILED_RANKS  | tr '\n' ' ')"
[ -n "$SKIPPED_RANKS" ] && echo "[run]   skipped ranks: $(echo $SKIPPED_RANKS | tr '\n' ' ')"
[ -n "$ERROR_RANKS" ]   && echo "[run]   error ranks:   $(echo $ERROR_RANKS   | tr '\n' ' ')"
printf "${COLOR}═══════════════════════════════════════════════════════════════${RESET}\n"
exit $RC
