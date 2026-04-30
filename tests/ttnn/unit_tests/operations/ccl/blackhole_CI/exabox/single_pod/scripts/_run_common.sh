#!/bin/bash
# Common launch helper. Sourced by run_chain_test.sh and run_pipeline_test.sh.
#
# Caller must set:
#   TEST                  pytest test name (no .py prefix)
#   TEST_FILE             test file basename, default = test_fake_moe_traffic.py
#   EXTRA_ENV             extra env vars to inject into the bash -c, e.g. "TT_METAL_SLOW_DISPATCH_MODE=1"
#   PYTEST_TIMEOUT        per-test timeout, default 240
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_hosts.sh"

TT_METAL_HOME="${TT_METAL_HOME:-/data/llong/tt-metal}"
PIPELINE_DIR="$(cat /tmp/single_pod_current_dir.txt)"
SSH_WRAPPER="$SCRIPT_DIR/ssh_ulimit_wrapper.sh"
TEST_FILE="${TEST_FILE:-test_fake_moe_traffic.py}"
PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-240}"
EXTRA_ENV="${EXTRA_ENV:-}"

if [ -z "${TEST:-}" ]; then
  echo "Usage: TEST=<test_name> [TEST_FILE=<file.py>] $0" >&2
  exit 2
fi

# PIPELINE_DIR must exist on every remote host (each rank does `cd $PIPELINE_DIR`
# before invoking pytest). /tmp can age out from under us between sessions, so
# we verify and auto-recreate the (mostly-symlink) scaffolding when missing.
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
for h in $SINGLE_POD_HOSTS; do ensure_pipeline_dir_on_host "$h"; done

# Build the comma-separated host list with :4 slot suffix (rankfile drives ranks).
HOSTS_WITH_SLOTS="$(echo "$SINGLE_POD_HOSTS" | tr ' ' '\n' | sed 's/$/:4/' | paste -sd,)"

LOG_FILE="/tmp/single_pod_$(date +%Y%m%d_%H%M%S)_${TEST}.log"
echo "[run] PIPELINE_DIR=$PIPELINE_DIR"
echo "[run] TEST=$TEST_FILE::$TEST"
echo "[run] LOG_FILE=$LOG_FILE"

export PATH=/opt/openmpi-v5.0.7-ulfm/bin:$PATH
ulimit -n 65536
cd "$PIPELINE_DIR"

tt-run --skip-executable-check \
  --rank-binding "$PIPELINE_DIR/blitz_decode_pipeline_rank_binding_single_pod_ci.yaml" \
  --mpi-args "--map-by rankfile:file=$PIPELINE_DIR/blitz_decode_pipeline_rank_file_single_pod_ci --bind-to hwt:overload-allowed --host $HOSTS_WITH_SLOTS --tag-output --mca btl_tcp_if_exclude docker0,lo --prtemca plm_rsh_agent $SSH_WRAPPER --prtemca plm_ssh_no_tree_spawn 1" \
  bash -c "ulimit -n 65536; cd $PIPELINE_DIR && source $TT_METAL_HOME/python_env/bin/activate && \
    MESH_DEVICE=QUAD_BH TT_METAL_HOME=$TT_METAL_HOME $EXTRA_ENV \
    pytest -svv --timeout=$PYTEST_TIMEOUT \
    $TT_METAL_HOME/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/single_pod/$TEST_FILE::$TEST" \
  > "$LOG_FILE" 2>&1
RC=$?

echo "[run] exit=$RC"
echo "[run] tail:"
sed 's/\x1b\[[0-9;]*m//g' "$LOG_FILE" | tail -40
exit $RC
