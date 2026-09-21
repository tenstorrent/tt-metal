#!/usr/bin/env bash
# Cooperative machine-wide lock shared with scripts/run_safe_pytest.sh.
# No automatic reset: failed experiments are inspected by the coordinating agent.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
if [[ $# -lt 1 ]]; then
    echo "usage: bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh SCRIPT [ARGS...]" >&2
    exit 2
fi
exec 9>/tmp/tt-device.lock
echo "SPRINT waiting for exclusive device lock: $*"
flock -x 9
if [[ -e /tmp/tt-device.dirty ]]; then
    echo "SPRINT BLOCKED: prior run left device dirty; coordinator must inspect/recover" >&2
    exit 3
fi
export TT_METAL_HOME="$ROOT"
export PYTHONPATH="$ROOT:$ROOT/ttnn:$ROOT/tools:/localdev/cglagovich/tt-metal-blackhole-20260908/python_env/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/localdev/cglagovich/tt-metal-blackhole-20260908/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/localdev/cglagovich/compute-sprint-20260918/jit-cache}"
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export OMP_NUM_THREADS=8
export TT_METAL_OPERATION_TIMEOUT_SECONDS=30
touch /tmp/tt-device.dirty
echo "SPRINT start $(date -u +%FT%TZ) pid=$$ command=$*"
set +e
timeout --signal=TERM --kill-after=15s "${SDPA_SPRINT_TIMEOUT:-600}s" /opt/venv/bin/python "$@"
result=$?
set -e
if [[ $result -eq 0 ]]; then
    rm /tmp/tt-device.dirty
else
    echo "SPRINT FAILED: exit=$result; leaving dirty marker, no automatic device reset" >&2
fi
echo "SPRINT end $(date -u +%FT%TZ) exit=$result"
exit "$result"
