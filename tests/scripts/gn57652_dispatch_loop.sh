#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Issue #57652: loop the failing legacy group norm test for the "Custom test dispatch" workflow.
#
#   bash tests/scripts/gn57652_dispatch_loop.sh single [N]  the failing test only, N times in one pytest run (default 2000)
#   bash tests/scripts/gn57652_dispatch_loop.sh group [N]   the whole Merge Gate group (the CI job command; at this commit it
#                                                           includes the failing test), up to N passes or ~95 minutes
set -u
MODE=${1:-}
N=${2:-2000}
TEST="tests/ttnn/unit_tests/operations/fused/test_group_norm.py::test_group_norm_with_height_sharded"
CASE=${GN57652_CASE:-legacy}
BUDGET_MIN=${GN57652_BUDGET_MIN:-95}
GROUP_DIRS="tests/ttnn/unit_tests/operations/matmul tests/ttnn/unit_tests/operations/fused tests/ttnn/unit_tests/operations/reduce"

echo "=== machine"
echo "runner=${RUNNER_NAME:-?} host=$(hostname) commit=$(git -c safe.directory='*' rev-parse --short HEAD 2>/dev/null || echo '?')"
for d in /sys/class/tenstorrent/*/; do
    [ -e "$d/tt_serial" ] || continue
    echo "card: type=$(cat "$d/tt_card_type") serial=$(cat "$d/tt_serial") fw_bundle=$(cat "$d/tt_fw_bundle_ver")" \
        "arc_fw=$(cat "$d/tt_arc_fw_ver") eth_fw=$(cat "$d/tt_eth_fw_ver") aiclk=$(cat "$d/tt_aiclk")"
done 2>/dev/null

python3 -c "import pytest_repeat" 2>/dev/null || uv pip install -q pytest-repeat 2>/dev/null || pip install -q pytest-repeat
# Same settings as the Merge Gate job: quiet device logs, and a hung op fails after 5 s instead of blocking.
export TT_LOGGER_LEVEL=${TT_LOGGER_LEVEL:-warn} TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5}

case $MODE in
single)
    echo "=== single: $TEST [$CASE] x $N"
    pytest -q -rf --tb=short --maxfail=20 --count="$N" "$TEST" -k "$CASE"
    rc=$?
    echo "=== single done: exit code $rc (0 means all $N runs passed)"
    exit $rc
    ;;
group)
    echo "=== group: pytest --timeout 60 -x -m merge_gate $GROUP_DIRS, up to $N passes or $BUDGET_MIN minutes"
    start=$SECONDS
    runs=0
    failed=0
    while [ "$runs" -lt "$N" ] && [ $(((SECONDS - start) / 60)) -lt "$BUDGET_MIN" ]; do
        runs=$((runs + 1))
        log=/tmp/gn57652_group_$runs.log
        pytest --timeout 60 -x -q -rf --tb=short -m merge_gate $GROUP_DIRS >"$log" 2>&1
        rc=$?
        echo "pass $runs: exit $rc,$(grep -E '^=+ .*(passed|failed|error).* =+$' "$log" | tail -n 1 | tr -d '=') ($(((SECONDS - start) / 60)) min)"
        if [ "$rc" -ne 0 ]; then
            failed=$((failed + 1))
            tail -n 80 "$log"
        fi
    done
    echo "=== group done: $runs passes, $failed failed"
    [ "$failed" -eq 0 ]
    ;;
*)
    echo "usage: $0 single|group [N]"
    exit 2
    ;;
esac
