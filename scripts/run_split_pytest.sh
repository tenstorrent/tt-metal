#!/usr/bin/env bash

set -uo pipefail

usage() {
    echo "Usage: $0 <device-count> <test-path> [pytest arguments...]" >&2
}

if [[ $# -lt 2 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
    usage
    exit 2
fi

device_count=$1
test_path=$2
shift 2
pytest_args=("$@")

if ! command -v pytest >/dev/null 2>&1; then
    echo "ERROR: pytest was not found in PATH" >&2
    exit 2
fi

if ! pytest_help=$(pytest --help 2>&1); then
    echo "ERROR: unable to inspect pytest options" >&2
    exit 2
fi
if [[ "$pytest_help" != *"--splits"* || "$pytest_help" != *"--group"* ]]; then
    echo "ERROR: pytest-split is not installed" >&2
    exit 2
fi

pids=()
devices=()
groups=()

cleanup() {
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
}
trap 'cleanup; exit 130' INT TERM HUP

for ((device = 0; device < device_count; device++)); do
    group=$((device + 1))
    (
        set -o pipefail
        TT_VISIBLE_DEVICES="$device" pytest "$test_path" \
            --splits "$device_count" --group "$group" "${pytest_args[@]}" 2>&1 |
            sed -u "s/^/[device ${device} group ${group}] /"
    ) &
    pids+=("$!")
    devices+=("$device")
    groups+=("$group")
done

failed=()
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        failed+=("device ${devices[$i]} / group ${groups[$i]}")
    fi
done

if (( ${#failed[@]} > 0 )); then
    printf 'FAILED: %s\n' "${failed[@]}" >&2
    exit 1
fi

exit 0
