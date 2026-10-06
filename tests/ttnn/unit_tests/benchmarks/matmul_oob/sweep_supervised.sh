#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Run sweep_enumerated.py with --resume until it finishes. When its output stops growing for STALL_MIN minutes (a
# config hung the device), kill it, reset the device and start it again: the resumed run records the hung config
# with status "hang" and moves on. Stops if a restart makes no progress (the hang isn't in a config).
#
#   PYTHON=python_env/bin/python tests/ttnn/unit_tests/benchmarks/matmul_oob/sweep_supervised.sh \
#       --cases-csv cases.csv --out sweep.csv --run my_run
#
# Environment: PYTHON (default python), STALL_MIN (default 20), DEVICE_ID (default 0).

set -u
here=$(dirname "$0")
stall_min=${STALL_MIN:-20}
out=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ ${args[i]} == "--out" ]]; then
        out=${args[i + 1]}
    fi
done
if [[ -z $out ]]; then
    echo "usage: $0 --cases-csv CASES --out OUT [sweep_enumerated.py options]" >&2
    exit 2
fi

last_size=-1
while true; do
    rm -f generated/profiler/.logs/zone_src_locations.log
    "${PYTHON:-python}" "$here/sweep_enumerated.py" "$@" --resume &
    pid=$!
    stalled=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        if [[ -f $out ]] && (($(date +%s) - $(stat -c %Y "$out") > stall_min * 60)); then
            echo "$(date -u +%FT%TZ) supervisor: no output for ${stall_min} min; killing $pid and resetting the device"
            kill -9 "$pid"
            stalled=1
            break
        fi
    done
    wait "$pid"
    status=$?
    if ((stalled == 0)); then
        exit "$status"
    fi
    tt-smi -r "${DEVICE_ID:-0}"
    size=$(stat -c %s "$out")
    if ((size == last_size)); then
        echo "$(date -u +%FT%TZ) supervisor: no progress since the last restart; stopping"
        exit 1
    fi
    last_size=$size
done
