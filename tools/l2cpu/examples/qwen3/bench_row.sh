#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
#
# One benchmark row = one process from a fresh chip reset through scripts/l2cpu_run.sh (lock + reset + run; the
# x280 harts leave reset once per chip reset). Source scripts/l2cpu_env.sh first.
# Usage: bench_row.sh [--image <sampling firmware .bin>] <bench.py args...>
# Log: $L2CPU_QWEN3_OUT/bench/log_<args>.log (default ./l2cpu_qwen3_out).
HERE=$(cd "$(dirname "$0")" && pwd)
img=""
if [ "$1" = "--image" ]; then img=$2; shift 2; fi
out=${L2CPU_QWEN3_OUT:-l2cpu_qwen3_out}/bench
mkdir -p "$out"
tag=$( (basename "${img:-default}" .bin; echo "$*") | tr ' :/\n' '____' | tr -cd 'A-Za-z0-9._-')
log=$out/log_${tag}.log
if [ -n "$img" ]; then export L2S_FW_IMAGE=$img; fi
"$HERE/../../scripts/l2cpu_run.sh" "bench $*" "${PY:-python3}" -u "$HERE/bench.py" "$@" > "$log" 2>&1
rc=$?
echo "RC=$rc ${img:-default image} $*"; grep -E "ms/token median|Traceback|Error" "$log" | tail -3
exit $rc
