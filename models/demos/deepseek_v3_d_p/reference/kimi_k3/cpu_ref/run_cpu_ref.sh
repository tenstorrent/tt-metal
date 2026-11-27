#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Detached launcher for k31_cpu_ref.py on an allocated Exabox CPU host (no device is opened).
# usage: run_cpu_ref.sh <tag> <k31_cpu_ref.py args...>   -> log in $LOGDIR/<tag>.log
# env: PY (python with torch + safetensors), LOGDIR, SCRIPTS
PY=${PY:-/data/nmaurice/tt-blaze-m3-1-flash-perf-baseline/tt-metal/python_env/bin/python}
LOGDIR=${LOGDIR:-/data/nmaurice/k31-gptq/cpu_ref/logs}
SCRIPTS=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
tag=$1; shift
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-30} MKL_NUM_THREADS=${MKL_NUM_THREADS:-30} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd /tmp
setsid nohup bash -c "echo start \$(date -u) \$(hostname -s); $PY $SCRIPTS/k31_cpu_ref.py $*; echo end rc=\$? \$(date -u)" > "$LOGDIR/$tag.log" 2>&1 < /dev/null &
echo "launched $tag pid $! log $LOGDIR/$tag.log"
