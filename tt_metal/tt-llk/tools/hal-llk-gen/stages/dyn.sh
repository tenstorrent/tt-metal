#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Dynamic analysis (optional, slow: about 1 h per kernel and TRISC for test_eltwise_binary -k add):
# every selected variant is run with one TRISC single-stepped through the tt-exalens GDB server
# (python/dyn/dyn_gdb_plugin.py), which counts the Tensix instructions it issues (TTI executed, TT stores
# into the instruction buffer). One pytest pass per kernel and TRISC, each after tt-smi -r.
#
# Usage: dyn.sh <ws copy> <dyn out dir> <status file> "<test[:k]> ..." "<kernels: original and/or hal>"
# The counts, pytest logs and per-variant status records go to <dyn out dir>. <status file> gets only
# completion, error counts, wall seconds and pytest result lines (no instruction counts).
set -uo pipefail
CW=${1:?ws copy}
DYN_DIR=${2:?dyn out dir}
STATUS=${3:?status file}
TESTS=${4:?tests}
KERNELS=${5:-original hal}
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CT=$CW/tt_metal/tt-llk/tests
python3 -c "import ttexalens" 2>/dev/null || { echo "dyn: tt-exalens is not installed in this python"; exit 2; }
mkdir -p "$DYN_DIR"
for f in "$DYN_DIR"/*.jsonl; do
  [[ -e $f ]] && { echo "dyn: $DYN_DIR already has results of an earlier attempt; refusing to mix"; exit 2; }
done
export TMPDIR=$DYN_DIR/tmp
mkdir -p "$TMPDIR"
echo "dynamic analysis started $(date -u +%FT%TZ), kernels: $KERNELS" >> "$STATUS"
rc_all=0
for kernel in $KERNELS; do
  for e in $TESTS; do
    name=${e%%:*}; k=""; [[ "$e" == *:* ]] && k=${e#*:}
    py=python_tests/$name.py; [[ $kernel == hal ]] && py=python_tests/${name}_hal.py
    kargs=(); [[ -n "$k" ]] && kargs=(-k "$k")
    for core in unpack math pack; do
      tt-smi -r > /dev/null 2>&1 || echo "tt-smi -r failed" >> "$STATUS"
      start=$(date +%s)
      (cd "$CT" && DYN_GDB_CORE=$core DYN_GDB_CLIENT=rsp DYN_GDB_OUT=$DYN_DIR/${kernel}_$core.jsonl \
         DYN_GDB_STATUS=$DYN_DIR/status_${kernel}_$core.jsonl PYTHONPATH=$TOOL_DIR/python/dyn \
         pytest -p dyn_gdb_plugin "$py" ${kargs[@]+"${kargs[@]}"} --timeout=0 -q -p no:cacheprovider > "$DYN_DIR/${kernel}_$core.log" 2>&1)
      rc=$?; (( rc == 0 )) || rc_all=1
      echo "$kernel $core ($name): exit $rc, wall seconds $(( $(date +%s) - start )) | $(grep -E '[0-9]+ (passed|failed|error)' "$DYN_DIR/${kernel}_$core.log" | tail -1)" >> "$STATUS"
    done
  done
done
ref=(); [[ -n "${DYN_REFERENCE:-}" ]] && ref=(--reference "$DYN_REFERENCE")
python3 "$TOOL_DIR/python/dyn/dyn_status.py" "$DYN_DIR" --kernels $KERNELS ${ref[@]+"${ref[@]}"} >> "$STATUS"
rm -rf "$TMPDIR"
echo "dynamic analysis ended $(date -u +%FT%TZ)" >> "$STATUS"
exit $rc_all
