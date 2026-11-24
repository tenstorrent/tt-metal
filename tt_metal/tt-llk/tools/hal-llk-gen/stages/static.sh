#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Static counts: compile-only builds (no card) of every original and HAL test, once with LLK asserts on
# (the test framework default) and once with TT_LLK_DISABLE_ASSERTS=1, then code size (codesize.py) and,
# if ELFRAY is set, TTI / TT counts (tt_tti_counts.py) per build.
#
# Usage: static.sh <ws copy> <out dir> "<test[:k]> ..."
#   <ws copy> = hard-link copy of the run's workspace (made by hal-llk-gen; never the run's own ws)
# Output: <out>/{assert_on,assert_off}/{compile.txt,codesize.{json,md},tt_tti_counts.{json,md}}, <out>/status.txt
set -uo pipefail
CW=${1:?ws copy}
OUT=${2:?out dir}
TESTS=${3:?tests}
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CT=$CW/tt_metal/tt-llk/tests
TOOLS=$CT/sfpi/compiler/bin
mkdir -p "$OUT"

files=(); ks=()
for e in $TESTS; do
  name=${e%%:*}; k=""; [[ "$e" == *:* ]] && k=${e#*:}
  files+=("python_tests/$name.py" "python_tests/${name}_hal.py"); ks+=("$k")
done
kargs=(-k "${ks[0]}")   # a common -k filter only if all tests have the same one (as the harness)
for k in "${ks[@]}"; do [[ "$k" == "${ks[0]}" ]] || kargs=(); done
[[ -z "${ks[0]}" ]] && kargs=()
first=${TESTS%% *}; first=${first%%:*}
orig=$(grep -oE 'sources/[A-Za-z0-9_]+\.cpp' "$CT/python_tests/$first.py" | head -1 | xargs basename)
hal=$(grep -oE 'sources/[A-Za-z0-9_]+\.cpp' "$CT/python_tests/${first}_hal.py" | head -1 | xargs basename)

rc_all=0
for mode in assert_on assert_off; do
  D=$OUT/$mode; rm -rf "$D"; mkdir -p "$D/codesize_tmp"
  if [[ $mode == assert_off ]]; then export TT_LLK_DISABLE_ASSERTS=1; else unset TT_LLK_DISABLE_ASSERTS; fi
  env | grep -E '^TT_LLK_DISABLE_ASSERTS=' > "$D/env.txt"
  (cd "$CT" && TMPDIR=$D/codesize_tmp nice -n 19 timeout 2h pytest "${files[@]}" ${kargs[@]+"${kargs[@]}"} \
     --compile-producer -n 8 -q -p no:cacheprovider > "$D/compile.txt" 2>&1)
  rc=$?; (( rc == 0 )) || rc_all=1
  echo "$mode: compile exit $rc, $(grep -E '(passed|failed|error)' "$D/compile.txt" | tail -1)" | tee -a "$OUT/status.txt"
  nice -n 19 python3 "$TOOL_DIR/harness/codesize.py" --artefacts "$D/codesize_tmp/tt-llk-build" --tools "$TOOLS" \
    --orig "$orig" --hal "$hal" --json "$D/codesize.json" --md "$D/codesize.md" > /dev/null 2>&1 \
    || { echo "$mode: codesize.py failed" | tee -a "$OUT/status.txt"; rc_all=1; }
  if [[ -n "${ELFRAY:-}" ]]; then
    ELFRAY=$ELFRAY nice -n 19 python3 "$TOOL_DIR/python/tt_tti_counts.py" "$D" --objdump "$TOOLS/riscv-tt-elf-objdump" \
      --orig "$orig" --hal "$hal" --md "$D/tt_tti_counts.md" --json "$D/tt_tti_counts.json" > "$D/tt_tti_counts_stdout.txt" 2>&1 \
      || { echo "$mode: tt_tti_counts.py failed" | tee -a "$OUT/status.txt"; rc_all=1; }
  fi
done
unset TT_LLK_DISABLE_ASSERTS
[[ -n "${ELFRAY:-}" ]] || echo "ELFRAY not set: code size only, no TTI/TT counts" | tee -a "$OUT/status.txt"
exit $rc_all
