#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Postcheck: the 4-step state sequence on the run's own workspace, after the harness has finished.
#   1. tt-smi -r, every HAL test        3. no reset, every HAL test
#   2. no reset, every original test    4. tt-smi -r, every HAL test
# Then (if ELFRAY is set) static TT/TTI counts of the harness build (<run>/codesize_tmp).
#
# Usage: postcheck.sh <run dir> <out dir> "<test[:k]> ..."
# Output: <out>/checks.txt (one line per test and step: "<file>: exit <rc>, <last pytest line>"),
#         <out>/stepN_*.txt pytest logs, <out>/tt_tti_counts.{md,json}
set -uo pipefail
RUN_DIR=${1:?run dir}
OUT=${2:?out dir}
TESTS=${3:?tests}
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
T=$RUN_DIR/ws/tt_metal/tt-llk/tests
TEST_TIMEOUT=${TEST_TIMEOUT:-60m}
[[ -d $T ]] || { echo "postcheck: $T does not exist"; exit 2; }
mkdir -p "$OUT"
export TMPDIR=$RUN_DIR/tmp

run_k() {  # <python test> <k> <log>
  local args=("python_tests/$1.py")
  [[ -n "$2" ]] && args+=(-k "$2")
  (cd "$T" && timeout "$TEST_TIMEOUT" pytest "${args[@]}" -q > "$3" 2>&1)
  echo "$(basename "$3"): exit $?, $(tail -1 "$3")"
}
each() {  # <step> <hal|original>
  local e name k
  for e in $TESTS; do
    name=${e%%:*}; k=""; [[ "$e" == *:* ]] && k=${e#*:}
    if [[ $2 == hal ]]; then run_k "${name}_hal" "$k" "$OUT/step$1_hal_$name.txt"
    else run_k "$name" "$k" "$OUT/step$1_original_$name.txt"; fi
  done
}

{
  echo "== 1. tt-smi -r, HAL tests";       tt-smi -r > "$OUT/reset1.txt" 2>&1; each 1 hal
  echo "== 2. no reset, original tests";  each 2 original
  echo "== 3. no reset, HAL tests";       each 3 hal
  echo "== 4. tt-smi -r, HAL tests";       tt-smi -r > "$OUT/reset4.txt" 2>&1; each 4 hal
} | tee "$OUT/checks.txt"

if [[ -n "${ELFRAY:-}" ]]; then
  first=${TESTS%% *}; first=${first%%:*}
  orig=$(grep -oE 'sources/[A-Za-z0-9_]+\.cpp' "$T/python_tests/$first.py" | head -1 | xargs basename)
  hal=$(grep -oE 'sources/[A-Za-z0-9_]+\.cpp' "$T/python_tests/${first}_hal.py" | head -1 | xargs basename)
  ELFRAY=$ELFRAY python3 "$TOOL_DIR/python/tt_tti_counts.py" "$RUN_DIR" --objdump "$T/sfpi/compiler/bin/riscv-tt-elf-objdump" \
    --orig "$orig" --hal "$hal" --md "$OUT/tt_tti_counts.md" --json "$OUT/tt_tti_counts.json" \
    > "$OUT/tt_tti_counts_stdout.txt" 2>&1 || echo "tt_tti_counts failed (see $OUT/tt_tti_counts_stdout.txt)"
fi
n_ok=$(grep -c ': exit 0,' "$OUT/checks.txt")
n_all=$(grep -c ': exit ' "$OUT/checks.txt")
echo "postcheck: $n_ok of $n_all pytest runs exit 0"
(( n_ok == n_all ))
