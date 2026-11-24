#!/usr/bin/env bash
# Verify S on a CPU runner: compile F and S (--node-list) as the gate does (collapse on, L1_TO_L1, non-SoL),
# time both, count collected tests, fingerprint both build trees.  env: ARCH OUT WORKERS RUNNER_TEMP GITHUB_WORKSPACE
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cd "$GITHUB_WORKSPACE/tt_metal/tt-llk/tests"
./setup_testing_env.sh > "$OUT/setup.log" 2>&1
BIN="$PWD/sfpi/compiler/bin"
cd python_tests
for SET in F S; do
  rm -rf "$RUNNER_TEMP/tt-llk-build"
  EXTRA=(.)
  if [[ $SET == S ]]; then
    mapfile -t FILES < <(cut -d: -f1 "$T/nodes_${ARCH}.txt" | sort -u)
    EXTRA=(--node-list="$T/nodes_${ARCH}.txt" "${FILES[@]}")
  fi
  start=$(date +%s)
  set +e
  LLK_PERF_RUN_TYPES=L1_TO_L1 TT_LLK_DISABLE_ASSERTS=1 timeout -k 30 3600 pytest -q --override-ini=log_cli=false \
    --compile-producer -n "$WORKERS" -m "perf and not accuracy and not nightly" --timeout=600 "${EXTRA[@]}" > "$OUT/pytest_$SET.log" 2>&1
  rc=$?
  set -e
  echo "$SET rc=$rc wall=$(( $(date +%s) - start ))s $(tail -1 "$OUT/pytest_$SET.log")" | tee -a "$OUT/verify.txt"
  python3 "$T/hash_elfs.py" "$RUNNER_TEMP/tt-llk-build/sources" "$BIN/riscv-tt-elf-objdump" "$PWD/helpers" "$OUT/elf_hashes_$SET.csv"
  gzip -f "$OUT/elf_hashes_$SET.csv"
done
