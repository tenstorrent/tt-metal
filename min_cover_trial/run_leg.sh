#!/usr/bin/env bash
# Compile the whole merge-gate suite of one arch (L1_TO_L1, non-SoL, asserts off, no runtime collapse),
# record nodeid -> variant map, then fingerprint every ELF.  env: ARCH OUT WORKERS RUNNER_TEMP GITHUB_WORKSPACE
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/map"
cd "$GITHUB_WORKSPACE/tt_metal/tt-llk/tests"
./setup_testing_env.sh > "$OUT/setup.log" 2>&1
BIN="$PWD/sfpi/compiler/bin"
cd python_tests
start=$(date +%s)
set +e
LLK_CODEPATH_DIR="$OUT/map" LLK_PERF_RUN_TYPES=L1_TO_L1 TT_LLK_DISABLE_ASSERTS=1 \
  timeout -k 30 5400 pytest -q --override-ini=log_cli=false --compile-producer -n "$WORKERS" \
  -m "perf and not accuracy and not nightly" --timeout=600 . > "$OUT/pytest.log" 2>&1
echo "pytest rc=$? wall=$(( $(date +%s) - start ))s" | tee "$OUT/compile.txt"
set -e
tail -5 "$OUT/pytest.log"
cat "$OUT"/map/map.*.jsonl | gzip > "$OUT/map.jsonl.gz"; rm -rf "$OUT/map"
start=$(date +%s)
python3 "$T/hash_elfs.py" "$RUNNER_TEMP/tt-llk-build/sources" "$BIN/riscv-tt-elf-objdump" "$PWD/helpers" "$OUT/elf_hashes.csv"
echo "hash wall=$(( $(date +%s) - start ))s" | tee -a "$OUT/compile.txt"
gzip "$OUT/elf_hashes.csv"
"$BIN/riscv-tt-elf-g++" --version | head -1 >> "$OUT/compile.txt"
git -C "$GITHUB_WORKSPACE" rev-parse HEAD >> "$OUT/compile.txt"
