#!/bin/bash
# Equivalence checks for the cheap merge-gate levers, one leg. Env: ARCH GROUP NGROUPS OUT. cwd = repo root (full checkout).
set -uo pipefail
ROOT=$PWD
T=$ROOT/tt_metal/tt-llk/tests
PT=$T/python_tests
TRIAL=$ROOT/cheap_trial
mkdir -p "$OUT"
BUILD="$RUNNER_TEMP/tt-llk-build"
MARK="perf and not accuracy"
[[ $ARCH == blackhole ]] && MARK+=" and not nightly"
gzip -dc "$TRIAL/$ARCH.json.gz" > "$RUNNER_TEMP/durations.json"
export LLK_SPLIT_DURATIONS="$RUNNER_TEMP/durations.json"
tm() { local label=$1; shift; local s=$(date +%s.%N); "$@" > "$OUT/$label.log" 2>&1; local rc=$?; local e=$(date +%s.%N)
  python3 -c "import json;print(json.dumps({'label':'$label','wall_s':round($e-$s,2),'rc':$rc}))" | tee -a "$OUT/times.jsonl"; tail -2 "$OUT/$label.log"; }
cp "$PT/helpers/test_config.py" "$RUNNER_TEMP/test_config.pr.py"
use_main() { cp "$TRIAL/main/test_config.py" "$PT/helpers/test_config.py"; }
use_pr() { cp "$RUNNER_TEMP/test_config.pr.py" "$PT/helpers/test_config.py"; }

# 1. Collection (lever 2): node lists before/after, both modes, this leg.
cd "$PT"
use_main
for mode in producer consumer; do
  COMMON=(-q --override-ini=log_cli=false -m "$MARK" --timeout=60 --perf-splits "$NGROUPS" --perf-group "$GROUP" --compile-$mode --collect-only -p no:xdist -p no:cacheprovider)
  tm collect_${mode}_before python3 -m pytest "${COMMON[@]}" .
  tm collect_${mode}_after python3 -m pytest "${COMMON[@]}" -o "python_files=perf_*.py" .
  for x in before after; do grep '::' "$OUT/collect_${mode}_$x.log" > "$OUT/nodes_${mode}_$x.txt"; done
  if cmp -s "$OUT/nodes_${mode}_before.txt" "$OUT/nodes_${mode}_after.txt"; then r=IDENTICAL; else r=DIFFERENT; fi
  echo "{\"check\":\"collect_$mode\",\"result\":\"$r\",\"nodes\":$(wc -l < "$OUT/nodes_${mode}_before.txt"),\"nodes_after\":$(wc -l < "$OUT/nodes_${mode}_after.txt")}" | tee -a "$OUT/checks.jsonl"
done
gzip "$OUT"/nodes_*.txt

# 2. Main build in the full tree, file accesses traced (lever 1 audit).
cd "$T"
command -v strace || { apt-get update -qq && apt-get install -y -qq strace; } > /dev/null
export LLK_PERF_PHASE=compile LLK_PERF_COMPILE_WORKERS=${WORKERS:-24}
cp "$TRIAL/main/run_llk_perf_$ARCH.sh" "$T/run_llk_perf_$ARCH.sh"
rm -rf "$BUILD"
tm build_main strace -f -qq -e trace=%file,execve -e signal=none -o "|python3 $TRIAL/audit.py $ROOT - $OUT/audit-compile.json" ./run_llk_perf_$ARCH.sh "$GROUP" "$NGROUPS"
sleep 5
[[ -d "$BUILD" ]] || tm build_main_nostrace ./run_llk_perf_$ARCH.sh "$GROUP" "$NGROUPS"
mv "$BUILD" "$RUNNER_TEMP/build-main"
# Python side of a measuring session: collect in consumer mode under strace too.
cd "$PT"
strace -f -qq -e trace=%file,execve -e signal=none -o "|python3 $TRIAL/audit.py $ROOT - $OUT/audit-consumer.json" python3 -m pytest -q --override-ini=log_cli=false -m "$MARK" --perf-splits "$NGROUPS" --perf-group "$GROUP" --compile-consumer --collect-only -p no:xdist . > /dev/null 2>&1
sleep 5
cd "$T"

# 3. Sparse tree in place (same path), PR code: dedup on, then dedup off with the same flags.
git -C "$ROOT" sparse-checkout set --cone .github tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon ttnn/cpp/ttnn/operations/experimental cheap_trial
ls "$ROOT" "$ROOT/ttnn/cpp/ttnn/operations" | tee "$OUT/sparse-tree.txt"
use_pr
tm build_pr ./run_llk_perf_$ARCH.sh "$GROUP" "$NGROUPS"
mv "$BUILD" "$RUNNER_TEMP/build-pr"
sed -i 's/            if built.exists():/            if False:/' "$PT/helpers/test_config.py"
grep -c 'if False:' "$PT/helpers/test_config.py"
tm build_nodedup ./run_llk_perf_$ARCH.sh "$GROUP" "$NGROUPS"
mv "$BUILD" "$RUNNER_TEMP/build-nodedup"
use_main
tm build_main_sparse ./run_llk_perf_$ARCH.sh "$GROUP" "$NGROUPS"
mv "$BUILD" "$RUNNER_TEMP/build-main-sparse"
python3 "$TRIAL/compare_elfs.py" "$T/sfpi/compiler/bin/riscv-tt-elf-objcopy" "$RUNNER_TEMP" "$OUT/elfs.json" | tee -a "$OUT/checks.jsonl"
