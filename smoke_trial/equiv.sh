#!/bin/bash
# Smoke-leg equivalence, one leg. Env: ARCH GROUP NGROUPS OUT DURATIONS. cwd = repo root (full checkout of the lever branch).
set -uo pipefail
ROOT=$PWD
T=$ROOT/tt_metal/tt-llk/tests
PT=$T/python_tests
TRIAL=$ROOT/smoke_trial
mkdir -p "$OUT"
BUILD="$RUNNER_TEMP/tt-llk-build"
MARK="not perf and not nightly and not quasar and not accuracy"
SPLIT=(--splits "$NGROUPS" --group "$GROUP")
[[ $ARCH == blackhole ]] && SPLIT+=(--splitting-algorithm least_duration)
cp "$DURATIONS" "$PT/.test_durations"
tm() { local label=$1; shift; local s=$(date +%s.%N); "$@" > "$OUT/$label.log" 2>&1; local rc=$?; local e=$(date +%s.%N)
  python3 -c "import json;print(json.dumps({'label':'$label','wall_s':round($e-$s,2),'rc':$rc}))" | tee -a "$OUT/times.jsonl"; tail -2 "$OUT/$label.log"; }
cp "$PT/helpers/test_config.py" "$RUNNER_TEMP/test_config.pr.py"
use_main() { cp "$TRIAL/main/test_config.py" "$PT/helpers/test_config.py"; }
use_pr() { cp "$RUNNER_TEMP/test_config.pr.py" "$PT/helpers/test_config.py"; }

# 1. Collection: node lists before/after, as the leg collects (default mode) and in producer mode.
cd "$PT"
use_main
for mode in default producer; do
  M=(); [[ $mode == producer ]] && M=(--compile-producer)
  COMMON=(-q --override-ini=log_cli=false -m "$MARK" "${SPLIT[@]}" --timeout=60 "${M[@]}" --collect-only -n 0 -p no:cacheprovider)
  tm collect_${mode}_before python3 -m pytest "${COMMON[@]}" .
  tm collect_${mode}_after python3 -m pytest "${COMMON[@]}" -o "python_files=test_*.py" .
  for x in before after; do grep '::' "$OUT/collect_${mode}_$x.log" > "$OUT/nodes_${mode}_$x.txt"; done
  if cmp -s "$OUT/nodes_${mode}_before.txt" "$OUT/nodes_${mode}_after.txt"; then r=IDENTICAL; else r=DIFFERENT; fi
  echo "{\"check\":\"collect_$mode\",\"result\":\"$r\",\"nodes\":$(wc -l < "$OUT/nodes_${mode}_before.txt"),\"nodes_after\":$(wc -l < "$OUT/nodes_${mode}_after.txt")}" | tee -a "$OUT/checks.jsonl"
done
gzip "$OUT"/nodes_*.txt
PRODUCE=(python3 -m pytest -q --override-ini=log_cli=false -n auto --dist=worksteal -m "$MARK" "${SPLIT[@]}" --timeout=60 --compile-producer -p no:cacheprovider -o "python_files=test_*.py" .)

# 2. Main build in the full tree, file accesses traced (sparse checkout audit), and a traced collection.
command -v strace || { apt-get update -qq && apt-get install -y -qq strace; } > /dev/null
rm -rf "$BUILD"
tm build_main strace -f -qq -e trace=%file,execve -e signal=none -o "|python3 $TRIAL/audit.py $ROOT - $OUT/audit-compile.json" "${PRODUCE[@]}"
sleep 5
mv "$BUILD" "$RUNNER_TEMP/build-main"
strace -f -qq -e trace=%file,execve -e signal=none -o "|python3 $TRIAL/audit.py $ROOT - $OUT/audit-collect.json" python3 -m pytest -q --override-ini=log_cli=false -m "$MARK" "${SPLIT[@]}" --collect-only -n 0 -p no:cacheprovider . > /dev/null 2>&1
(cd "$T" && strace -f -qq -e trace=%file,execve -e signal=none -o "|python3 $TRIAL/audit.py $ROOT - $OUT/audit-setup.json" ./setup_testing_env.sh > /dev/null 2>&1)
sleep 5

# 3. Sparse tree in place, main code; then the lever code (TRISC once), then the lever code without reuse.
git -C "$ROOT" sparse-checkout set --no-cone /.github/ /tt_metal/tt-llk/ /tt_metal/hw/ /tt_metal/hostdevcommon/ /tt_metal/sfpi-info.sh /tt_metal/sfpi-version /ttnn/cpp/ttnn/operations/experimental/ /smoke_trial/
ls "$ROOT" "$ROOT/tt_metal" | tee "$OUT/sparse-tree.txt"
cp "$DURATIONS" "$PT/.test_durations"
use_main
tm build_main_sparse "${PRODUCE[@]}"
mv "$BUILD" "$RUNNER_TEMP/build-main-sparse"
use_pr
tm build_pr "${PRODUCE[@]}"
mv "$BUILD" "$RUNNER_TEMP/build-pr"
sed -i 's/            if built.exists():/            if False:/' "$PT/helpers/test_config.py"
grep -c 'if False:' "$PT/helpers/test_config.py"
tm build_nodedup "${PRODUCE[@]}"
mv "$BUILD" "$RUNNER_TEMP/build-nodedup"
use_pr
python3 "$TRIAL/compare_elfs.py" "$T/sfpi/compiler/bin/riscv-tt-elf-objcopy" "$RUNNER_TEMP" "$OUT/elfs.json" | tee -a "$OUT/checks.jsonl"
