#!/bin/bash
# TRISC-once fix trial, one leg. Env: ARCH GROUP NGROUPS OUT. cwd = repo root.
set -uo pipefail
ROOT=$PWD
T=$ROOT/tt_metal/tt-llk/tests
PT=$T/python_tests
TRIAL=$ROOT/trisc_trial
GXX=$T/sfpi/compiler/bin/riscv-tt-elf-g++
mkdir -p "$OUT"
BUILD="$RUNNER_TEMP/tt-llk-build"
gzip -dc "$TRIAL/$ARCH.json.gz" > "$RUNNER_TEMP/durations.json"
export LLK_SPLIT_DURATIONS="$RUNNER_TEMP/durations.json"
export LLK_PERF_PHASE=compile LLK_PERF_COMPILE_WORKERS=${WORKERS:-24}
tm() { local label=$1; shift; python3 "$TRIAL/timed.py" "$label" "$OUT/times.jsonl" "$OUT/$label.log" "$@"; tail -2 "$OUT/$label.log"; }
cp "$PT/helpers/test_config.py" "$RUNNER_TEMP/new.py"
cp "$TRIAL/old_test_config.py" "$RUNNER_TEMP/old.py"
sed 's/                self._compile_kernel_part_once(compile_command, source)/                run_shell_command(compile_command, TestConfig.TESTS_WORKING_DIR, source)/' "$RUNNER_TEMP/new.py" > "$RUNNER_TEMP/plain.py"
grep -c "_compile_kernel_part_once(compile_command" "$RUNNER_TEMP/plain.py"
nproc; free -g

# 1. Hardware-free tests of the key, with the SFPI compiler.
cd "$PT"
LLK_HARDWARE_FREE=1 python3 -m pytest -c pytest.ini --rootdir=. --confcutdir=. -p helpers.llk_pytest_plugin --override-ini=log_cli=false -q \
  ../hardware_free/harness/test_config.py > "$OUT/hardware_free.log" 2>&1; echo "hardware_free rc=$?"; tail -3 "$OUT/hardware_free.log"
cp "$RUNNER_TEMP/old.py" helpers/test_config.py
LLK_HARDWARE_FREE=1 python3 -m pytest -c pytest.ini --rootdir=. --confcutdir=. -p helpers.llk_pytest_plugin --override-ini=log_cli=false -q \
  ../hardware_free/harness/test_config.py > "$OUT/hardware_free_old.log" 2>&1; echo "hardware_free_old rc=$?"; tail -3 "$OUT/hardware_free_old.log"
cp "$RUNNER_TEMP/new.py" helpers/test_config.py

# 2. Property test with the SFPI compiler, old and new key.
cd "$TRIAL"
./mixed.sh "$RUNNER_TEMP/old.py" old ${TRIALS:-150} "$OUT/property" "$GXX" | tee -a "$OUT/checks.txt"
./mixed.sh "$RUNNER_TEMP/new.py" new ${TRIALS:-150} "$OUT/property" "$GXX" | tee -a "$OUT/checks.txt"
find "$OUT/property" -name w -type d -prune -exec rm -rf {} +

# 3. Compile sessions: plain (no reuse), old key, new key; then plain and new again for timing.
cd "$T"
for arm in plain old new plain new; do
  cp "$RUNNER_TEMP/$arm.py" "$PT/helpers/test_config.py"
  rm -rf "$BUILD"
  label=build_$arm; [[ -d "$RUNNER_TEMP/build-$arm" ]] && label=build_${arm}_2
  tm $label ./run_llk_perf_$ARCH.sh "$GROUP" "$NGROUPS"
  if [[ -d "$RUNNER_TEMP/build-$arm" ]]; then rm -rf "$BUILD"; else mv "$BUILD" "$RUNNER_TEMP/build-$arm"; fi
done
cp "$RUNNER_TEMP/new.py" "$PT/helpers/test_config.py"
python3 "$TRIAL/compare.py" "$T/sfpi/compiler/bin/riscv-tt-elf-objcopy" "$RUNNER_TEMP" "$OUT/elfs.json" | tee -a "$OUT/checks.txt"
python3 "$TRIAL/shapes.py" "$RUNNER_TEMP/old.py" "$RUNNER_TEMP/new.py" "$RUNNER_TEMP/build-new" "$OUT" | tee -a "$OUT/checks.txt"
