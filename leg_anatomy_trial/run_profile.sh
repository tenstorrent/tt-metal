#!/bin/bash
# v2: harness floor with a copy-stub compiler. Env: ARCH GROUP NGROUPS OUT; cwd = tests/python_tests
set -uo pipefail
mkdir -p "$OUT"
MARK="perf and not accuracy and not nightly"
COMMON=(-q --override-ini=log_cli=false -o "addopts=-s" -m "$MARK" --timeout=600 --perf-splits "$NGROUPS" --perf-group "$GROUP" -p no:cacheprovider)
T() { local label=$1; shift; local s=$(date +%s.%N); "$@" > "$OUT/$label.log" 2>&1; local rc=$?; local e=$(date +%s.%N)
  python3 -c "import json;print(json.dumps({'label':'$label','wall_s':round($e-$s,2),'rc':$rc}))" >> "$OUT/times.jsonl"; tail -1 "$OUT/$label.log"; }
BUILD="$RUNNER_TEMP/tt-llk-build"; SAVE="$RUNNER_TEMP/saved-build"
BIN=../sfpi/compiler/bin
T collect_producer_n0 python3 -m pytest "${COMMON[@]}" --compile-producer --collect-only -p no:xdist
T collect_producer_n0_prof python3 -m cProfile -o "$OUT/collect.prof" -m pytest "${COMMON[@]}" --compile-producer --collect-only -p no:xdist
T collect_consumer_n0 python3 -m pytest "${COMMON[@]}" --compile-consumer --collect-only -p no:xdist
T collect_all_n0 python3 -m pytest -q -o "addopts=-s" -m "$MARK" --compile-producer --collect-only -p no:xdist -p no:cacheprovider
T cold_n10 python3 -m pytest "${COMMON[@]}" --compile-producer -n 10 --junitxml="$OUT/cold_n10.xml" .
rm -rf "$SAVE"; cp -a "$BUILD" "$SAVE"
# stub compiler: copy the cold ELF at the same -o path
mv "$BIN/riscv-tt-elf-g++" "$BIN/riscv-tt-elf-g++.real"
cat > "$BIN/riscv-tt-elf-g++" <<STUB
#!/bin/bash
cat > /dev/null
out=""; prev=""; for a in "\$@"; do [[ "\$prev" == "-o" ]] && out="\$a"; prev="\$a"; done
src="$SAVE\${out#$BUILD}"
if [[ -f "\$src" ]]; then mkdir -p "\$(dirname "\$out")"; cp "\$src" "\$out"; else exec "$BIN/riscv-tt-elf-g++.real" "\$@"; fi
STUB
chmod +x "$BIN/riscv-tt-elf-g++"
T floor_n10 python3 -m pytest "${COMMON[@]}" --compile-producer -n 10 --junitxml="$OUT/floor_n10.xml" .
T floor_n0_prof python3 -m cProfile -o "$OUT/floor_n0.prof" -m pytest "${COMMON[@]}" --compile-producer -p no:xdist --junitxml="$OUT/floor_n0.xml" .
T floor_n0 python3 -m pytest "${COMMON[@]}" --compile-producer -p no:xdist .
mv "$BIN/riscv-tt-elf-g++.real" "$BIN/riscv-tt-elf-g++"
for p in collect floor_n0; do
  python3 -c "import pstats; s=pstats.Stats('$OUT/$p.prof'); s.sort_stats('cumulative').print_stats(70); s.sort_stats('tottime').print_stats(40)" > "$OUT/$p.prof.txt" 2>&1
done
