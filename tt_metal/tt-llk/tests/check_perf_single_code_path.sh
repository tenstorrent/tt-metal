#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Builds a sample of the perf tests with counters off and on and fails unless both builds are the same program:
# code that differs with counters on brings the counter overhead back. Compile only, no device needed.
#
# Usage: check_perf_single_code_path.sh [arch ...]   (default: wormhole blackhole quasar)
#        PER_FILE=<variants per test file, default 3> JOBS=<pytest workers, default 8>
set -euo pipefail

ARCHES=("$@")
[ ${#ARCHES[@]} -gt 0 ] || ARCHES=(wormhole blackhole quasar)
PER_FILE="${PER_FILE:-3}"
JOBS="${JOBS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
OUT="${RUNNER_TEMP:-$(mktemp -d)}/perf-single-code-path"
rm -rf "$OUT"
mkdir -p "$OUT" perf_data

status=0
for arch in "${ARCHES[@]}"; do
  if [ "$arch" = quasar ]; then
    marks="perf and quasar"
    files=(quasar/perf_*_quasar.py)
  else
    marks="perf and not accuracy and not quasar"
    files=(perf_*.py)
  fi
  CHIP_ARCH="$arch" pytest --collect-only --compile-producer -q -m "$marks" --override-ini=log_cli=false "${files[@]}" \
    > "$OUT/$arch-collect.log" 2>&1 || true
  grep '::' "$OUT/$arch-collect.log" > "$OUT/$arch-all.txt" || true
  # PER_FILE evenly spaced variants of every test file, so the sample spans the formats of each test
  awk -F'::' -v k="$PER_FILE" 'NR == FNR { n[$1]++; next }
    { s = int((n[$1] + k - 1) / k); if ((i[$1]++) % s == 0) print }' \
    "$OUT/$arch-all.txt" "$OUT/$arch-all.txt" > "$OUT/$arch-ids.txt"
  if [ ! -s "$OUT/$arch-ids.txt" ]; then
    tail -20 "$OUT/$arch-collect.log"
    echo "$arch: no perf tests collected" >&2
    exit 1
  fi
  echo "$arch: $(wc -l < "$OUT/$arch-ids.txt") of $(wc -l < "$OUT/$arch-all.txt") perf variants"

  sols=("" "--speed-of-light")
  # the quasar perf tests do not build with --speed-of-light, and CI builds them without it
  [ "$arch" = quasar ] && sols=("")
  for sol in "${sols[@]}"; do
    tag="$arch${sol:+-sol}"
    for counters in off on; do
      flags=()
      [ "$counters" = on ] && flags=(--enable-perf-counters)
      CHIP_ARCH="$arch" RUNNER_TEMP="$OUT/$tag-$counters" xargs -d '\n' -a "$OUT/$arch-ids.txt" \
        pytest -q --compile-producer $sol "${flags[@]}" -n "$JOBS" --timeout=120 \
        --override-ini=log_cli=false > "$OUT/$tag-$counters.log" 2>&1 \
        || { tail -20 "$OUT/$tag-$counters.log"; echo "$tag counters $counters: compile failed" >&2; exit 1; }
    done
    echo "== $tag"
    python3 helpers/perf/single_code_path.py "$OUT/$tag-off" "$OUT/$tag-on" || status=1
  done
done
exit $status
