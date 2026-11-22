#!/bin/bash
# C0 (torrentci#3525): one clang-tidy analysis pass from a cleaned build tree,
# as a queue entry would run it: fresh configure, then the build preset.
#
#   analyse.sh <label> <plain|ctcache> [shard-index shard-count]
#
# ctcache mode runs ctcache 1.2.0 through the shim (local directory backend
# at $CTCACHE_DIR; C0_READ_ONLY and C0_REF_DIR as in ctcache-shim.py).
# Prints one "C0RESULT" line with the timings and counts.
set -euo pipefail
label=$1
mode=$2
shard=${3:-}
shards=${4:-}
tools=${C0_TOOLS:-/tmp/c0}
out=/tmp/c0-out
mkdir -p "$out"
rm -rf .build/clang-tidy
mkdir -p .build/clang-tidy/fixes
if [ "$mode" = ctcache ]; then
  export CTCACHE_S3_OK=1 CTCACHE_BIN="$tools/ctcache-shim.py" CTCACHE_LOCAL=1
  export CTCACHE_KEEP_COMMENTS=1 CTCACHE_NO_LOCAL_STATS=1 C0_LOG="$out/$label.jsonl"
  : "${CTCACHE_DIR:?}"
  rm -f "$C0_LOG"
else
  unset CTCACHE_S3_OK CTCACHE_BIN
fi
t0=$(date +%s.%N)
cmake --preset clang-tidy-fix-parallel \
  -DCMAKE_CXX_CLANG_TIDY="$PWD/cmake/clang-tidy-export-fixes-wrapper.sh;$PWD/.build/clang-tidy/fixes;--warnings-as-errors=*" \
  -DCMAKE_C_CLANG_TIDY="$PWD/cmake/clang-tidy-export-fixes-wrapper.sh;$PWD/.build/clang-tidy/fixes;--warnings-as-errors=*" \
  -DTT_USE_SYSTEM_SFPI=ON >"$out/$label-configure.log" 2>&1 || { tail -50 "$out/$label-configure.log"; exit 1; }
t1=$(date +%s.%N)
ccache -z >/dev/null 2>&1 || true
build_exit=0
if [ -n "$shard" ]; then
  # Every object the full preset builds; a dry run (-n) stops at CMake's
  # glob re-check, so read the command closure instead.
  ninja -C .build/clang-tidy -f build-Debug.ninja -t commands all all_verify_interface_header_sets \
    | grep -o -E ' -o [^ ]+\.o( |$)' | awk '{print $2}' | sort -u >"$out/$label-all-objects.txt"
  awk -v s="$shard" -v n="$shards" '(NR - 1) % n == s' "$out/$label-all-objects.txt" >"$out/$label-objects.txt"
  echo "shard $shard/$shards: $(wc -l <"$out/$label-objects.txt") of $(wc -l <"$out/$label-all-objects.txt") compile edges"
  [ -s "$out/$label-objects.txt" ] || { echo "no compile edges found for the shard"; exit 1; }
  # shellcheck disable=SC2046
  ninja -C .build/clang-tidy -f build-Debug.ninja -k 0 -j "${CMAKE_BUILD_PARALLEL_LEVEL:-16}" \
    $(cat "$out/$label-objects.txt") >"$out/$label-build.log" 2>&1 || build_exit=$?
else
  cmake --build --preset clang-tidy-fix-parallel >"$out/$label-build.log" 2>&1 || build_exit=$?
fi
t2=$(date +%s.%N)
ccache_line=$(ccache --print-stats 2>/dev/null | awk -F'\t' '$1=="direct_cache_hit"||$1=="preprocessed_cache_hit"||$1=="cache_miss"||$1=="remote_storage_hit"||$1=="remote_storage_miss"{printf "%s=%s ", $1, $2}' || true)
fix_files=$(grep -l -s 'DiagnosticName' .build/clang-tidy/fixes/*.yaml 2>/dev/null | wc -l || true)
edges=$(grep -c -E '^\[[0-9]+/[0-9]+\]' "$out/$label-build.log" || true)
stats=""
if [ "$mode" = ctcache ]; then
  stats=$(python3 "$tools/summarize.py" "$C0_LOG")
fi
printf 'C0RESULT label=%s mode=%s configure_s=%.0f build_s=%.0f build_exit=%s edges=%s fix_files=%s %s ccache: %s\n' \
  "$label" "$mode" "$(awk "BEGIN{print $t1 - $t0}")" "$(awk "BEGIN{print $t2 - $t1}")" "$build_exit" "$edges" "$fix_files" "$stats" "$ccache_line" \
  | tee -a "$out/results.txt"
if [ "$build_exit" -ne 0 ]; then
  grep -E 'error:|FAILED' "$out/$label-build.log" | head -20 || true
fi
