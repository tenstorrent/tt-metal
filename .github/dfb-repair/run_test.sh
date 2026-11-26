#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# run_test.sh <Suite.Test> [more tests] : run tests on Quasar ttsim: one process per test, a
# fresh kernel cache each time (device-code edits take effect), and a time limit. Prints one
# verdict line per test with its log. Exit code 0 when every test passes or skips.
set -u
if [ -z "${DFB_LOOP_ENV:-}" ] || [ ! -f "$DFB_LOOP_ENV" ]; then
  echo "run_test.sh: DFB_LOOP_ENV is not set; it is set inside a loop attempt" >&2
  exit 2
fi
. "$DFB_LOOP_ENV"
[ $# -gt 0 ] || { echo "usage: run_test.sh <Suite.Test> [more tests]" >&2; exit 2; }
out="${DFB_ATTEMPT_DIR:-/tmp}/test_runs"
mkdir -p "$out"
status=0
for t in "$@"; do
  log="$out/$(printf '%s' "$t" | tr -c 'A-Za-z0-9_.-' '_')_$(date +%H%M%S).log"
  cache=$(mktemp -d "$out/cache.XXXXXX")
  TT_METAL_CACHE="$cache" timeout --foreground -k 30 "$DFB_TEST_TIMEOUT" "$DFB_BIN" --gtest_filter="$t" > "$log" 2>&1
  rc=$?
  rm -rf "$cache"
  if grep -qF "[       OK ] $t (" "$log"; then
    [ $rc -eq 0 ] && v=PASS || v="CRASH-AFTER-PASS(rc=$rc)"
  elif grep -qF "[  SKIPPED ] $t" "$log"; then v=SKIP
  elif grep -qF "[  FAILED  ] $t" "$log"; then v=FAIL
  elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then v=TIMEOUT
  else v="CRASH(rc=$rc)"
  fi
  case $v in PASS|SKIP) ;; *) status=1 ;; esac
  echo "$v $t  log: $log"
done
exit $status
