#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Wormhole LLK perf runner, shared by the 5 wh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# pytest-split sharding: compile this shard's items (producer), then measure
# them (consumer) -- one invocation each over the whole perf suite.
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_wormhole.sh <group> <n_groups>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-true}"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"

case "$SPEED_OF_LIGHT" in
  true)
    SPEED_OF_LIGHT_ARGS=(--speed-of-light)
    ;;
  false)
    SPEED_OF_LIGHT_ARGS=()
    ;;
  *)
    echo "SPEED_OF_LIGHT must be 'true' or 'false', got '$SPEED_OF_LIGHT'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

# EXPERIMENT (not for merge): does a per-suite chip reset recover the FASTER level?
#
# main already schedules the measuring pass one test at a time (--dist loadgroup, #57015),
# which made the nightly reproducible -- but it settles on the CONTENDED (slow) level:
# 09-20 -> 09-22 moved 2.81% of PACK_ISOLATE configs slower, 0.00% faster, total pack
# cycles +0.74%. The 2x2 factorial says only a chip reset recovers the uncontended values,
# because the mechanism is sequential state carry-over between suites.
#
# This keeps --dist loadgroup, splits the run into one pytest invocation PER TEST FILE so
# there are suite boundaries to reset at, and issues `tt-smi -r all` at each. `-r all`
# rather than `-r 0`: per tt-smi's CLI, -r takes UMD logical IDs and "omit targets or use
# 'all' to reset all devices", so 0 is not necessarily the board under test.
#
# Distribution is stock pytest-split applied per file, so the shards cover every file
# exactly once between them. Verified test-for-test identical to the normal structure on a
# common base: 40,653 collected / 2,123 skipped / 38,530 executed in both. No custom
# packer; the file list is discovered at runtime, so new suites are picked up automatically.
PERF_RESET="${PERF_RESET:-1}"

echo "[experiment] tt-smi board inventory:"
tt-smi -ls 2>&1 | sed 's/^/[experiment]   /' || echo "[experiment]   WARNING: tt-smi -ls failed"
ls -la /dev/tenstorrent/ 2>&1 | sed 's/^/[experiment]   dev: /' || true

maybe_reset() {
    [ "$PERF_RESET" = "1" ] || return 0
    echo "[experiment] tt-smi -r all before $1"
    tt-smi -r all || echo "[experiment] WARNING: tt-smi reset failed; continuing"
}

PERF_FILES=$(pytest -q --collect-only -m "perf and not accuracy" . 2>/dev/null \
  | grep '::' | sed 's/::.*//' | sort -u)
if [ -z "$PERF_FILES" ]; then echo "[experiment] ERROR: collected no perf files" >&2; exit 1; fi
echo "[experiment] suites ($(echo "$PERF_FILES" | wc -l | tr -d ' ')):"
echo "$PERF_FILES" | sed 's/^/[experiment]   /'

run_pytest() {
    local rc=0
    pytest "$@" || rc=$?
    [ "$rc" -eq 5 ] && return 0
    return "$rc"
}

for pf in $PERF_FILES; do
    tag=$(basename "$pf" .py)
    maybe_reset "$tag (compile)"
    run_pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 \
      -m "perf and not accuracy" --timeout=60 \
      --splits "$N_GROUPS" --group "$GROUP" \
      --junitxml="pytest-report-wormhole-${GROUP}-${tag}-compile.xml" "$pf"
    maybe_reset "$tag (measure)"
    run_pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer --dist loadgroup -n 15 -x \
      -m "perf and not accuracy" --timeout=60 \
      --splits "$N_GROUPS" --group "$GROUP" \
      --junitxml="pytest-report-wormhole-${GROUP}-${tag}-run.xml" "$pf"
done

junitparser merge pytest-report-wormhole-${GROUP}-*-compile.xml pytest-report-wormhole-${GROUP}-*-run.xml pytest-report-wormhole-${GROUP}.xml
