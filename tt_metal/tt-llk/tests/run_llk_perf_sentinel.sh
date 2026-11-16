#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Compile the sentinel tests of perf_stability_sentinels.json once and measure them three times:
# twice in collection order (a, b) and once in another fixed order (c). Write the comparisons to <out dir>.
# Usage: CHIP_ARCH=<arch> run_llk_perf_sentinel.sh <out dir>
set -euo pipefail

OUT="$(realpath -m "${1:?usage: run_llk_perf_sentinel.sh <out dir>}")"
LLK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"
python3 "$LLK/perf/stability_check.py" nodes --arch "${CHIP_ARCH:?}" --out "$OUT"

cd "$LLK/tests/python_tests"
# Its own build tree, on the runner's disk: a compile session clears the tree it builds in.
RUNNER_TEMP="$(mktemp -d -p "${RUNNER_TEMP:-/tmp}")"
export RUNNER_TEMP
TAG="${PERF_RUN_TAG:-local}-sentinel"
mapfile -t FILES < <(cut -d: -f1 "$OUT/nodes.txt" | sort -u)
PYTEST=(pytest -q --override-ini=log_cli=false -m "perf and not accuracy" --timeout=60 --node-list="$OUT/nodes.txt")

PERF_RUN_TAG="$TAG-compile" "${PYTEST[@]}" --compile-producer -n 10 "${FILES[@]}" || true
for pass in a b c; do
  ORDER=()
  [[ $pass == c ]] && ORDER=(--test-order-file "$OUT/order.json" --rewind-runner sentinel)
  PERF_RUN_TAG="$TAG-$pass" "${PYTEST[@]}" --compile-consumer --dist loadgroup -n 15 "${ORDER[@]}" \
    --junitxml="$OUT/junit-$pass.xml" "${FILES[@]}" || true
done

python3 "$LLK/perf/stability_check.py" compare "$LLK/perf_data/runs/$TAG-a" "$LLK/perf_data/runs/$TAG-b" --out "$OUT/repeat.json"
python3 "$LLK/perf/stability_check.py" compare "$LLK/perf_data/runs/$TAG-a" "$LLK/perf_data/runs/$TAG-c" --out "$OUT/reorder.json"
