#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# Standard pytest-split sharding: compile this shard's items (producer), then
# measure them (consumer) -- one invocation each over the whole perf suite,
# identical in shape to run_llk_perf_wormhole.sh.
#
# This reverts the slice-level bin-pack + per-slice `tt-smi -r 0` reset from
# #46478 (a workaround for #44910). That instability was root-caused to a leaked
# DBG_FEATURE_DISABLE bit-11 write from the unpacker and fixed at the source in
# #45727, so the reset is redundant; the per-slice test-by-test execution it
# required is dropped too (without the reset it also triggered TENSIX timeouts).
# Blackhole now runs the same shape as Wormhole.
#
# Usage: run_llk_perf_blackhole.sh <group> <n_groups>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

pytest $PYTEST_COMPILE_EXTRA --speed-of-light --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-blackhole-${GROUP}-compile.xml" .
pytest $PYTEST_RUN_EXTRA --speed-of-light --compile-consumer -n 15 -x -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-blackhole-${GROUP}-run.xml" .
junitparser merge pytest-report-blackhole-${GROUP}-compile.xml pytest-report-blackhole-${GROUP}-run.xml pytest-report-blackhole-${GROUP}.xml
