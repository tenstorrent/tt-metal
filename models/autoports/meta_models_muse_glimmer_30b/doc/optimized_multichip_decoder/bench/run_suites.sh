#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The acceptance gate for the optimized multichip decoder: the multichip stage's
# own two correctness modules, unchanged, run against the optimized default path.
# The correctness floor does not move when the layer gets faster ($optimize:
# "your initial functional test suite remains the correctness floor").
#
# Two pytest invocations, for the reason the multichip stage documented:
# test_multichip_decoder.py holds a session-scoped 1x4 mesh, and
# test_multichip_vs_single_chip.py opens a 1x1 mesh first; run together the 1x1
# open finds the four dies still owned and times out on an Ethernet core.
set -uo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/optimized_multichip_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests
mkdir -p "$D/logs"

# ``|| true`` keeps the second module and the figure check running when the first
# module fails, but the exit codes are captured and this script exits non-zero on
# any of them -- it is the acceptance gate, so it has to be able to fail.
python -m pytest "$T/test_multichip_decoder.py" -q --no-header -p no:randomly \
  --junitxml="$D/test_results.xml" > "$D/logs/full_test_run.log" 2>&1 || true
main_exit=$(grep -acE "^(FAILED|ERROR)" "$D/logs/full_test_run.log" || true)
grep -aoE "[0-9]+ (passed|failed)[^|]*" "$D/logs/full_test_run.log" | tail -1

python -m pytest "$T/test_multichip_vs_single_chip.py" -q --no-header -p no:randomly \
  --junitxml="$D/test_results_vs_single_chip.xml" > "$D/logs/vs_single_chip_run.log" 2>&1 || true
vs_exit=$(grep -acE "^(FAILED|ERROR)" "$D/logs/vs_single_chip_run.log" || true)
grep -aoE "[0-9]+ (passed|failed)[^|]*" "$D/logs/vs_single_chip_run.log" | tail -1

# The multichip-vs-single-chip worst values are the ones that can see a
# parallelisation or scheduling fault; print them, not just the pass count.
grep -haoE "worst\[[^]]*\]: [0-9.]+ on [a-z0-9]+" "$D/logs/vs_single_chip_run.log" || true

# Every mechanically-sourced figure in README.md, work_log.md and
# context_contract.json, re-derived from the committed CSVs and logs.  Three
# rounds of $stage-review found the same class of defect -- a number from a
# superseded run -- so it is now a gate rather than a habit.
python "$D/bench/check_reported_figures.py"
figures_exit=$?

if [ "$main_exit" != "0" ] || [ "$vs_exit" != "0" ] || [ "$figures_exit" != "0" ]; then
  echo "GATE FAILED: ${main_exit} failing test(s) in the main module, ${vs_exit} in the comparison module, figure check exit ${figures_exit}"
  exit 1
fi
echo "GATE PASSED"
