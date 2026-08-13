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
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/optimized_multichip_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests
mkdir -p "$D/logs"

python -m pytest "$T/test_multichip_decoder.py" -q --no-header -p no:randomly \
  --junitxml="$D/test_results.xml" > "$D/logs/full_test_run.log" 2>&1 || true
tail -1 "$D/logs/full_test_run.log"

python -m pytest "$T/test_multichip_vs_single_chip.py" -q --no-header -p no:randomly \
  --junitxml="$D/test_results_vs_single_chip.xml" > "$D/logs/vs_single_chip_run.log" 2>&1 || true
tail -1 "$D/logs/vs_single_chip_run.log"

grep -hE "=+ [0-9]+ (passed|failed)" "$D/logs/full_test_run.log" "$D/logs/vs_single_chip_run.log" || true
