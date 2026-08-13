#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The acceptance gate: both correctness modules, as **two** pytest invocations.
#
# They cannot share a session.  test_multichip_decoder.py holds a session-scoped
# 1x4 mesh (session fixtures tear down last), and test_multichip_vs_single_chip.py
# opens a 1x1 mesh and then a 1x4 one; run together, the 1x1 open finds the four
# dies still owned and times out on an Ethernet core, which costs a tt-smi -r.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/multichip_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests

python -m pytest "$T/test_multichip_decoder.py" -q --no-header -p no:randomly \
  --junitxml="$D/test_results.xml" > "$D/logs/full_test_run.log" 2>&1 || true
tail -1 "$D/logs/full_test_run.log"

python -m pytest "$T/test_multichip_vs_single_chip.py" -q --no-header -p no:randomly \
  --junitxml="$D/test_results_vs_single_chip.xml" > "$D/logs/vs_single_chip_run.log" 2>&1 || true
tail -1 "$D/logs/vs_single_chip_run.log"

grep -hE "=+ [0-9]+ (passed|failed)" "$D/logs/full_test_run.log" "$D/logs/vs_single_chip_run.log"
python "$D/bench/summarize_pcc.py"
python "$D/bench/refresh_context_contract.py"
