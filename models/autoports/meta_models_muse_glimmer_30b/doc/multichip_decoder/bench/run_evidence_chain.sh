#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Every device job this stage's evidence needs, in order, one at a time
# ($tt-device-usage). Watcher and profiler are in separate runs and never
# overlap. Correctness (``run_suites.sh``) is deliberately *not* in here: it is
# the acceptance gate and is run on its own.
#
#   bash bench/run_evidence_chain.sh
#
# Produces:
#   logs/layer_ab_final.log              the shipped config, warmed e2e
#   logs/layer_ab_single_baseline.log    the single-chip baseline, same harness
#   tracy/{sliding,full}/*               eight signposted device-time windows
#   logs/watcher_run.log, watcher/       a watcher-clean correctness run
#   logs/real_weight_ccl_dtype_gate.log  the BFP8 decode payload against 0.995
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/multichip_decoder
mkdir -p "$D/logs"

echo "=== 1/5 warmed A/B, the shipped config (4 chips)"
python "$D/bench/layer_ab.py" --mesh 1x4 --candidates tp4 --prefill-seq 8192 --decode-context 2048 \
  > "$D/logs/layer_ab_final.log" 2>&1
grep -E '^AB|^FAILED' "$D/logs/layer_ab_final.log"

echo "=== 2/5 warmed A/B, the single-chip baseline (1 chip, no fabric, no L1_SMALL)"
python "$D/bench/layer_ab.py" --mesh 1x1 --candidates single --prefill-seq 8192 --decode-context 2048 \
  > "$D/logs/layer_ab_single_baseline.log" 2>&1
grep -E '^AB|^FAILED' "$D/logs/layer_ab_single_baseline.log"

echo "=== 3/5 device-time profiles (no watcher in this run)"
bash "$D/bench/run_tracy.sh" > "$D/logs/run_tracy_console.log" 2>&1
tail -3 "$D/logs/run_tracy_console.log"
# ~700 KB of Tracy chatter, over the repo's 500 KB file hook.
gzip -9 -f "$D/logs/run_tracy_console.log"

echo "=== 4/5 watcher (no profiler in this run)"
# The watcher script exits with pytest's code, and a 1x4 FABRIC_1D_RING mesh
# intermittently SIGABRTs at teardown *after* every test has reported (see the
# note in that script).  The artifact is still complete, so the chain records the
# code and carries on rather than stopping here.
bash "$D/bench/run_watcher.sh" > "$D/logs/watcher_run.log" 2>&1 || true
tail -14 "$D/logs/watcher_run.log"

echo "=== 5/5 the BFP8 decode collective payload against the 0.995 real-weight bar"
python "$D/bench/ccl_dtype_gate.py" > "$D/logs/real_weight_ccl_dtype_gate.log" 2>&1
grep -E '^GATE-WORST' "$D/logs/real_weight_ccl_dtype_gate.log"

echo "=== chain done ==="
