#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The whole behaviour-carrying evidence chain, in the order the work log's
# "Artifact freshness" section claims, so that ordering is a property of a
# committed script rather than of how the commands happened to be typed:
#
#   variant sweep -> multi-chunk A/B -> pytest suite (+ junit XML)
#     -> PCC summary -> context contract -> ten Tracy captures -> watcher
#
# Every artifact it writes therefore postdates tt/fused_decoder.py and
# tests/test_fused_decoder.py, which is the check a reviewer can run.
#
# One device job at a time.  The watcher run is last and is a separate process
# from every Tracy run ($tt-device-usage: never combine TT_METAL_WATCHER with
# the device profiler).
#
# Runtime is roughly 45 minutes.  Watch it with:
#   tail -f models/autoports/meta_models_muse_glimmer_30b/doc/fused_decoder/logs/chain.log
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/fused_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_fused_decoder.py
B=$D/bench

step () { echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$D/logs/chain.log"; }

: > "$D/logs/chain.log"

step "variant sweep (wall-clock A/B of every candidate topology)"
python "$B/ab_latency.py" --decode-iters 128 \
    --impl functional,fused,packed_gate_up,swiglu,packed_qkv_gate,fused_kv_update \
    > "$D/logs/variant_sweep.log" 2>&1

step "multi-chunk (16384-token) prefill A/B against the functional decoder"
python "$B/ab_latency.py" --prefill-seq 16384 --tag mc16384 --impl functional,fused \
    > "$D/logs/multichunk_prefill_ab.log" 2>&1

step "full fused suite"
python -m pytest "$T" -q --no-header --junitxml="$D/test_results.xml" \
    > "$D/logs/full_test_run.log" 2>&1

step "PCC summary + context contract, both derived from the suite run"
python "$B/summarize_pcc.py"
python "$B/refresh_context_contract.py"

step "Tracy: 8 fused windows + 2 multi-chunk functional baselines"
bash "$B/run_tracy.sh" > "$D/logs/run_tracy_console.log" 2>&1

step "watcher (separate run, no profiler attached)"
bash "$B/run_watcher.sh" > "$D/logs/watcher_run.log" 2>&1

step "done"
grep -E "====.*(passed|failed)" "$D/logs/full_test_run.log" "$D/logs/watcher_run.log" | tee -a "$D/logs/chain.log"
grep -c "markers were dropped" "$D"/logs/tracy_*.log | tee -a "$D/logs/chain.log"
