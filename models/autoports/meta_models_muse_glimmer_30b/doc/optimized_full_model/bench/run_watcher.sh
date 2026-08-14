#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Watcher-instrumented correctness run for the optimized full model: the
# full-model stage's six device cases plus this stage's new contract tests,
# so every op kind the optimization touches is watched.  The three changes are all
# layout changes -- an async all-gather writing width-sharded L1 instead of DRAM,
# two elementwise ops moved onto a *padded* width shard, and two reshards plus an
# SFPU multiply on an 80-core grid -- which is exactly the class of change that
# shows up as an out-of-bounds NOC write or a stale L1 read rather than as a wrong
# number, so watcher is the instrument for it.
#
# The two ``prefill_trace`` cases are **not** in this list and that is a finding rather
# than an omission.  Each is watcher-clean on its own
# (``logs/watcher_bisect_optin.log``, ``logs/watcher_bisect_rebind_fixed.log``), and so
# are all four isolated arms of ``bench/prefill_trace_release_probe.py``.  What trips a
# fabric ERISC assert on acteth core 29-25 is releasing a prefill trace and then building
# and running *another* model on the same mesh in the same process -- which this module's
# fixtures do and the shipped default never does, because with ``prefill_trace`` off no
# prefill trace is ever captured or released.  Run them with
# ``bash bench/run_watcher.sh`` plus the two single-case commands in the README.
#
# Must be a separate run from any profiling ($tt-device-usage: never combine
# TT_METAL_WATCHER with Tracy / the device profiler).
#
# No ``-e``: as in the two previous stages, a 1x4 FABRIC_1D_RING mesh
# intermittently times out returning an ethernet core to base firmware at process
# exit ("Timed out while waiting for active ethernet core 29-25 to become active
# again"), which aborts the interpreter *after* every test has reported.  The exit
# code is captured and printed rather than swallowed, and the artifact survives.
set -uo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/optimized_full_model
TAG=${WATCHER_TAG:-}
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_full_model.py
rm -rf "$D/watcher$TAG"; mkdir -p "$D/watcher$TAG" "$D/logs"

export TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1
export TT_METAL_LOGS_PATH=$D/watcher$TAG
python -m pytest \
  "$T::test_prefill_is_reproducible[1024]" \
  "$T::test_split_sampling_feeds_the_sampled_token_back_on_device" \
  "$T::test_steady_state_decode_does_no_per_token_host_work" \
  "$T::test_topk_runs_through_the_multi_core_factory" \
  "$T::test_device_sampling_keeps_each_batch_row_token_in_its_own_row" \
  "$T::test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form" \
  "$T::test_decode_embedding_gathers_straight_into_the_boundary_layout[7]" \
  "$T::test_decode_embedding_gathers_straight_into_the_boundary_layout[202047]" \
  "$T::test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one" \
  "$T::test_batched_prefill_and_decode_with_mixed_lengths[32]" \
  -q --no-header -p no:randomly -rA 2>&1 | tee "$D/logs/watcher_pytest$TAG.log"
watcher_exit=${PIPESTATUS[0]}
echo "pytest exit code: $watcher_exit (0 = clean; 134 = SIGABRT at teardown, see the note above)"

# ``TT_METAL_LOGS_PATH`` writes to ``$LOGS/generated/watcher/watcher.log``, alongside 6 MB
# of inspector/kernel-name build metadata that is not stage evidence.  Keep the log, at the
# top of the artifact directory where the README cites it, and drop the rest.
LOG=$D/watcher$TAG/generated/watcher/watcher.log
if [ -f "$LOG" ]; then
  gzip -9 -f "$LOG"
  tmp=$(mktemp -d)
  mv "$LOG.gz" "$tmp/watcher.log.gz"
  rm -rf "$D/watcher$TAG"
  mkdir -p "$D/watcher$TAG"
  mv "$tmp/watcher.log.gz" "$D/watcher$TAG/watcher.log.gz"
  rmdir "$tmp"
fi
echo "=== watcher verdict, re-derived from the log ==="
python models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/bench/check_watcher.py \
  "$D/watcher$TAG/watcher.log.gz" | tee "$D/logs/check_watcher$TAG.log"
