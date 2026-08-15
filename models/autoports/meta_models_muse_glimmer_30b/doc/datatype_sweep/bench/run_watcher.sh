#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Watcher-instrumented correctness run on the **selected** precision config.
#
# Every device test here builds through ``build_generator``, which reads
# ``doc/datatype_sweep/selected_precision_config.json``, so this run watches the
# policy the stage selected rather than a policy a harness argument installed.
#
# Why watcher for a datatype stage specifically: a precision change re-picks the
# matmul geometry tables (they are keyed by ``(role, weight dtype)``, and the L1
# circular-buffer budget is dtype-scaled), changes the packed size of every
# weight shard, and changes the payload width of the two decode collectives.
# That is the class of change that surfaces as an out-of-bounds NOC write or a
# stale L1 read rather than as a wrong number.
#
# The default case list is the optimized-full-model stage's gated ten, so the
# artifact is directly comparable with that stage's ``WATCHER_CLEAN``.  The two
# ``prefill_trace`` cases stay out, for the reason that stage documents (README
# limitation 6: a twelve-case process including them trips a teardown-time fabric
# fault that truncates the watcher log and would invalidate this artifact).
#
# This stage's four precision device cases are watched too, but in **their own
# process** (``WATCHER_TAG=_precision``), because ``test_full_model.py`` and
# ``test_precision_config.py`` each own a module-scoped ``mesh`` fixture and the
# second one to run cannot open a mesh the first still holds.  That is a test
# harness property, not a model one, and splitting the runs is the fix rather
# than reworking two suites' fixtures for a watcher pass.
#
# Must be a separate run from any profiling ($tt-device-usage: never combine
# TT_METAL_WATCHER with Tracy / the device profiler).
#
# No ``-e``: as in the three previous stages, a 1x4 FABRIC_1D_RING mesh
# intermittently times out returning an ethernet core to base firmware at process
# exit, which aborts the interpreter *after* every test has reported.  The exit
# code is captured and printed rather than swallowed.
#
# Usage:  bash doc/datatype_sweep/bench/run_watcher.sh                 # the gated ten
#         WATCHER_TAG=_precision bash doc/datatype_sweep/bench/run_watcher.sh   # the precision cases
# Watch:  tail -f models/autoports/meta_models_muse_glimmer_30b/doc/datatype_sweep/logs/watcher_pytest.log
set -uo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
M=models/autoports/meta_models_muse_glimmer_30b
D=$PWD/$M/doc/datatype_sweep
TAG=${WATCHER_TAG:-}
FULL=$M/tests/test_full_model.py
PREC=$M/tests/test_precision_config.py
rm -rf "$D/watcher$TAG"; mkdir -p "$D/watcher$TAG" "$D/logs"

PRECISION_NODES=(
  "$PREC::test_the_shipped_artifact_is_the_policy_the_build_runs"
  "$PREC::test_the_readiness_factory_path_builds_the_selected_config"
  "$PREC::test_a_layer_exception_reaches_only_the_layers_it_names"
  "$PREC::test_a_per_role_fidelity_request_reaches_only_that_role"
)
NODES=(
  "$FULL::test_prefill_is_reproducible[1024]"
  "$FULL::test_split_sampling_feeds_the_sampled_token_back_on_device"
  "$FULL::test_steady_state_decode_does_no_per_token_host_work"
  "$FULL::test_topk_runs_through_the_multi_core_factory"
  "$FULL::test_device_sampling_keeps_each_batch_row_token_in_its_own_row"
  "$FULL::test_lm_head_softcap_runs_in_l1_and_matches_the_dram_form"
  "$FULL::test_decode_embedding_gathers_straight_into_the_boundary_layout[7]"
  "$FULL::test_decode_embedding_gathers_straight_into_the_boundary_layout[202047]"
  "$FULL::test_swiglu_multiply_runs_on_the_wide_grid_and_returns_the_narrow_one"
  "$FULL::test_batched_prefill_and_decode_with_mixed_lengths[32]"
)
if [ "$TAG" = "_precision" ]; then NODES=("${PRECISION_NODES[@]}"); fi
if [ "$#" -gt 0 ]; then NODES=("$@"); fi

export TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1
export TT_METAL_LOGS_PATH=$D/watcher$TAG
python -m pytest "${NODES[@]}" \
  -q --no-header -p no:randomly -rA 2>&1 | tee "$D/logs/watcher_pytest$TAG.log"
watcher_exit=${PIPESTATUS[0]}
echo "pytest exit code: $watcher_exit (0 = clean; 134 = SIGABRT at teardown, see the note above)"

# ``TT_METAL_LOGS_PATH`` writes to ``$LOGS/generated/watcher/watcher.log`` alongside
# megabytes of inspector/kernel-name build metadata that is not stage evidence.
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

# A tripped assert aborts the process *inside* the watcher's own dump, leaving the log
# truncated, so the console is checked directly rather than inferred from the log.
echo "=== console check for a tripped assert ==="
if grep -q "tripped assert\|subordinate_erisc detected invalid NOC" "$D/logs/watcher_pytest$TAG.log"; then
  echo "WATCHER_CONSOLE_TRIPPED_ASSERT"
  grep -h "subordinate_erisc detected invalid NOC\|tripped assert" "$D/logs/watcher_pytest$TAG.log" | head -3
else
  echo "WATCHER_CONSOLE_NO_TRIPPED_ASSERT"
fi | tee "$D/logs/check_watcher_console$TAG.log"

echo "=== watcher verdict, re-derived from the log ==="
python $M/doc/functional_decoder/bench/check_watcher.py \
  "$D/watcher$TAG/watcher.log.gz" | tee "$D/logs/check_watcher$TAG.log"
