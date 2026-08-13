#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Watcher-instrumented correctness run over the node ids that cover every
# structurally distinct multichip path: both kinds' multi-chunk prefill and
# decode, the one-tile and sub-tile prefill branches, continuation prefill with a
# sub-window sliding tail (the per-device single-KV-head hand-off), batch 13 (no
# batch-core rectangle) and batch 32 (the op's ceiling), the non-zero cache slot,
# traced replay, the 64-step soak, the real-weight decode off the BFP8 cache, the
# graph audits that assert the DRAM-sharded dispatches and the two reductions,
# and the BFP8-payload and reduce-scatter/all-reduce overrides.  The sequential
# 1x1-then-1x4 comparison module is *not* in this run; the note further down
# explains why.
#
# Must be a separate run from any profiling ($tt-device-usage: never combine
# TT_METAL_WATCHER with Tracy / the device profiler).
# No ``-e``: the pytest below is allowed to fail *at teardown* without losing the
# artifact.  A 1x4 FABRIC_1D_RING mesh intermittently times out returning an
# ethernet core to base firmware when the process exits
# ("Timed out while waiting for active ethernet core 29-25 to become active
# again"), which aborts the interpreter after every test has already reported.
# The exit code is captured and printed rather than swallowed.
set -uo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/multichip_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_multichip_decoder.py
rm -rf "$D/watcher"; mkdir -p "$D/watcher"

export TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1
export TT_METAL_LOGS_PATH=$D/watcher
python -m pytest \
  "$T::test_prefill_pcc[12345-sliding]" "$T::test_prefill_pcc[12345-full]" \
  "$T::test_prefill_pcc[1-sliding]" "$T::test_prefill_pcc[100-full]" \
  "$T::test_prefill_pcc[128-sliding]" "$T::test_prefill_pcc[8193-full]" \
  "$T::test_decode_pcc[3000-sliding]" "$T::test_decode_pcc[3000-full]" \
  "$T::test_continuation_prefill[64-100-sliding]" \
  "$T::test_continuation_prefill[1024-1024-sliding]" \
  "$T::test_continuation_prefill[4096-3000-full]" \
  "$T::test_batched_prefill_decode_pcc[13-sliding]" \
  "$T::test_batched_prefill_decode_pcc[32-full]" \
  "$T::test_multi_chunk_prefill_nonzero_user[sliding]" \
  "$T::test_traced_decode_pcc[sliding]" "$T::test_traced_decode_pcc[full]" \
  "$T::test_decode_soak[sliding]" "$T::test_decode_soak[full]" \
  "$T::test_real_weights_decode_pcc[sliding]" \
  "$T::test_real_weights_traced_decode_and_batch[full]" \
  "$T::test_decode_uses_dram_sharded_matmuls[sliding]" \
  "$T::test_decode_has_exactly_two_collectives[sliding]" \
  "$T::test_decode_has_exactly_two_collectives[full]" \
  "$T::test_qkv_output_shard_is_padded_not_wrong[sliding]" \
  "$T::test_kv_cache_holds_the_expected_head[sliding]" \
  "$T::test_kv_cache_holds_the_expected_head[full]" \
  "$T::test_replicas_are_bit_identical[sliding]" \
  "$T::test_decode_ccl_dtype_override[sliding]" \
  "$T::test_ccl_mode_override[all_reduce]" "$T::test_ccl_mode_override[rs_ag]" \
  "$T::test_determinism[full]" \
  -q --no-header -p no:randomly
watcher_exit=$?
echo "pytest exit code: $watcher_exit (0 = clean; 134 = SIGABRT, see the teardown note above)"

# The comparison module is deliberately *not* watched here.  It opens a 1x1 mesh,
# and opening one shortly after a FABRIC_1D_RING 1x4 mesh has closed intermittently
# times out on an Ethernet core ("Timed out while waiting for active ethernet core
# 29-25 to become active again"), which costs a tt-smi -r.  It runs on its own in
# the acceptance gate; every op it dispatches is already covered above by the same
# layer on the same mesh.

# The repo-root .gitignore excludes any path component named "generated", and the
# check-large-files hook rejects anything over 500 KB.
mv "$D/watcher/generated/watcher/watcher.log" "$D/watcher/generated/watcher/kernel_names.txt" "$D/watcher/"
rm -rf "$D/watcher/generated"
for pat in "Watcher detected" tripped sanitize TT_ASSERT DEBUG_ASSERT "out of bounds" fault Error; do
  printf '%-18s %s\n' "$pat" "$(grep -ci "$pat" "$D/watcher/watcher.log" || true)"
done
# Each dump writes a "Dump #N" line and a "Dump #N completed" line, so the raw
# grep count is double the number of dumps.  Review round 4 caught the README
# quoting the doubled figure.
printf 'lines %s, dumps %s\n' "$(wc -l < "$D/watcher/watcher.log")" \
  "$(grep -c 'Dump #[0-9]* completed' "$D/watcher/watcher.log" || true)"
gzip -9 -f "$D/watcher/watcher.log" "$D/watcher/kernel_names.txt"
echo "WATCHER_EXIT=$watcher_exit"
exit "$watcher_exit"
