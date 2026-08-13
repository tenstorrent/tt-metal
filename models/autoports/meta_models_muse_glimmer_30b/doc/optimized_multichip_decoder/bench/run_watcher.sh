#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Watcher-instrumented correctness run for the *optimized* multichip decoder:
# the multichip stage's node id list plus this stage's two new contract tests,
# so the async CCL primitives, the shared L1_SMALL global semaphores and the
# sharded layer boundary are all watched.  Async collectives are exactly the
# thing $optimize says to always run under watcher ("it's easy to make mistakes
# that end up in data corruption or hangs").
#
# The list covers every structurally distinct multichip path: both kinds' multi-chunk prefill and
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
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/optimized_multichip_decoder
# ``TAG`` names this run's artifacts, so a deliberately-unsafe arm cannot
# overwrite the clean one.  ``rm -rf "$D/watcher"`` below is why that matters.
TAG=${WATCHER_TAG:-}
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_multichip_decoder.py
rm -rf "$D/watcher$TAG"; mkdir -p "$D/watcher$TAG"

export TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1
export TT_METAL_LOGS_PATH=$D/watcher$TAG
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
  "$T::test_collective_implementation_is_split_by_payload[sliding]" \
  "$T::test_collective_implementation_is_split_by_payload[full]" \
  "$T::test_decode_boundary_layout_is_a_fixed_point[sliding]" \
  "$T::test_decode_boundary_layout_is_a_fixed_point[full]" \
  -q --no-header -p no:randomly -rA 2>&1 | tee "$D/logs/watcher_pytest$TAG.log"
watcher_exit=${PIPESTATUS[0]}
echo "pytest exit code: $watcher_exit (0 = clean; 134 = SIGABRT, see the teardown note above)"

# The comparison module is deliberately *not* watched here.  It opens a 1x1 mesh,
# and opening one shortly after a FABRIC_1D_RING 1x4 mesh has closed intermittently
# times out on an Ethernet core ("Timed out while waiting for active ethernet core
# 29-25 to become active again"), which costs a tt-smi -r.  It runs on its own in
# the acceptance gate; every op it dispatches is already covered above by the same
# layer on the same mesh.

# The repo-root .gitignore excludes any path component named "generated", and the
# check-large-files hook rejects anything over 500 KB.
if [ -f "$D/watcher$TAG/generated/watcher/watcher.log" ]; then
  mv "$D/watcher$TAG/generated/watcher/watcher.log" "$D/watcher$TAG/generated/watcher/kernel_names.txt" "$D/watcher$TAG/"
  rm -rf "$D/watcher$TAG/generated"
else
  echo "NO WATCHER DUMP -- the run did not reach the layer; this is not evidence"
fi
# Grep the **console** as well as the dump.  A tripped assert is reported by the
# watcher *server*, on stderr -- "Watcher stopped the device due to tripped
# assert" / "Watcher detected tripped assert and stopped device" -- and does not
# necessarily appear in watcher.log at all.  Counting only the dump file reports
# a clean run when the device was in fact stopped; this stage hit exactly that.
CONSOLE=${WATCHER_CONSOLE_LOG:-$D/logs/watcher_pytest$TAG.log}
for pat in "Watcher detected" tripped sanitize TT_ASSERT DEBUG_ASSERT "out of bounds" fault Error; do
  dump=$(grep -ci "$pat" "$D/watcher$TAG/watcher.log" 2>/dev/null || true)
  console=0
  [ -n "$CONSOLE" ] && [ -f "$CONSOLE" ] && console=$(grep -aci "$pat" "$CONSOLE" || true)
  printf '%-18s dump=%-6s console=%s\n' "$pat" "$dump" "$console"
done
# Each dump writes a "Dump #N" line and a "Dump #N completed" line, so the raw
# grep count is double the number of dumps.  Review round 4 caught the README
# quoting the doubled figure.
printf 'lines %s, dumps %s\n' "$(wc -l < "$D/watcher$TAG/watcher.log" 2>/dev/null || echo 0)" \
  "$(grep -c 'Dump #[0-9]* completed' "$D/watcher$TAG/watcher.log" 2>/dev/null || true)"
# A run that never reached the layer is not a clean run.  The multichip stage's
# teardown fault wedges the *next* process, so this guard is what stops an
# aborted-at-startup run from being read as evidence.
# pytest colours its short-summary lines, so "PASSED" is preceded by an ANSI
# escape and an anchored grep matches nothing -- which is the *same* class of
# mistake this guard exists to catch, and it reported 0 for a fully passing run
# until it was fixed.  Parse the summary line instead, and print it verbatim.
printf 'tests reported: %s\n' \
  "$(sed 's/\x1b\[[0-9;]*m//g' "$CONSOLE" 2>/dev/null | grep -aoE '[0-9]+ (passed|failed)' | paste -sd' ' || echo NONE)"
[ -f "$D/watcher$TAG/watcher.log" ] && gzip -9 -f "$D/watcher$TAG/watcher.log" "$D/watcher$TAG/kernel_names.txt"
echo "WATCHER_EXIT=$watcher_exit"
exit "$watcher_exit"
