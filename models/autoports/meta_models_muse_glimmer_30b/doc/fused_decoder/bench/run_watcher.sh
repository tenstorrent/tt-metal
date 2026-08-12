#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Watcher-instrumented correctness run over 18 fused node ids covering every
# structurally distinct path: both kinds' multi-chunk prefill, decode,
# continuation prefill (including the sub-window tail), traced replay, batch 13
# (fallback head-concat) and batch 32, the non-zero cache slot, the awkward
# page-count multi-chunk prefill (which is the only case that exercises the
# *halved* paged-SDPA chunk), the graph audit, the norm-config shapes, the
# fused-vs-unfused comparison, and the 64-step stress soak.
#
# Must be a separate run from any profiling ($tt-device-usage: never combine
# TT_METAL_WATCHER with Tracy / the device profiler).
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/fused_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests/test_fused_decoder.py
rm -rf "$D/watcher"; mkdir -p "$D/watcher"

export TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1
export TT_METAL_LOGS_PATH=$D/watcher
python -m pytest \
  "$T::test_prefill_pcc[12345-sliding]" "$T::test_prefill_pcc[12345-full]" \
  "$T::test_decode_pcc[3000-sliding]" "$T::test_decode_pcc[3000-full]" \
  "$T::test_continuation_prefill_pcc[64-100-sliding]" \
  "$T::test_continuation_prefill_pcc[4096-3000-sliding]" "$T::test_continuation_prefill_pcc[4096-3000-full]" \
  "$T::test_traced_decode_advances_positions[sliding]" "$T::test_traced_decode_advances_positions[full]" \
  "$T::test_batched_prefill_decode_pcc[13-sliding]" "$T::test_batched_prefill_decode_pcc[32-full]" \
  "$T::test_multi_chunk_prefill_nonzero_user[sliding]" \
  "$T::test_multi_chunk_prefill_page_table_bound[full]" \
  "$T::test_prefill_seq_len_equals_max_and_chunk[sliding]" \
  "$T::test_fused_graph_uses_fused_ops[12345-sliding]" "$T::test_decode_norm_config_shapes[sliding]" \
  "$T::test_fused_vs_functional_equivalence[12345-full]" \
  "$T::test_repeated_run_stress[sliding]" \
  -q --no-header

# The repo-root .gitignore excludes any path component named "generated", and the
# check-large-files hook rejects anything over 500 KB.
mv "$D/watcher/generated/watcher/watcher.log" "$D/watcher/generated/watcher/kernel_names.txt" "$D/watcher/"
rm -rf "$D/watcher/generated"
for pat in "Watcher detected" tripped sanitize TT_ASSERT DEBUG_ASSERT "out of bounds" fault Error; do
  printf '%-18s %s\n' "$pat" "$(grep -ci "$pat" "$D/watcher/watcher.log")"
done
printf 'lines %s, dumps %s\n' "$(wc -l < "$D/watcher/watcher.log")" "$(grep -c Dump "$D/watcher/watcher.log")"
gzip -9 -f "$D/watcher/watcher.log" "$D/watcher/kernel_names.txt"
