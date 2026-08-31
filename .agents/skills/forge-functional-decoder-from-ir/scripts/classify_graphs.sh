#!/usr/bin/env bash
#
# classify_graphs.sh — classify every TTNN IR (.mlir) under a directory by the
# workload it captures, so you translate the right graph for each forward path.
#
# Classification is workload-derived (not op-name matching): the same rule the
# forge skill uses. Signals reported per graph:
#   * fill_cache          > 0  -> prefill (fills whole KV cache over a full seq)
#   * paged_update_cache / update_cache / sdpa_decode > 0, and NO fill_cache,
#     with a single-token step -> decode (appends one token to a passed-in cache)
#   * has_full_vocab_logits    -> graph also RETURNS raw logits (…x{vocab}xbf16)
#                                 (a logits-returning variant of the same path)
#   * is_runtime               -> a *runtime* dump (ttnn_runtime_…); prefer the
#                                 non-runtime compiler graph for translation
#
# Pick, per model: one prefill graph and one decode graph (compiler, non-runtime).
# Whether it also returns logits does not change the layer math you translate.
#
# Usage: classify_graphs.sh <dir-with-mlir-files>
#
set -euo pipefail
DIR="${1:?usage: classify_graphs.sh <dir>}"

printf "%-70s | %-8s | %8s %10s %8s | %s | %s\n" \
  "file" "ROLE" "fillc" "pagedupd" "sdpaDec" "logits" "runtime"
printf '%.0s-' {1..120}; echo

while IFS= read -r f; do
  b="$(basename "$f")"
  fc=$(grep -c 'ttnn.fill_cache' "$f" 2>/dev/null || true); fc=${fc:-0}
  pu=$(grep -c 'ttnn.paged_update_cache' "$f" 2>/dev/null || true); pu=${pu:-0}
  uc=$(grep -c 'ttnn.update_cache' "$f" 2>/dev/null || true); uc=${uc:-0}
  sd=$(grep -c 'scaled_dot_product_attention_decode' "$f" 2>/dev/null || true); sd=${sd:-0}
  # full-vocab logits returned by @main (large last dim bf16, seq dim small)
  lg=$(grep -E 'func.func @main\(' "$f" | head -1 | grep -oE 'tensor<[0-9]+x[0-9]+x1[0-9]{4,6}xbf16' | head -1 || true)
  rt="no"; case "$b" in ttnn_runtime_*|*runtime*) rt="yes";; esac
  role="?"
  if [ "$fc" -gt 0 ]; then role="prefill"
  elif [ $((pu+uc+sd)) -gt 0 ]; then role="decode"; fi
  printf "%-70s | %-8s | %8s %10s %8s | %-6s | %s\n" \
    "$b" "$role" "$fc" "$pu" "$sd" "${lg:+yes}" "$rt"
done < <(find "$DIR" -name '*.mlir' | sort)
