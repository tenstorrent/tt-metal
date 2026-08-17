#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Advertised-context (262144-token) evidence runs, one pytest process per case so each of the
# largest allocations in the stage starts from empty device DRAM.
set -uo pipefail
cd "${TT_METAL_HOME:?}"
L=models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/logs
T=models/autoports/qwen_qwen3_6_35b_a3b/tests/test_long_context.py
mkdir -p "$L"
# one run == one evidence file; the five cases below accumulate into it
rm -f models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/long_context.jsonl
for sel in "longest_decode_context and linear" \
           "longest_prefill and linear" \
           "longest_decode_context and full" \
           "longest_prefill and full" \
           "max_batch_full_context_capacity"; do
  name=$(echo "$sel" | tr ' ' '_')
  timeout 21600 python -m pytest "$T" -q --no-header -p no:cacheprovider -m slow -k "$sel" \
    > "$L/long_${name}.log" 2>&1
  rc=$?
  echo "LONGCASE sel='${sel}' exit=${rc} summary=$(grep -oE '[0-9]+ (passed|failed)[^,]*' "$L/long_${name}.log" | head -1)"
done
echo "LONG_RUN_DONE"
