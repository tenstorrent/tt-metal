#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run inside the source-built environment at the repository root. Serial device use.
set -euo pipefail
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
artifact_dir=models/autoports/qwen_qwen3_8_27b/doc/functional_decoder
runner=models.autoports.qwen_qwen3_8_27b.tests.run_decoder
python -m "$runner" --layer 0 --batch 32 --seq 129 --resume --output "$artifact_dir/linear_b32_final.json"
python -m "$runner" --layer 3 --batch 32 --seq 129 --resume --remap --output "$artifact_dir/full_b32_final.json"
python -m "$runner" --layer 0 --batch 8 --seq 129 --resume --output "$artifact_dir/linear_b8_final.json"
python -m "$runner" --layer 0 --batch 16 --seq 129 --resume --output "$artifact_dir/linear_b16_final.json"
python -m "$runner" --layer 3 --batch 16 --seq 129 --resume --remap --output "$artifact_dir/full_b16_final.json"
python -m "$runner" --layer 3 --seq 4096 --resume --remap --output "$artifact_dir/full_b1_fp64.json"
python -m models.autoports.qwen_qwen3_8_27b.tests.ragged_decode
python -m pytest -q --confcutdir=models/autoports/qwen_qwen3_8_27b/tests \
  models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py \
  --basetemp="$artifact_dir/synthetic_cases" \
  --junitxml="$artifact_dir/synthetic_pytest.xml"
