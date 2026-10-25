#!/usr/bin/env bash
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/fused_decoder
runner=models/autoports/qwen_qwen3_8_27b/tests/fused_run.sh
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
for layer in 0 3; do
  label=watcher_layer${layer}
  TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=/workspace/$base/$label \
    bash "$runner" check "$label" --layer "$layer" --seq 385 --requests --resume --remap --compare-functional > "$base/$label.log" 2>&1
  echo "PASS $label"
done
for layer in 0 3; do
  for seq in 262142 262144; do
    label=final_layer${layer}_${seq}
    extra=()
    if [[ $seq == 262144 ]]; then extra+=(--prefill-only); fi
    bash "$runner" check "$label" --layer "$layer" --seq "$seq" "${extra[@]}" > "$base/$label.log" 2>&1
    echo "PASS $label"
  done
done
for batch in 32 8 16; do
  for layer in 0 3; do
    label=workload_layer${layer}_b${batch}
    bash "$runner" check "$label" --layer "$layer" --seq 4096 --batch "$batch" --workload --timing > "$base/$label.log" 2>&1
    echo "PASS $label"
  done
done
bash "$runner" check odd_batch --layer 0 --seq 129 --batch 3 --resume > "$base/odd_batch.log" 2>&1
python -m models.autoports.qwen_qwen3_8_27b.tests.ragged_fused_decode > "$base/ragged_decode.log" 2>&1
python -m pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_fused_decoder.py --basetemp "$base/synthetic_cases" --junitxml "$base/synthetic_pytest.xml"
