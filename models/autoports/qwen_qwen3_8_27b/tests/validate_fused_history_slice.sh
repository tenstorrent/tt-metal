#!/usr/bin/env bash
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/fused_decoder
runner=models/autoports/qwen_qwen3_8_27b/tests/fused_run.sh
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=/workspace/$base/watcher_linear_history_slice \
  bash "$runner" check watcher_linear_history_slice --layer 0 --seq 385 --requests --resume --compare-functional > "$base/watcher_linear_history_slice.log" 2>&1
echo 'PASS history-slice watcher'
for seq in 262142 262144; do
  extra=()
  if [[ $seq == 262144 ]]; then extra+=(--prefill-only); fi
  bash "$runner" check history_slice_${seq} --layer 0 --seq "$seq" "${extra[@]}" > "$base/history_slice_${seq}.log" 2>&1
  echo "PASS history-slice $seq"
done
for batch in 32 8 16; do
  bash "$runner" check history_slice_workload_b${batch} --layer 0 --seq 4096 --batch "$batch" --workload --timing > "$base/history_slice_workload_b${batch}.log" 2>&1
  echo "PASS history-slice batch $batch"
done
bash "$runner" check history_slice_odd_batch --layer 0 --seq 129 --batch 3 --resume > "$base/history_slice_odd_batch.log" 2>&1
python -m pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_fused_decoder.py -k 0 --basetemp "$base/synthetic_linear_history_slice" --junitxml "$base/synthetic_linear_history_slice.xml"
