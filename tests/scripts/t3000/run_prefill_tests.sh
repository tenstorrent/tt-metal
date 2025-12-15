#!/bin/bash
set -eo pipefail


fail=0
start_time=$(date +%s)

echo "LOG_METAL: Running simple_text_demo.py for all supported HF models"

# Llama models
declare -a models=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.2-11B-Vision-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "mistralai/Mixtral-8x7B-v0.1"
  "Qwen/Qwen2.5-72B-Instruct"
  "Qwen/Qwen2.5-Coder-32B"
  "Qwen/Qwen3-32B"
)

output_dir="profiling_simple_text_demo"
if [ ! -d "$output_dir" ]; then
  mkdir "$output_dir"
else
  echo "LOG_METAL: Output directory $output_dir already exists. Using existing directory."
fi


# Run Llama models
for hf_model in "${models[@]}"; do
  echo "LOG_METAL: Testing $hf_model"
  model_safe=$(echo "$hf_model" | tr '/' '_')
  output_dir_local="$output_dir/profiling_$model_safe"
  if [ ! -d "$output_dir_local" ]; then
    mkdir -p "$output_dir_local"
  else
    echo "LOG_METAL: Output directory $output_dir_local please delete manually if you want to rerun the test for $hf_model"
    exit 1
  fi
  cmd_string="pytest models/tt_transformers/demo/simple_text_demo.py -k 'device-perf and performance' 2>&1 | tee $output_dir_local/log.txt"
  HF_MODEL=$hf_model \
    python3 -m tracy -o "$output_dir_local" --op-support-count 10000 -v -r -p -m $cmd_string || fail=$((fail + 1))

  echo "LOG_METAL: Completed $hf_model"
done

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "LOG_METAL: All simple_text_demo tests completed in $duration seconds"
echo "LOG_METAL: Total failures: $fail"

$TT_METAL_HOME/tests/scripts/t3000/run_tt_perf_report.sh

if [[ $fail -ne 0 ]]; then
  exit 1
fi
