#!/bin/bash
set -eo pipefail

output_dir="profiling_simple_text_demo"

if [ ! -d "$output_dir" ]; then
  echo "LOG_METAL: Output directory $output_dir does not exist"
  exit 1
fi

fail=0
start_time=$(date +%s)

echo "LOG_METAL: Running tt-perf-report on all ops_perf_results_*.csv files"

# Find all model directories
for model_dir in "$output_dir"/profiling_*; do
  if [ ! -d "$model_dir" ]; then
    continue
  fi

  model_name=$(basename "$model_dir" | sed 's/^profiling_//')
  echo "LOG_METAL: Processing $model_name"

  # Find all ops_perf_results_*.csv files in reports subdirectories
  ops_files=$(find "$model_dir/" -name "ops_perf_results_*.csv")

  if [ -z "$ops_files" ]; then
    echo "LOG_METAL: No ops_perf_results_*.csv files found in $model_dir"
    fail=$((fail + 1))
    continue
  fi

  # Run tt-perf-report on each file
  for ops_file in $ops_files; do
    echo "LOG_METAL: Running tt-perf-report on $ops_file and saving to $model_dir/$model_name.log"
    tt-perf-report "$ops_file" > "$model_dir/$model_name.log" 2>&1 || {
      echo "LOG_METAL: Failed to run tt-perf-report on $ops_file"
      fail=$((fail + 1))
    }
  done

  echo "LOG_METAL: Completed $model_name"
done

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "LOG_METAL: All tt-perf-report runs completed in $duration seconds"
echo "LOG_METAL: Total failures: $fail"

if [[ $fail -ne 0 ]]; then
  exit 1
fi
