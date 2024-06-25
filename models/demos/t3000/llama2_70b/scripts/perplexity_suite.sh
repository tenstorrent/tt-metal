#!/bin/bash

# Set variables
models=("llama2" "llama3")
types=("tt" "meta")
seq_lens=("128" "2k")
gen_lens=("128")

layers=80
sampling_method="greedy"


# Define a function to run a single test case
run_test_case() {
  local model=$1
  local type=$2
  local seq_len=$3
  local gen_len=$4

  # Print a description of what's running with all input params
  echo "Running test case for model: $model, type: $type, seq_len: $seq_len, gen_len: $gen_len"

  pytest -svv models/demos/t3000/llama2_70b/demo/eval.py::test_LlamaModel_demo[wormhole_b0-True-wikitext-${seq_len}-${gen_len}-${sampling_method}-${type}-70b-${layers}L-${model}]

}

# Run tests for llama2 and llama3 with datasets wikitext-128 and wikitext-2k
for type in "${types[@]}"; do
  for model in "${models[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
      for gen_len in "${gen_lens[@]}"; do
        run_test_case "$model" "$type" "$seq_len" "$gen_len"
      done
    done
  done
done


echo "All tests completed."
