#!/bin/bash

# Set variables
model1="llama2"
model2="llama3"
layers=80
dataset_128="wikitext-128"
dataset_2k="wikitext-2k"
sampling_method="greedy"

# Define a function to run a single test case
run_test_case() {
  local model=$1
  local dataset=$2

  echo "Running test for model: $model with dataset: $dataset"

  pytest -svv models/experimental/llama2_70b/demo/eval.py::test_LlamaModel_demo[wormhole_b0-True-${dataset}-${sampling_method}-tt-70b-T3000-${layers}L-${model}]
}

# Run tests for llama2 and llama3 with datasets wikitext-128 and wikitext-2k
run_test_case "$model1" "$dataset_128"
run_test_case "$model1" "$dataset_2k"
run_test_case "$model2" "$dataset_128"
run_test_case "$model2" "$dataset_2k"

echo "All tests completed."
