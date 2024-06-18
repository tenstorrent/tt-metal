#!/bin/bash

# Set variables
model1="llama2"
model2="llama3"
layers=80
dataset_128="wikitext-128"
dataset_2k="wikitext-2k"
sampling_method="greedy"
type1="tt"
type2="meta"

# Define a function to run a single test case
run_test_case() {
  local model=$1
  local dataset=$2
  local type=$3

  echo "Running test for model: $model with dataset: $dataset and implementation type: $type"

  pytest -svv models/experimental/llama2_70b/demo/eval.py::test_LlamaModel_demo[wormhole_b0-True-${dataset}-${sampling_method}-${type}-70b-${layers}L-${model}]
}

# Run tests for llama2 and llama3 with datasets wikitext-128 and wikitext-2k
run_test_case "$model1" "$dataset_128" "$type1"
run_test_case "$model1" "$dataset_2k" "$type1"
run_test_case "$model2" "$dataset_128" "$type1"
run_test_case "$model2" "$dataset_2k" "$type1"

run_test_case "$model1" "$dataset_128" "$type2"
run_test_case "$model1" "$dataset_2k" "$type2"
run_test_case "$model2" "$dataset_128" "$type2"
run_test_case "$model2" "$dataset_2k" "$type2"


echo "All tests completed."
