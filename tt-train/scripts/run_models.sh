# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -e

[[ -z "${TT_METAL_HOME}" ]] && echo "TT_METAL_HOME is not set" && exit 1
[[ -z "${TT_METAL_RUNTIME_ROOT}" ]] && echo "TT_METAL_RUNTIME_ROOT is not set" && exit 1

#export TT_MESH_GRAPH_DESC_PATH="${TT_METAL_HOME}/tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_8x4_mesh_graph_descriptor.textproto"
export TT_LOGGER_LEVEL=off

tt_train=${TT_METAL_HOME}/tt-train
build_examples=${TT_METAL_HOME}/build/tt-train/sources/examples
nano_gpt_bin=${build_examples}/nano_gpt/nano_gpt
linear_regression_tp_dp_bin=${build_examples}/linear_regression_tp_dp/linear_regression_tp_dp

# Additional metadata
metadata_header="date,model_name,model_filename,binary_name,args,git_commit_hash,"

# ("model name" "filename friendly name" "path to binary/script" "arguments")
declare -a MODELS=()
MODELS+=("NanoGPT Shakespeare" "nanogpt" "${nano_gpt_bin}" "-c ${tt_train}/configs/training_configs/training_shakespeare_nanogpt.yaml")
# MODELS+=("NanoLlama3 Shakespeare " "nanollama" "${nano_gpt_bin}" "-c ${tt_train}/configs/training_configs/training_shakespeare_nanollama3.yaml")
# MODELS+=("Linear Regression TP+DP" "linear_regression_tp_dp" "${linear_regression_tp_dp_bin}" "--mesh_shape=8x4")

# Change idx increment to the number of fields in MODELS array
for (( idx=0 ; idx<${#MODELS[@]} ; idx+=4 )) ;
do
  model_name=${MODELS[idx]}
  model_filename=${MODELS[idx+1]}
  binary=${MODELS[idx+2]}
  args=${MODELS[idx+3]}

  # Get epoch time in nanoseconds, then cut to microseconds
  current_time=$(date +%s%N | cut -b1-16)

  echo "Running ${model_filename}: ${binary} ${args}"
  out_basename="${model_filename}_memory_analysis_${current_time}"

  ${binary} ${args} | tee ${out_basename}.log
  python ${tt_train}/scripts/analyze_memory.py --logs ${out_basename}.log

  # quietly set empty string if not in git repo
  git_commit_hash=$(git rev-parse HEAD) || echo ""

  # Prepend metadata header to first line
  sed -i "1s/^/${metadata_header}/" ${out_basename}.csv
  # Prepend additional metadata to second line.
  metadata="${current_time},${model_name},${model_filename},${binary},${args},${git_commit_hash},"
  # Use parameter expansion to escape all forward slashes: ${metadata//pattern/string}, pattern: \/, string: \\/
  sed -i "2s/^/${metadata//\//\\/}/" ${out_basename}.csv

done
