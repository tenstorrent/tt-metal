#!/bin/bash

echo "LOG_METAL: Checking number of devices"
python3 -c "import ttnn; print('Number of devices:', ttnn.get_num_devices())"

# Split long set of tests into two groups
# This one runs all the T3K tests
fail=0
start_time=$(date +%s)

echo "LOG_METAL: Running run_t3000_llama3_perplexity_tests_t3000"

llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/
llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/
llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/
llama90b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct/

MESH_DEVICE=TG
for LLAMA_DIR in "$llama1b" "$llama3b" "$llama8b" "$llama11b"; do
    MESH_DEVICE=$MESH_DEVICE LLAMA_DIR=$LLAMA_DIR WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k "accuracy and ci-token-matching" --timeout=3600 ; fail+=$?
done

# 70B and 90B tests has the same configuration between `-k "attention-accuracy"` and `-k "attention-performance"` so we only run one of them
for LLAMA_DIR in "$llama70b" "$llama90b"; do
    MESH_DEVICE=$MESH_DEVICE LLAMA_DIR=$LLAMA_DIR WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/tt_transformers/demo/simple_text_demo.py -k "accuracy and ci-token-matching" --timeout=3600 ; fail+=$?
done
done
