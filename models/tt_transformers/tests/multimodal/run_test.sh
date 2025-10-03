#!/bin/bash
set -e

fail=0
# Llama3.2-11B
llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
tt_cache=/mnt/MLPerf/huggingface/tt_cache/$llama11b

CI=true HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py  --timeout 420 || fail=1

llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/
# CI=true HF_MODEL=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py  --timeout 420 || fail=1

echo "LOG_METAL: Llama3 accuracy tests for $llama_dir completed"
if [[ $fail -ne 0 ]]; then
exit 1
fi
