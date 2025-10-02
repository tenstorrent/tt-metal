#!/bin/bash
set -e

run_llama3_func() {
  fail=0


  # Llama3.2-11B
  llama11b=/mnt/MLPerf/huggingface/hub/meta-llama/Llama-3.2-11B-Vision-Instruct/


  HF_MODEL=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_transformer.py  --timeout 420 || fail=1
  echo "LOG_METAL: Llama3 accuracy tests for $llama_dir completed"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi

}
