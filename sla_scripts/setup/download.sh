#!/bin/bash
set -e
source ~/teja/tt-metal/python_env/bin/activate
exec > >(tee -a ~/teja/download.log) 2>&1
echo "=== start $(date)"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.3-70B-Instruct',
  local_dir='/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct',
  allow_patterns=['*.safetensors','*.json','tokenizer*'], max_workers=16)
"
echo "=== DONE $(date)"
