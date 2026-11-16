#!/bin/bash
# tt-inference-server v0.9.0, prod spec for Llama-3.3-70B on WH Galaxy
cd ~/teja/tt-inference-server
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export MODEL_SOURCE=local MODEL_WEIGHTS_DIR=/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct
export PERSISTENT_VOLUME_ROOT=/home/ubuntu/teja/tis_volume AUTOMATIC_HOST_SETUP=1
python3 run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow server --docker-server --no-auth \
  --vllm-override-args '{"enable_prompt_tokens_details": true}' "$@"
