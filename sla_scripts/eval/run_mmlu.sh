#!/bin/bash
# MMLU (lm-eval mmlu_llama: 5-shot, chat template, greedy, 10 tokens) against the vLLM server on :8000.
# usage: run_mmlu.sh <tag> [extra lm_eval args, e.g. --limit 5]
TAG="$1"; shift
source ~/teja/lmeval_venv/bin/activate
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 30; done
lm_eval --model local-completions \
  --model_args model=${MODEL:-llama70b},base_url=http://localhost:8000/v1/completions,tokenizer=/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct,tokenized_requests=True,max_length=8000,num_concurrent=${NC:-32},max_retries=6,timeout=${TMO:-120} \
  --tasks mmlu_llama --apply_chat_template --fewshot_as_multiturn --batch_size 1 \
  --use_cache ~/teja/runs/mmlu_cache_${TAG} --output_path ~/teja/runs/mmlu_${TAG} --log_samples "$@" > ~/teja/runs/mmlu_${TAG}.log 2>&1
echo "MMLU EXIT $?" >> ~/teja/runs/mmlu_${TAG}.log
