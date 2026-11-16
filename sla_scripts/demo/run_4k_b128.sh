#!/bin/bash
# 128 users (DP4 x 32) at 4K prompts, best WH config, greedy
export RS_LINKS=4 RS_WORKERS=2
export MM_CFG='{"1024x8192":[1,4],"8192x3584":[16,0],"8192x1280":[32,0]}'
SP=$(python3 -c 'import json;n=128;print(json.dumps({"temperature":[0.0]*n,"top_p":[0.08]*n,"top_k":[1]*n,"frequency_penalty":[0.0]*n,"presence_penalty":[0.0]*n,"repetition_penalty":[1.0]*n}))')
K="performance and batch-32 and not log-probs" bash ~/teja/run_tp8dp4.sh best_4k_b128 --batch_size 32 \
  --input_prompts models/ttt_compat/demo/sample_prompts/input_data_long_4k.json --max_seq_len 8192 \
  --page_params '{"page_block_size": 32, "page_max_num_blocks_per_dp": 4608}' --sampling_params "$SP"
echo "RUN4K FINISHED"
