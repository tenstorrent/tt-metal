#!/bin/bash
R=~/teja/run_tp8dp4.sh
P=models/ttt_compat/demo/sample_prompts
K="performance and batch-1 and not ci" bash $R dp4_4k_b1 --input_prompts $P/input_data_long_4k.json --max_seq_len 8192
K="performance and batch-32 and not log-probs" bash $R dp4_4k_b8 --batch_size 8 --input_prompts $P/input_data_long_4k.json --max_seq_len 8192
K="performance and long-context-16k and not ci" bash $R dp4_16k_b1
K="performance and long-context-32k" bash $R dp4_32k_b1
echo "=== TP8DP4 LONG DONE $(date)" >> ~/teja/runs/queue.log
