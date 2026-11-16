#!/bin/bash
R=~/teja/run_tp32.sh
$R "long-4k-b1" 4k_b32 --timeout 3600 --batch_size 32 --page_params '{"page_block_size": 64, "page_max_num_blocks": 4096}'
$R "long-16k-b1" 16k_b1 --timeout 3600
$R "long-32k-b1" 32k_b1 --timeout 3600
echo "=== QUEUE DONE $(date)" >> ~/teja/runs/queue.log
