#!/bin/bash
# After the TP32 MMLU run ends: stop TP32 server, reset, start TP8/DP4 server (stock config), run MMLU.
until grep -q "MMLU EXIT" ~/teja/runs/mmlu_full.log 2>/dev/null; do sleep 60; done
pkill -f server_example_tt.py; sleep 20
tmux kill-window -t teja:vllmsrv 2>/dev/null
tt-smi -glx_reset_auto > ~/teja/runs/reset_mmlu_tp8.log 2>&1
export OUT=/home/ubuntu/teja/runs/mmlu_tp8_server
source ~/teja/bh_vllm/wh_env.sh
tmux new-window -d -t teja -n tp8srv "export OUT=$OUT; source ~/teja/bh_vllm/wh_env.sh; bash ~/teja/bh_vllm/run_server.sh --max-model-len 8192"
sleep 120
bash ~/teja/run_mmlu.sh tp8dp4
echo "CHAIN DONE" >> ~/teja/runs/mmlu_tp8dp4.log
