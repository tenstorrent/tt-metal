#!/bin/bash
# TP32 MMLU, then TP8/DP4 MMLU. Each retried up to 3 times; --use_cache resumes answered questions.
srv_tp32() { tmux new-window -d -t teja -n vllmsrv "bash ~/teja/run_vllm_tp32.sh mmlu2 --max-model-len 8192"; }
srv_tp8() { tmux new-window -d -t teja -n tp8srv "export OUT=/home/ubuntu/teja/runs/mmlu2_tp8_server; source ~/teja/bh_vllm/wh_env.sh; bash ~/teja/bh_vllm/run_server.sh --max-model-len 8192"; }
stop_all() { pkill -f server_example_tt.py; sleep 20; tmux kill-window -t teja:vllmsrv 2>/dev/null; tmux kill-window -t teja:tp8srv 2>/dev/null; tt-smi -glx_reset_auto >> ~/teja/runs/reset_mmlu2.log 2>&1; }
run() {  # $1 tag, $2 server fn
  for try in 1 2 3; do
    stop_all; $2; sleep 120
    bash ~/teja/run_mmlu.sh $1
    rg -q "MMLU EXIT 0" ~/teja/runs/mmlu_$1.log && return
    cp ~/teja/runs/mmlu_$1.log ~/teja/runs/mmlu_$1.try$try.log
  done
}
run tp32_v2 srv_tp32
run tp8dp4_v2 srv_tp8
stop_all
echo "CHAIN2 DONE" > ~/teja/runs/mmlu_chain2.done
