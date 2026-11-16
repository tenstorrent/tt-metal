#!/bin/bash
# TP32 MMLU without --async-scheduling; resumes from the v2 answer cache.
cd ~/teja/runs
[ -f mmlu_cache_tp32_v3_rank0.db ] || cp mmlu_cache_tp32_v2_rank0.db mmlu_cache_tp32_v3_rank0.db
pkill -f server_example_tt.py; sleep 20
tmux kill-window -t teja:vllmsrv 2>/dev/null
tt-smi -glx_reset_auto > reset_mmlu3.log 2>&1
tmux new-window -d -t teja -n vllmsrv "ASYNC= bash ~/teja/run_vllm_tp32.sh mmlu3 --max-model-len 8192"
until curl -sf localhost:8000/v1/models > /dev/null; do sleep 30; done
bash ~/teja/run_mmlu.sh tp32_v3
touch mmlu_tp32_v3.done
