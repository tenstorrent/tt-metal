#!/bin/bash
# usage: run_tp8dp4.sh <tag> [extra pytest args]
cd ~/teja/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/home/ubuntu/teja/cache/llama70b_ttt
export MESH_DEVICE=TG
TAG="$1"; shift
LOG=~/teja/runs/tp8dp4_${TAG}.log
echo "=== start $(date) $*" > $LOG
pytest models/ttt_compat/demo/simple_text_demo.py -k "${K:-performance and batch-32 and not log-probs}" --timeout 5400 --data_parallel 4 "$@" >> $LOG 2>&1
echo "=== EXIT $? $(date)" >> $LOG
