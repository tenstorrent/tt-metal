#!/bin/bash
# usage: run_tp32.sh <pytest -k expr> <tag> [extra pytest args]
cd ~/teja/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/home/ubuntu/teja/cache/llama70b
export MESH_DEVICE=TG
K="$1"; TAG="$2"; shift 2
LOG=~/teja/runs/tp32_${TAG}.log
echo "=== start $(date) -k '$K' $*" > $LOG
pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "$K" "$@" >> $LOG 2>&1
echo "=== EXIT $? $(date)" >> $LOG
