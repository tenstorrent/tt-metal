#!/bin/bash
# usage: run_vllm_tp32.sh <tag> [extra vllm args]   -- TP32 llama3_70b_galaxy server, 4 lanes x 8 = 32 users
TAG="$1"; shift
cd ~/teja/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/home/ubuntu/teja/cache/llama70b
export MESH_DEVICE=TG TT_LLAMA_TEXT_VER=llama3_70b_galaxy VLLM_RPC_TIMEOUT=900000
LOG=~/teja/runs/vllm_server_${TAG}.log
cd ~/teja/vllm-tt-plugin
echo "=== start $(date)" > $LOG
python examples/server_example_tt.py --model "$HF_MODEL" --served-model-name llama70b \
  --data_parallel_size 4 --max_num_seqs 8 ${ASYNC---async-scheduling} --port 8000 \
  --additional-config '{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING", "worker_l1_size": 1344544, "trace_region_size": 220000000}}' \
  "$@" >> $LOG 2>&1
echo "=== EXIT $? $(date)" >> $LOG
