#!/bin/bash
cd ~/teja/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/home/ubuntu/teja/cache/llama70b_ttt
export MESH_DEVICE=TG
LOG=~/teja/runs/profile_tp8_dp4_decode${PROF_TAG:+_$PROF_TAG}.log
echo "=== start $(date)" > $LOG
pytest "models/ttt_compat/tests/test_device_perf.py::test_device_perf_one_iter[decode-llama3_70b-2-1024-2-10-4-32-False]" -s --timeout 3600 >> $LOG 2>&1
echo "=== EXIT $? $(date)" >> $LOG
