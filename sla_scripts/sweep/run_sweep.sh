#!/bin/bash
cd ~/teja/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0
python ~/teja/sweep/sweep_dram_mm.py "$@" > ~/teja/sweep/sweep_$(echo "${@:-all}" | tr ' ' _).log 2>&1
echo "=== EXIT $? $(date)" >> ~/teja/sweep/sweep_$(echo "${@:-all}" | tr ' ' _).log
