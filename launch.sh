#!/bin/bash
# usage: launch.sh <tag> "<env>" -- run the demo session on this host under the device lock, log to dsv4-logs/pf_pf_host_<tag>.log
P=models/demos/blackhole/deepseek_v41_flash
setsid nohup /mnt/tt-data/ssinghal/wt/pf_host/$P/tools/pfrun.sh "DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1 $2 timeout 43200 pytest -x -s -q $P/demo/text_demo.py -k session" > /mnt/tt-data/ssinghal/dsv4-logs/pf_pf_host_$1.log 2>&1 < /dev/null &
