#!/bin/bash
# usage: pf_demo.sh <logname> <envfile>   -- 40-layer demo session (text_demo.py -k session) from the pf_mhc worktree under the host lock
name=$1; envf=$2
W=/mnt/tt-data/ssinghal/wt/pf_mhc
$W/run_wt.sh "source $envf && echo ENV && env | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_)' | sort && timeout 20000 pytest -x -s -q -o junit_suite_name=pf_mhc_$name models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k session" > /mnt/tt-data/ssinghal/dsv4-logs/pf_pf_mhc_$name.log 2>&1
