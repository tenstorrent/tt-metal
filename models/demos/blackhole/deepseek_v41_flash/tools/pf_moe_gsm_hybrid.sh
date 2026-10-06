#!/bin/bash
cd /mnt/tt-data/ssinghal/wt/pf_moe
D=models/demos/blackhole/deepseek_v41_flash
exec $D/tools/devrun_pf.sh "env DSV41_LAYERS=0-39 DSV41_PREFILL_MOE=unified DSV41_UNI_LAYERS=$1 DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1 DSV41_BUILD_SLOTS=10 DSV41_SESSION=gsm8k_b16 timeout 43200 pytest -x -s -q -o junit_suite_name=pf_moe_gsm $D/demo/text_demo.py -k session"
