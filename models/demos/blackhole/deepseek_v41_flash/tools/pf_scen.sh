#!/bin/bash
# usage: pf_scen.sh <logname> <envfile> [pytest file default tests/test_prefill_scen_device.py]; envfile = lines "export VAR=value"
name=$1; envf=$2; tf=${3:-tests/test_prefill_scen_device.py}
W=/mnt/tt-data/ssinghal/wt/pf_mhc
$W/run_wt.sh "source $envf && timeout 10000 pytest -x -s -q models/demos/blackhole/deepseek_v41_flash/$tf" > /mnt/tt-data/ssinghal/dsv4-logs/pf_pf_mhc_$name.log 2>&1
