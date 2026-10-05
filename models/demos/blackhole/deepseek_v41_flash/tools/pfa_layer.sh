#!/bin/bash
# usage (through pfrun.sh): pfa_layer.sh <tag> <variants-file>  -- full prefill-layer test (tests/test_prefill_layer_device.py, S=9/128, layers 0/2/20) per variant
tag=$1; vf=$2
cd /mnt/tt-data/ssinghal/wt/pf_attn
while IFS='|' read -r name envs; do
  [ -z "$name" ] && continue
  log=/mnt/tt-data/ssinghal/dsv4-logs/pf_pf_attn_layer_${tag}_${name}.log
  env $envs timeout 2400 pytest -s -q models/demos/blackhole/deepseek_v41_flash/tests/test_prefill_layer_device.py > $log 2>&1
  echo "== $name ($envs): $(grep -E ' passed| failed' $log | tail -1)"
  grep -E "^PREFILL LAYER [0-9]+ S=[0-9]+: hidden" $log | cut -c1-260 | sed "s/^/   /"
done < $vf
