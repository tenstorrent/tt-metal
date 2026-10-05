#!/bin/bash
# usage (run through pfrun.sh, i.e. under the device lock): pfa_sweep.sh <tag> <variants-file> [layers] [extra env]
# variants-file: one "name|ENV=val ENV2=val" per line; every variant runs tests/test_prefill_sparse_device.py (S=4096, C=1024, U=4, DYN) and logs to dsv4-logs/pf_pf_attn_<tag>_<name>.log
tag=$1; vf=$2; layers=${3:-2,3}; extra=${4:-DSV41_PS_DYN=1}
cd /mnt/tt-data/ssinghal/wt/pf_attn
while IFS='|' read -r name envs; do
  [ -z "$name" ] && continue
  log=/mnt/tt-data/ssinghal/dsv4-logs/pf_pf_attn_${tag}_${name}.log
  env $extra $envs DSV41_PS_S=4096 DSV41_PS_C=1024 DSV41_PS_U=4 DSV41_PS_LAYERS=$layers \
    timeout 1500 pytest -x -s -q models/demos/blackhole/deepseek_v41_flash/tests/test_prefill_sparse_device.py > $log 2>&1
  echo "== $name ($envs): $(grep -E 'passed|failed' $log | tail -1)"
  grep -E "^PSPARSE\[auto\] layer|selection layer" $log | cut -c1-330 | sed "s/^/   /"
done < $vf
