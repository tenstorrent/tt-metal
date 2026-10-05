#!/bin/bash
# usage (through pfrun.sh): pfa_prof.sh <tag> <variants-file> [layers]  -- like pfa_sweep.sh but with the device profiler; prints the device-time total per layer (tools/opsum.py)
tag=$1; vf=$2; layers=${3:-2,3}
cd /mnt/tt-data/ssinghal/wt/pf_attn
while IFS='|' read -r name envs; do
  [ -z "$name" ] && continue
  O=/mnt/tt-data/ssinghal/dsv4-logs/pf_pf_attn_prof_${tag}_${name}; rm -rf $O; mkdir -p $O
  log=$O.log
  env DSV41_PS_PROF=1 DSV41_PS_DYN=1 $envs DSV41_PS_S=4096 DSV41_PS_C=1024 DSV41_PS_U=4 DSV41_PS_LAYERS=$layers TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_DIR=$O \
    timeout 1500 pytest -x -s -q models/demos/blackhole/deepseek_v41_flash/tests/test_prefill_sparse_device.py > $log 2>&1
  echo "== $name ($envs): $(grep -E ' passed| failed' $log | tail -1)"
  grep -E "^PSPARSE\[auto\] layer" $log | cut -c1-120 | sed "s/^/   /"
  python3 models/demos/blackhole/deepseek_v41_flash/tools/opsum.py $O > $O.txt 2>&1
  grep -E "^====|^---- total" $O.txt | sed "s/^/   /"
  rm -f $O/.logs/profile_log_device.csv
done < $vf
