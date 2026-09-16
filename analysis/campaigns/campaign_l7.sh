#!/bin/bash
# L7: where the generic single-core TopK launch cost goes. The row law of the campaign puts c_launch_g at
# 7.2 us (topk/topk_campaign_results.md s4) and the ops report shows only 0.5 us of firmware around the
# kernel, so the rest is inside the kernel. This block runs the SMOKE cell set (which carries the
# (rows 32, N 4096, K 32) single-core generic cell) twice, zones off and zones on, and reduces the zoned
# device log per RISC and zone.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null     # SDPA zones stay off for this block
run_one() {   # run_one <tag> <zones 0|1>
  local TAG=$1 Z=$2
  $SD/set_topk_zone_config.sh $Z
  cd $TTM; export TT_METAL_HOME=$TTM ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1; source $PYENV
  rm -rf generated/profiler/reports/*; rm -f generated/profiler/.logs/profile_log_device.csv
  local T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "### $TAG zones=$Z start $T0 sha=$(git rev-parse --short HEAD)"
  python -m tracy -r -p -v $TTM/analysis/campaigns/topk_campaign.py --classes SMOKE --iters 3 --out $DD \
      > $DD/$TAG.log 2>&1 || true
  if [ ! -s generated/profiler/.logs/profile_log_device.csv ]; then echo "RUN FAILED, see $DD/$TAG.log"; return 1; fi
  { echo "PROVENANCE: p100a 110 cores 1350MHz; tt-metal $(git rev-parse HEAD) branch $(git branch --show-current); $T0; TOPK_ZONES=$Z; cmd: python -m tracy -r -p -v analysis/campaigns/topk_campaign.py --classes SMOKE --iters 3";
    cat generated/profiler/.logs/profile_log_device.csv; } > $DD/$TAG.csv
  local R=$(ls -d generated/profiler/reports/*/ 2>/dev/null | sort | tail -1); sleep 2
  [ -n "$R" ] && cp $R/ops_perf_results*.csv $DD/${TAG}_ops_perf_results.csv 2>/dev/null
  tail -n +2 $DD/$TAG.csv > /tmp/zr_l7_$$.csv
  python analysis/zone_reduce.py /tmp/zr_l7_$$.csv --out $DD/$TAG | tee -a $DD/$TAG.log | grep -v "^wrote"
  rm -f /tmp/zr_l7_$$.csv
  cd $DD
  echo "### $TAG done $(date -u +%H:%M:%SZ)"
}
run_one l7_topk_smoke_zoff 0
run_one l7_topk_smoke_zon 1
$SD/set_topk_zone_config.sh 0 >/dev/null
echo "### L7 done $(date -u +%FT%TZ)"
