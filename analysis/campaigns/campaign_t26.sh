#!/bin/bash
# T2.6 TopK campaign on the fresh main-tip build, per topk/sweep_plan.md section 6 run order.
# Path resolution for both layouts (workspace or $TTM/analysis/campaigns): see campaign_paths.sh.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"
OUT=${OUT:-$HANDOFF/data/topk}   # in a checkout HANDOFF is DD, so the output lands under the invocation directory
SCRIPT=${SCRIPT:-$([ -f "$SD/topk_campaign.py" ] && echo "$SD/topk_campaign.py" || echo "$HANDOFF/topk/topk_campaign.py")}
cd $TTM_FRESH; export TT_METAL_HOME=$TTM_FRESH ARCH_NAME=blackhole PYTHONPATH=$TTM_FRESH:$TTM_FRESH/ttnn:$TTM_FRESH/tools
source $PYENV
grp(){ # classes, mode
  CL=$1; MODE=${2:-plain}; TAG=$(echo $CL | tr ',' '_'); T0=$(date -u +%FT%TZ)
  rm -rf generated/profiler/reports/*
  if [ "$MODE" = mp ]; then TP="python -m tracy -r -p --perf-counter-multipass --profiler-capture-perf-counters=all"; else TP="python -m tracy -r -p -v"; fi
  echo "### topk group $CL mode=$MODE start $T0 sha=$(git rev-parse --short HEAD)"
  $TP $SCRIPT --classes $CL --out $OUT > $OUT/run_$TAG.log 2>&1 || echo "group $CL exited nonzero"
  R=$(ls -d generated/profiler/reports/*/ 2>/dev/null | sort | tail -1)
  if [ -n "$R" ] && ls $R/ops_perf_results*.csv >/dev/null 2>&1; then
    cp $R/ops_perf_results*.csv $OUT/ops_perf_results_$TAG.csv
    cp generated/profiler/.logs/profile_log_device.csv $OUT/profile_log_device_$TAG.csv 2>/dev/null || true
    python3 $SCRIPT --postprocess $OUT/ops_perf_results_$TAG.csv $OUT/cells_$TAG.csv >> $OUT/run_$TAG.log 2>&1 && mv $OUT/topk_campaign_results.csv $OUT/results_$TAG.csv
    FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $OUT/run_$TAG.log | sed "s/firmware bundle version: //")
    echo "PROVENANCE: p100a fw_bundle=$FW; tt-metal-fresh $(git rev-parse HEAD) branch $(git branch --show-current) (main tip 2dbd14bf632 merged, local BRISC firmware-size patch f3fbc8984f0); $T0; cmd: $TP $SCRIPT --classes $CL --out $OUT; report $R" > $OUT/PROVENANCE_$TAG.txt
    echo "### topk group $CL done $(date -u +%H:%M:%SZ) cells=$(wc -l < $OUT/cells_$TAG.csv) results=$(wc -l < $OUT/results_$TAG.csv 2>/dev/null)"
  else
    echo "### topk group $CL FAILED (no ops report) $(date -u +%H:%M:%SZ)"; grep -E "Error|Traceback|FATAL" $OUT/run_$TAG.log | head -5
  fi
}
grp SMOKE
grp L1,L2,L3,L4,L5
grp L6
grp A
grp B,C,D
grp E,F
grp G
grp R1
grp R2,R3
grp R4
grp COUNTERS mp
echo "### T2.6 done $(date -u +%H:%M:%SZ)"
