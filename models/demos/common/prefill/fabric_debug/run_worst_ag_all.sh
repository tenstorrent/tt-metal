#!/usr/bin/env bash
# Run all three legs sequentially and report AllGather device-kernel mean/min/max per leg.
WK=/home/ppopovic/tt-metal/models/demos/common/prefill/fabric_debug; RES=$WK/logs/worst_ag_summary.txt
: > "$RES"
parse(){ F=$(find "$1" -name 'ops_perf_results*.csv' 2>/dev/null|head -1)
  if [ -z "$F" ]; then echo "$2  NO-CSV (leg failed)" | tee -a "$RES"; return; fi
  awk -F, -v lab="$2" 'NR>1&&$4==0&&$1=="AllGatherDeviceOperation"&&$20>0{n++;k=$20/1000;s+=k;if(k>mx)mx=k;if(k<mn||mn==""){mn=k}}
    END{printf "%-14s mean=%6.1fus  min=%6.1fus  max=%6.1fus  n=%d\n",lab,s/n,mn,mx,n}' "$F" | tee -a "$RES"; }
bash "$WK/run_worst_ag_single.sh" 1d >>"$WK/logs/all_1d.log" 2>&1;  parse /data/ppopovic/prof_out/worst_ag_1gal_1d "single_1D"
bash "$WK/run_worst_ag_single.sh" 2d >>"$WK/logs/all_2d.log" 2>&1;  parse /data/ppopovic/prof_out/worst_ag_1gal_2d "single_2D"
bash "$WK/run_worst_ag_pipe.sh"      >>"$WK/logs/all_pipe.log" 2>&1; parse /data/ppopovic/prof_out/worst_ag_pipe "connected_2D"
echo "=== ALL DONE $(date) ===" | tee -a "$RES"
