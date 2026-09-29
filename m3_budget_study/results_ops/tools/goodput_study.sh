#!/usr/bin/env bash
# Goodput study in Pavlo's sim (CPU only, no device): 16x[2,4] vs 16x[4,2] under several calibrations, and
# goodput per op fix on 16x[2,4] with our calibration. Outputs: results_ops/goodput/*.json, summarized by
# tools/goodput_summary.py into goodput_results.md tables.
#
#   NODE=... bash tools/goodput_study.sh [mesh|fix|all]
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
R=$(dirname "$T")
O=$R/goodput
NODE=${NODE:-$(ls ~/.vscode-server/cli/servers/*/server/node | head -1)}
W_=${WORKERS:-16}
what=${1:-all}
mkdir -p "$O/fix"
FEAT=near,full,p0p1
# CAL.pipe.ringC for our [2,4] dense ring = Pavlo's pipeline fit x (our zone ring_c / his zone ring_c); the sim scales
# [4,2] by effs['4x2'].dense.ring_c / effs['2x4'].dense.ring_c. ringScan stays at his fit (0.014).
ringc() { python3 -c "import json;print(round(0.30248237322064786*json.load(open('$T/cal_effs_ours_2x4_w$1.json'))['2x4']['dense']['ring_c']/0.3575944448870825,6))"; }
run() { # name W args...
  local n=$1 W=$2; shift 2
  echo "== $n W=$W $*"
  nice -n 10 "$NODE" "$T/run_goodput.js" --budget "$W" --slo 10,3 --workers "$W_" --out "$O/$n.json" "$@" | tail -n +2
}

if [[ $what == mesh || $what == all ]]; then
  for W in 4096 8192; do
    P="{\"ringC\":$(ringc $W)}"
    run pavlo_w$W $W --topo 2x4,4x2 --features $FEAT
    run ours_w$W $W --topo 2x4,4x2 --features $FEAT --effs "$T/cal_effs_ours_2x4_w$W.json,$T/cal_effs_ours_4x2_w$W.json" --pipe "$P"
    run ours_native_w$W $W --topo 2x4,4x2 --features $FEAT --effs "$T/cal_effs_ours_2x4_w$W.json,$T/cal_effs_ours_4x2_native_w$W.json" --pipe "$P"
    # sensitivities: Pavlo's pipe.ringC kept; [2,4] from prose only (the (4,2) profiles are prose only)
    run ours_native_pavlopipe_w$W $W --topo 2x4,4x2 --features $FEAT --effs "$T/cal_effs_ours_2x4_w$W.json,$T/cal_effs_ours_4x2_native_w$W.json"
    run ours_prose_w$W $W --topo 2x4,4x2 --features $FEAT --effs "$T/cal_effs_ours_2x4_prose_w$W.json,$T/cal_effs_ours_4x2_w$W.json" --pipe "$P"
    run ours_prose_native_w$W $W --topo 2x4,4x2 --features $FEAT --effs "$T/cal_effs_ours_2x4_prose_w$W.json,$T/cal_effs_ours_4x2_native_w$W.json" --pipe "$P"
  done
fi

if [[ $what == fix || $what == all ]]; then
  for W in 4096 8192; do
    P="{\"ringC\":$(ringc $W)}"
    B=$T/cal_effs_ours_2x4_w$W.json
    # fix values: target = 70% matmul / 80% DRAM+link; min-chip = roof / (min-chip ms - floor), sparse layers 3-6,
    # prose+code, h=139264 (the fastest chip: the cross-chip wait on the hot expert's column removed)
    if [[ $W == 4096 ]]; then cmb=0.7877; mrm=0.3626; dsm=0.1857; else cmb=0.8; mrm=0.3592; dsm=0.1816; fi
    declare -A FX=(
      [base]='{}'
      [sparse]='{"sparse":0.7}'
      [moe_reduce]='{"moe_reduce":0.8}'
      [moe_reduce_minchip]="{\"moe_reduce\":$mrm}"
      [combine]="{\"combine\":$cmb}"
      [experts]='{"experts":0.7}'
      [dispatch]='{"dispatch":0.8}'
      [dispatch_minchip]="{\"dispatch\":$dsm}"
      [all5]="{\"sparse\":0.7,\"moe_reduce\":0.8,\"combine\":$cmb,\"experts\":0.7,\"dispatch\":0.8}"
    )
    for k in "${!FX[@]}"; do
      echo "{\"2x4\":{\"moe\":${FX[$k]}}}" > "$O/fix/$k.w$W.effs.json"
      run fix/$k.w$W $W --topo 2x4 --features near,full --effs "$B,$O/fix/$k.w$W.effs.json" --pipe "$P"
    done
    # experts: its eff is above target (no effect); the imbalance the sim models is the fixed 1.2 in the roofline
    run fix/experts_imb1.w$W $W --topo 2x4 --features near,full --effs "$B" --pipe "$P" --set expertImb=1
    unset FX
  done
fi
