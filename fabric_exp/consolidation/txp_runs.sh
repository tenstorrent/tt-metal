#!/bin/bash
# Sender placement calibration probe (V4_SENDER_OWNERSHIP.md §8, 2026-11-02; study only). txp_runs.sh PASS ROUTER...:
# per router (variant directory), swapped in once, fresh kernel cache:
#   2D, VC2 off: ew16, bnd/bew1024, ew4352 (E0-bound)
#   one ERISC:   ew4352 (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1)
#   VC2 on, the study split (VC0 17/14/16, VC2 2/4): ew4352, torusns
#   VC2 on, v8 split (TT_FABRIC_EXP_SLOTS_VC2=17,10,16,8,10): vc02ew256, vc02ew1024, vc02ew4352 (VC0 + VC2 on one link;
#                per-sender rates from the profiler's log lines)
#   VC1 split (two ranks): vc1_alloc.yaml with the hang capture armed, vc1_inter.yaml; -O3
# Labels txp<PASS>__<router>__<row>; logs, summary CSVs and per-direction CSVs (_detail.csv) in fabric_exp/results.
# Report: txp_report.py PASS... A hang or a failed test stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: txp_runs.sh PASS ROUTER..."; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
yaml_of() {
    case $1 in
        bnd/*|sb/*) echo $CON/$1.yaml ;;
        *) echo fabric_exp/pcs3/$1.yaml ;;
    esac
}
VC2="TT_METAL_ENABLE_FABRIC_VC2=1+TT_FABRIC_EXP_LOG_SLOTS=1"  # settings joined by +: the specs below split on spaces
V8="TT_FABRIC_EXP_SLOTS_VC2=17,10,16,8,10"
ONE="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
# tag:row:settings (joined by +)
ROWS="ew16:ew16: bew1024:bnd/bew1024: ew4352:ew4352: ew4352-1e:ew4352:$ONE vc2on-ew4352:ew4352:$VC2 vc2on-torusns:torusns:$VC2 \
v8-vc02ew256:vc02ew256:$VC2+$V8 v8-vc02ew1024:vc02ew1024:$VC2+$V8 v8-vc02ew4352:vc02ew4352:$VC2+$V8"
# TXP_SET=vc2: the VC2 supplement instead (v8 split): VC2 contended (vc02bew*: VC0 40M, VC2 2M, VC2 finishes first) and
# VC2 alone (vc2ew*); no VC1 runs.
if [ "$TXP_SET" = vc2 ]; then
    ROWS=""
    for s in 256 1024 4352; do ROWS="$ROWS v8-vc02bew$s:vc02bew$s:$VC2+$V8 v8-vc2ew$s:vc2ew$s:$VC2+$V8"; done
fi
for r in "$@"; do [ -d fabric_exp/variants/$r ] || { echo "no variant $r"; exit 1; }; done
for spec in $ROWS; do row=$(echo $spec | cut -d: -f2); [ -f "$(yaml_of $row)" ] || { echo "no yaml for $row"; exit 1; }; done
for r in "$@"; do
    echo "##### $r (job at ${SECONDS} s)"
    c=/home/aagarwal/.tt-metal-cache-txp-$r
    rm -rf "${c:?}"; mkdir -p $c
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    for spec in $ROWS; do
        tag=$(echo $spec | cut -d: -f1); row=$(echo $spec | cut -d: -f2); env=$(echo $spec | cut -d: -f3 | tr + ' ')
        lab=txp${P}__${r}__${tag}
        rm -f generated/fabric/bandwidth_results_blackhole.csv
        run_cap $lab $(yaml_of $row) $c 300 $env
        rc=$?
        [ -f generated/fabric/bandwidth_results_blackhole.csv ] && \
            cp generated/fabric/bandwidth_results_blackhole.csv $out/${lab}_detail.csv
        [ $rc = 0 ] || stop
    done
    cv=/home/aagarwal/.tt-metal-cache-txpvc1-$r
    [ "$TXP_SET" = vc2 ] || VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
        run_vc1_cap txp${P}vc1__${r}__o3 $cv O3 || stop
    [ "$TXP_SET" = vc2 ] || VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 txp${P}vc1i__${r}__o3 $cv O3 || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
