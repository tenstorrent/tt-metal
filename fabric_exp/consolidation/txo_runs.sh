#!/bin/bash
# Sender ownership pass stage A/B (V4_SENDER_OWNERSHIP.md §12, 2026-11-02): one arm per codegen stage. txo_runs.sh PASS ROUTER...:
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
# TXO_SET picks the row set, so every arm of a set runs in one job: core (the 2D, one-ERISC, VC2-on and fan-in rows
# above), census (1D, ring, torus, N/S, scatter, and the VC2 supplement: vc02bew*, vc2ew*), vc1 (the VC1 split and
# inter-mesh tests only), vc2d (VC2 on, default VC2 slots: vc2ew4352, vc02ew4352, vc02bew4352).
case "${TXO_SET:-}" in
    core) ;;
    census)
        ROWS="lin1:lin1: ring1:ring1: torusns:torusns: ns4352:ns4352: ewscatter:ewscatter:"
        for s in 256 1024 4352; do ROWS="$ROWS v8-vc02bew$s:vc02bew$s:$VC2+$V8 v8-vc2ew$s:vc2ew$s:$VC2+$V8"; done ;;
    vc1) ROWS="" ;;
    one)  # one-ERISC rows only
        ROWS="ew4352-1e:ew4352:$ONE ewscatter-1e:ewscatter:$ONE mcast-1e:mcast:$ONE" ;;
    vc2d)  # VC2 on with each router's own VC2 slots (as multivc_runs.sh's default): VC2 alone, VC0 + VC2 both ways
        ROWS="d-vc2ew4352:vc2ew4352:$VC2 d-vc02ew4352:vc02ew4352:$VC2 d-vc02bew4352:vc02bew4352:$VC2" ;;
    bisect)  # the final round's regressing rows (passes f1/f2, 2026-11-03): VC2 default slots, VC2 alone at v8, first VC
        ROWS="d-vc2ew4352:vc2ew4352:$VC2 d-vc02ew4352:vc02ew4352:$VC2 d-vc02bew4352:vc02bew4352:$VC2 \
v8-vc2ew4352:vc2ew4352:$VC2+$V8 ew16:ew16: bew1024:bnd/bew1024:" ;;
    x3b)  # the one-ERISC fix A/B (X3a against X3b): one-ERISC rows, two-ERISC controls, VC2 on, the 256 B fan-in pair
        ROWS="ew4352-1e:ew4352:$ONE ewscatter-1e:ewscatter:$ONE mcast-1e:mcast:$ONE ew16:ew16: bew1024:bnd/bew1024: \
ew4352:ew4352: vc2on-ew4352:ew4352:$VC2 v8-vc02ew256:vc02ew256:$VC2+$V8 v8-vc02bew256:vc02bew256:$VC2+$V8" ;;
    *) echo "TXO_SET=core|census|vc1|vc2d|bisect|one|x3b"; exit 1 ;;
esac
for r in "$@"; do [ -d fabric_exp/variants/$r ] || { echo "no variant $r"; exit 1; }; done
for spec in $ROWS; do row=$(echo $spec | cut -d: -f2); [ -f "$(yaml_of $row)" ] || { echo "no yaml for $row"; exit 1; }; done
for r in "$@"; do
    echo "##### $r (job at ${SECONDS} s)"
    c=/home/aagarwal/.tt-metal-cache-txo-$r
    rm -rf "${c:?}"; mkdir -p $c
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    for spec in $ROWS; do
        tag=$(echo $spec | cut -d: -f1); row=$(echo $spec | cut -d: -f2); env=$(echo $spec | cut -d: -f3 | tr + ' ')
        lab=txo${P}__${r}__${tag}
        rm -f generated/fabric/bandwidth_results_blackhole.csv
        run_cap $lab $(yaml_of $row) $c 300 $env
        rc=$?
        [ -f generated/fabric/bandwidth_results_blackhole.csv ] && \
            cp generated/fabric/bandwidth_results_blackhole.csv $out/${lab}_detail.csv
        [ $rc = 0 ] || stop
    done
    cv=/home/aagarwal/.tt-metal-cache-txovc1-$r
    [ "$TXO_SET" != vc1 ] || VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
        run_vc1_cap txo${P}vc1__${r}__o3 $cv O3 || stop
    [ "$TXO_SET" != vc1 ] || VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 txo${P}vc1i__${r}__o3 $cv O3 || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
