#!/bin/bash
# Receiver multiple-of-4 sweep (SLOT_ALLOCATOR.md §9; coordinator, 2026-11-03), no host build: v4txa11 swapped in once
# (-O3, study settings, BW telemetry off), 3072 B slots (yamls from rxm4_yamls.py), explicit rows only.
#   2D mesh: TT_FABRIC_EXP_SLOTS_2D=20,21,R, R = 16 20 24 28 32 (20 + 3 x 21 + R <= 115 slots)
#   ring:    TT_FABRIC_EXP_SLOTS_1D=43,42,R, R = 24 28 32 (43 + 42 + R <= 117 slots)
# Order: data-checked validation first (mesh sanity's Mesh test at R = 20 24 28; val_ring_p150 at R = 24 28 32), then
# two benchmark passes with the arms interleaved (pass a ascending R, pass b descending), 2D rows then the ring row.
# After every run the logged depths must be the requested row; a failure, a missing result or a hang stops the job
# (hang capture armed). One kernel cache per topology and R, kept (code size per R: dyn_size.py).
#   rxm4_runs.sh [TAG]     labels rxm4<TAG>_<v|a|b>_<2d|mesh|ring>_r<R>: results/<label>.log, .csv, _tests.csv
# Hypothesis stated before the data (coordinator): on mesh routers a multiple-of-4 receiver costs <= 1 cycle/packet;
# on ring routers (first-level acks, completion groups of 1) ~8.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS_VC2 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D \
    TT_FABRIC_EXP_TIER_SLOTS TT_FABRIC_EXP_DYN_SLOTS TT_FABRIC_EXP_DYN_SLOTS_POLICY TT_FABRIC_EXP_WORKER_SLOTS \
    TT_FABRIC_EXP_MAX_SENDER_SLOTS TT_FABRIC_EXP_MAX_RECEIVER_SLOTS TT_FABRIC_EXP_LINEAR_SPARE_SLOTS
T=rxm4${1:-}
Y=$CON/slotalloc/yamls
for y in rxm4_2d_3072 rxm4_meshval_3072 rxm4_ringval_3072 rxm4_ring_3072; do
    [ -f $Y/$y.yaml ] || { echo "no $Y/$y.yaml (rxm4_yamls.py)"; exit 1; }
done
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
python3 $CON/swap.py apply fabric_exp/variants/v4txa11 $BK || stop
for k in 2d ring; do for r in 16 20 24 28 32; do rm -rf /home/aagarwal/.tt-metal-cache-rxm4-$k-$r; done; done

one() {  # kind(2d|mesh|ring) phase R yaml
    local kind=$1 phase=$2 r=$3 y=$4 l=${T}_$2_$1_r$3 rc want topo
    case $kind in
        2d|mesh) topo=2d; env=TT_FABRIC_EXP_SLOTS_2D=20,21,$r; want="VC0 sender slots {20, 21, 21, 21} receiver slots {$r}" ;;
        ring) topo=ring; env=TT_FABRIC_EXP_SLOTS_1D=43,42,$r; want="VC0 sender slots {43, 42} receiver slots {$r}" ;;
    esac
    local c=/home/aagarwal/.tt-metal-cache-rxm4-$topo-$r
    mkdir -p $c
    rm -f generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
    run_cap $l $y $c 600 TT_FABRIC_EXP_LOG_SLOTS=1 $env; rc=$?
    [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
    [ $rc = 2 ] && { echo "!!! $l: hang (captured): stopping"; stop; }
    [ $rc = 0 ] || { echo "!!! $l: rc $rc (failure or no result): stopping"; stop; }
    local n; n=$(sed 's/\x1b\[[0-9;]*m//g' $out/$l.log | grep -acF "$want")
    echo "   slots check: '$want' x$n"
    [ "$n" -gt 0 ] || { echo "!!! $l: slots not as requested"; stop; }
}

echo "##### validation (job at ${SECONDS} s)"
for r in 20 24 28; do one mesh v $r $Y/rxm4_meshval_3072.yaml; done
for r in 24 28 32; do one ring v $r $Y/rxm4_ringval_3072.yaml; done
for pass in a b; do
    [ $pass = a ] && R2="16 20 24 28 32" R1="24 28 32" || R2="32 28 24 20 16" R1="32 28 24"
    echo "##### pass $pass (job at ${SECONDS} s)"
    for r in $R2; do one 2d $pass $r $Y/rxm4_2d_3072.yaml; done
    for r in $R1; do one ring $pass $r $Y/rxm4_ring_3072.yaml; done
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
