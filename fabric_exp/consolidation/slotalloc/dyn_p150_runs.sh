#!/bin/bash
# Dynamic slot allocator A/B on the P150 (SLOT_ALLOCATOR.md; broker job on a09u32 after host_rebuild_dynslots.sh):
# v4txa11 (swapped in once; -O3, study settings, BW telemetry off) on one suite per step, in the order given, each with
# one slot arm. After each run the logged depths (TT_FABRIC_EXP_LOG_SLOTS=1) must be the step's expected row
# (slot_alloc_table.sh prints them); a mismatch, a failure or a hang stops the job (hang capture armed).
#   dyn_p150_runs.sh PASS SUITE:ARM=EXPECT ...
# ARM: tier  TT_FABRIC_EXP_TIER_SLOTS=1 (main's tier pick)       dyn    TT_FABRIC_EXP_DYN_SLOTS=1 (default policy)
#      def   no switch (the study router's default row today)   dynw24 dyn with TT_FABRIC_EXP_DYN_SLOTS_POLICY=worker_cap_slots=24
#      row   the suite's explicit row switch (TT_FABRIC_EXP_SLOTS_2D / _1D / _VC1 / _VC2) set to EXPECT
#      dynfloor  dyn with the 2D VC0 receiver floor (default policy since 2026-11-04, SLOT_ALLOCATOR §10); skipped on a
#            host library without it. On such a library dyn and dynw24 add vc0_receiver_floor=off, so they stay the
#            rows of the first A/B (lb2 171995, Galaxy 171946) and the expected rows below.
# EXPECT: W,F,R (VC0 only), W,F,R,F1,R1 (VC1 split), W,F,R+W2,R2 (VC2 on).
# SUITE (slot payload 4352 B unless _3k: 3072 B, slotalloc/yamls/*_3072.yaml from mk_dyn_yamls.py):
#   2d, 2d3k      slotstudy/sls_2d_p150.yaml (14 tests: 1-hop E/W, N/S, bidir, 3-hop, merge, turn, multicast)
#   e2d           exp_2d_4352.yaml (the census 2D suite)       mix  streams/p3_mixes.yaml       burst  exp_burst.yaml
#   1d, 1d3k      slotstudy/sls_1d_p150.yaml (6)              ring, ring3k  v4/exp_ring.yaml
#   torusy, torusy3k  v4/exp_torus_y.yaml                     txy, txy3k    torus/torusxy_2x4_sizes.yaml (XY torus, 2x4)
#   vc2e, vc2burst    exp_2d_4352.yaml / exp_burst.yaml with VC2 on (VC0 traffic)
#   vc2ew, vc2alone, vc02ew, vc02bew   pcs3 VC2 rows with VC2 on (VC0 alone / VC2 alone / both, VC0 or VC2 first done)
#   vc1, vc13k    the VC1 split (two ranks, run_vc1_cap): vc1/sls_vc1_all.yaml (82 finishes)
# Labels dyn<PASS>_<suite>_<arm>: results/<label>.log, .csv, _tests.csv (vc1: <label>_13129088_vc1.*). Each run's kernel
# cache (/home/aagarwal/.tt-metal-cache-dyn-<suite>-<arm>) is kept for the code-size check (dyn_size.py); the next run
# of the same suite and arm starts it fresh. Runs are not started past DYN_BUDGET s (default 1150) of job time.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: dyn_p150_runs.sh PASS SUITE:ARM=EXPECT ..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS_VC2 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D \
    TT_FABRIC_EXP_TIER_SLOTS TT_FABRIC_EXP_DYN_SLOTS TT_FABRIC_EXP_DYN_SLOTS_POLICY TT_FABRIC_EXP_WORKER_SLOTS \
    TT_FABRIC_EXP_MAX_SENDER_SLOTS TT_FABRIC_EXP_MAX_RECEIVER_SLOTS TT_FABRIC_EXP_LINEAR_SPARE_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
SA=$CON/slotalloc
Y3=$SA/yamls
VC2="TT_METAL_ENABLE_FABRIC_VC2=1"
for y in $Y3/sls_2d_p150_3072.yaml $Y3/sls_vc1_all_3072.yaml; do [ -f $y ] || { echo "no $y (mk_dyn_yamls.py)"; exit 1; }; done
grep -aq TT_FABRIC_EXP_DYN_SLOTS_POLICY build_Release/tt_metal/libtt_metal.so ||
    { echo "!!! the host library has no TT_FABRIC_EXP_DYN_SLOTS (host_rebuild_dynslots.sh first)"; exit 1; }
# The host library knows the VC0 receiver floor (policy key vc0_receiver_floor, 2026-11-04 rebuild).
FLOOR_HOST=0; grep -aq vc0_receiver_floor build_Release/tt_metal/libtt_metal.so && FLOOR_HOST=1
FLOOR_OFF=""; [ $FLOOR_HOST = 1 ] && FLOOR_OFF=vc0_receiver_floor=off
echo "host library: VC0 receiver floor $([ $FLOOR_HOST = 1 ] && echo "known (dyn, dynw24: $FLOOR_OFF)" || echo "unknown (no dynfloor arm)")"
python3 $CON/swap.py apply fabric_exp/variants/${DYN_VARIANT:-v4txa11} $BK || stop

clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
# check LOG KIND EXPECT: the logged depths must be EXPECT
check() {
    local log=$1 kind=$2 e=$3 w f r f1 r1 w2 r2 v0 v1 n0 n1=1
    case $kind in
        vc1) IFS=, read w f r f1 r1 <<< "$e"; v1="VC1 sender slots {$f1, $f1, $f1} receiver slots {$r1}" ;;
        vc2) IFS=, read w f r <<< "${e%+*}"; IFS=, read w2 r2 <<< "${e#*+}"; v1="VC2 sender slots {$w2} receiver slots {$r2}" ;;
        *) IFS=, read w f r <<< "$e" ;;
    esac
    case $kind in 1d) v0="VC0 sender slots {$w, $f} receiver slots {$r}" ;; *) v0="VC0 sender slots {$w, $f, $f, $f} receiver slots {$r}" ;; esac
    n0=$(clean $log | grep -acF "$v0")
    [ -n "${v1:-}" ] && n1=$(clean $log | grep -acF "$v1")
    echo "   slots check: '$v0' x$n0${v1:+, '$v1' x$n1}"
    [ $n0 -gt 0 ] && [ $n1 -gt 0 ] && return 0
    echo "!!! slots not as expected; logged:"; clean $log | grep -aoE 'VC[0-2] sender slots \{[^}]*\} receiver slots \{[^}]*\}' | sort | uniq -c
    clean $log | grep -a "dynamic slots" | sed 's/.*| //' | sort -u | head -12
    return 1
}

for step in "$@"; do
    suite=${step%%:*}; rest=${step#*:}; arm=${rest%%=*}; e=${rest#*=}
    [ "$e" != "$rest" ] || { echo "!!! $step: no expected row"; stop; }
    if [ $SECONDS -gt ${DYN_BUDGET:-1150} ]; then echo "=== budget: not starting $step or later at ${SECONDS} s"; break; fi
    kind=2d; env=(); n=""
    case $suite in
        2d) y=fabric_exp/slotstudy/sls_2d_p150.yaml; n=14 ;;
        2d3k) y=$Y3/sls_2d_p150_3072.yaml; n=14 ;;
        e2d) y=fabric_exp/exp_2d_4352.yaml ;;
        mix) y=fabric_exp/streams/p3_mixes.yaml ;;
        burst) y=fabric_exp/exp_burst.yaml ;;
        1d) y=fabric_exp/slotstudy/sls_1d_p150.yaml; n=6; kind=1d ;;
        1d3k) y=$Y3/sls_1d_p150_3072.yaml; n=6; kind=1d ;;
        ring) y=fabric_exp/v4/exp_ring.yaml; kind=1d ;;
        ring3k) y=$Y3/exp_ring_3072.yaml; kind=1d ;;
        torusy) y=fabric_exp/v4/exp_torus_y.yaml ;;
        torusy3k) y=$Y3/exp_torus_y_3072.yaml ;;
        txy) y=fabric_exp/torus/torusxy_2x4_sizes.yaml ;;
        txy3k) y=$Y3/torusxy_2x4_sizes_3072.yaml ;;
        vc2e) y=fabric_exp/exp_2d_4352.yaml; kind=vc2; env+=($VC2) ;;
        vc2burst) y=fabric_exp/exp_burst.yaml; kind=vc2; env+=($VC2) ;;
        vc2ew) y=fabric_exp/pcs3/ew4352.yaml; kind=vc2; env+=($VC2) ;;
        vc2alone) y=fabric_exp/pcs3/vc2ew4352.yaml; kind=vc2; env+=($VC2) ;;
        vc02ew) y=fabric_exp/pcs3/vc02ew4352.yaml; kind=vc2; env+=($VC2) ;;
        vc02bew) y=fabric_exp/pcs3/vc02bew4352.yaml; kind=vc2; env+=($VC2) ;;
        vc1) y=fabric_exp/vc1/sls_vc1_all.yaml; kind=vc1 ;;
        vc13k) y=$Y3/sls_vc1_all_3072.yaml; kind=vc1 ;;
        *) echo "!!! unknown suite $suite"; stop ;;
    esac
    case $arm in
        tier) env+=(TT_FABRIC_EXP_TIER_SLOTS=1) ;;
        dyn) env+=(TT_FABRIC_EXP_DYN_SLOTS=1 ${FLOOR_OFF:+TT_FABRIC_EXP_DYN_SLOTS_POLICY=$FLOOR_OFF}) ;;
        dynw24) env+=(TT_FABRIC_EXP_DYN_SLOTS=1 TT_FABRIC_EXP_DYN_SLOTS_POLICY=worker_cap_slots=24${FLOOR_OFF:+,$FLOOR_OFF}) ;;
        dynfloor) [ $FLOOR_HOST = 1 ] || { echo "=== skip $step: the host library has no VC0 receiver floor"; continue; }
                  env+=(TT_FABRIC_EXP_DYN_SLOTS=1) ;;
        def) ;;
        row) case $kind in
                 2d) env+=(TT_FABRIC_EXP_SLOTS_2D=$e) ;; 1d) env+=(TT_FABRIC_EXP_SLOTS_1D=$e) ;;
                 vc1) env+=(TT_FABRIC_EXP_SLOTS_VC1=$e) ;; vc2) env+=(TT_FABRIC_EXP_SLOTS_VC2=${e/+/,}) ;;
             esac ;;
        *) echo "!!! unknown arm $arm"; stop ;;
    esac
    l=dyn${P}_${suite}_${arm}
    c=/home/aagarwal/.tt-metal-cache-dyn-$suite-$arm
    rm -rf "${c:?}"; mkdir -p $c
    echo "##### $l: ${env[*]} expect $e (job at ${SECONDS} s)"
    if [ $kind = vc1 ]; then
        VC1CAP_EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1 ${env[*]}" VC1CAP_SPLIT_YAML=$y VC1CAP_SPLIT_FINISHES=82 \
            VC1CAP_SPLIT_TIMEOUT=300 run_vc1_cap $l $c O3 || stop
        log=$out/${l}_13129088_vc1.log
    else
        rm -f generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
        to=300; case $suite in txy|txy3k) to=900 ;; esac   # the XY-torus size yamls: 64-80 tests (torus_xy_runs.sh: 900 s)
        run_cap $l $y $c $to TT_FABRIC_EXP_LOG_SLOTS=1 "${env[@]}"; rc=$?
        [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
        [ -z "$n" ] || grep -aq "Tests: $n executed, 0 failed" <(clean $out/$l.log) || rc=1
        [ $rc = 0 ] || { echo "!!! $l failed (rc $rc)"; stop; }
        log=$out/$l.log
    fi
    check $log $kind $e || stop
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
