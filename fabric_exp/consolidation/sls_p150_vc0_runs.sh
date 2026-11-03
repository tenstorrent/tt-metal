#!/bin/bash
# Slot study (SLOT_STUDY.md, 2026-11-01), P3's P150 side: the lead v4rxv2i3 (-O3, study settings) on the 2x4 P150 mesh
# at the 2D and 1D splits the Galaxy runs (fabric_exp/slotstudy/sls_2d_p150.yaml, sls_1d_p150.yaml: the rows both
# systems have). Each split one run_cap (hang capture armed, fresh cache), slots logged and checked against the request.
#   sls_p150_vc0_runs.sh PASS 2d:TAG=W,F,R ... 1d:TAG=W,F,R ...   labels sls2d_<PASS>_<TAG>, sls1d_<PASS>_<TAG>
# (results/<label>.log, .csv summary, _tests.csv per test). A failure, hang or slot mismatch stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: sls_p150_vc0_runs.sh PASS 2d:TAG=W,F,R|1d:TAG=W,F,R ..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-slsvc0
python3 $CON/swap.py apply fabric_exp/variants/v4rxv2i3 $BK || stop
for spec in "$@"; do
    kind=${spec%%:*}; rest=${spec#*:}; tag=${rest%%=*}; a=${rest#*=}
    if [ $SECONDS -gt ${SLS_BUDGET:-1000} ]; then echo "=== budget: not starting $spec or later runs at ${SECONDS} s"; break; fi
    l=sls${kind}_${P}_${tag}
    case $kind in
        2d) y=fabric_exp/slotstudy/sls_2d_p150.yaml; sw=TT_FABRIC_EXP_SLOTS_2D; n=14 ;;
        1d) y=fabric_exp/slotstudy/sls_1d_p150.yaml; sw=TT_FABRIC_EXP_SLOTS_1D; n=6 ;;
        *) echo "!!! unknown kind $kind"; stop ;;
    esac
    echo "##### $l: $sw=$a (job at ${SECONDS} s)"
    rm -rf "${C:?}"; mkdir -p $C
    rm -f generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
    run_cap $l $y $C 300 TT_FABRIC_EXP_LOG_SLOTS=1 $sw=$a; rc=$?
    [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
    grep -aq "Tests: $n executed, 0 failed" $out/$l.log || rc=1
    [ $rc = 0 ] || { echo "!!! $l failed (rc $rc)"; stop; }
    IFS=, read w f r <<< "$a"
    if [ $kind = 2d ]; then want="VC0 sender slots {$w, $f, $f, $f} receiver slots {$r}"; else want="VC0 sender slots {$w, $f} receiver slots {$r}"; fi
    k=$(grep -acF "$want" $out/$l.log)
    echo "slots check: '$want' x$k"
    [ $k -gt 0 ] || { echo "!!! slots not as requested"; grep -a 'sender slots' $out/$l.log | sed 's/.*Fabric[^|]*| //' | sort | uniq -c | head; stop; }
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
