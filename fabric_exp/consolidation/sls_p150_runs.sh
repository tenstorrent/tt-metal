#!/bin/bash
# Slot study (SLOT_STUDY.md, 2026-11-01), P1 on the P150 VC1 split: the lead v4rxv2i3 at -O3 with explicit VC1 slot rows
# (TT_FABRIC_EXP_SLOTS_VC1 = worker, VC0 forwarding, VC0 receiver, VC1 forwarding, VC1 receiver; every router the same
# depths), each allocation one run of fabric_exp/vc1/sls_vc1_all.yaml (the vc1_alloc, vc1_inter and post-merge mix rows;
# 82 finishes over both ranks, the sanity groups data checked), a fresh cache per run, hang capture armed (run_vc1_cap).
# Slots logged (TT_FABRIC_EXP_LOG_SLOTS=1) and checked against the request after every run: a mismatch stops the job.
#   sls_p150_runs.sh PASS TAG=W,F0,R0,F1,R1 ...     labels sls<PASS>_<TAG>_p1_o3 (results/..._13129088_vc1{.log,_tests.csv})
# A failed or hung run stops the job (capture in results/<label>_hangcap). Runs are not started past SLS_BUDGET s
# (default 1000) of job time, so a last run that hangs still fits its capture under the broker's 1500 s.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: sls_p150_runs.sh PASS TAG=W,F0,R0,F1,R1 ..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-sls
python3 $CON/swap.py apply fabric_exp/variants/v4rxv2i3 $BK || stop
for spec in "$@"; do
    tag=${spec%%=*}; a=${spec#*=}
    if [ $SECONDS -gt ${SLS_BUDGET:-1000} ]; then echo "=== budget: not starting $tag ($a) or later runs at ${SECONDS} s"; break; fi
    l=sls${P}_${tag}_p1_o3
    echo "##### $tag = $a (job at ${SECONDS} s)"
    VC1CAP_EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1 TT_FABRIC_EXP_SLOTS_VC1=$a" VC1CAP_SPLIT_YAML=fabric_exp/vc1/sls_vc1_all.yaml \
        VC1CAP_SPLIT_FINISHES=82 VC1CAP_SPLIT_TIMEOUT=300 run_vc1_cap $l $C O3 || stop
    IFS=, read w f0 r0 f1 r1 <<< "$a"
    log=$out/${l}_13129088_vc1.log
    want0="VC0 sender slots {$w, $f0, $f0, $f0} receiver slots {$r0}"
    want1="VC1 sender slots {$f1, $f1, $f1} receiver slots {$r1}"
    n0=$(grep -acF "$want0" $log); n1=$(grep -acF "$want1" $log)
    echo "slots check: '$want0' x$n0, '$want1' x$n1"
    [ $n0 -gt 0 ] && [ $n1 -gt 0 ] || { echo "!!! slots not as requested"; grep -a 'sender slots' $log | sort | uniq -c | head; stop; }
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
