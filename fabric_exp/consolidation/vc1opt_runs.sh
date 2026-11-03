#!/bin/bash
# VC1 opt-level A/B (2026-11-01). The host builds VC1 routers at -Os (erisc_datamover_builder.cpp
# get_kernel_opt_level: -Os when the router has VC1 receiver channels); study runs force -O3. Arms, all on the VC1
# split (vc1_alloc.yaml with the hang capture armed, then vc1_inter.yaml), same job:
#   os    v4rxv2i3 at the builder's level (auto: -Os on every router of the split)
#   osni  v4rxv2i3ni = v4rxv2i3 + tt_buf_rw::note<> always_inline (v4txr5's api/dataflow/buf_rw_note.h), auto
#   o3    v4rxv2i3 at -O3
# vc1opt_runs.sh PASS ARM...; labels vo<pass>vc1[i]_<arm>_default_p1_<opt>; router ELFs of each alloc run saved under
# consolidation/builds/vo<pass>_<arm>/. A failed run, hang or stack check stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: vc1opt_runs.sh PASS ARM..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }

for a in "$@"; do
    case $a in
        os) v=v4rxv2i3; o=auto ;;
        osni) v=v4rxv2i3ni; o=auto ;;
        o3) v=v4rxv2i3; o=O3 ;;
        *) echo "unknown arm $a"; exit 1 ;;
    esac
    lo=$(echo $o | tr A-Z a-z)
    echo "##### $a: $v at $o (job at ${SECONDS} s)"
    c=/home/aagarwal/.tt-metal-cache-vc1opt-$a
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || stop
    touch $M; sleep 1
    VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
        run_vc1_cap vo${P}vc1_${a}_default_p1_$lo $c $o || stop
    stack_check $c $M || stop
    save_builds vo${P}_$a $c $M
    VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 vo${P}vc1i_${a}_default_p1_$lo $c $o || stop
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
