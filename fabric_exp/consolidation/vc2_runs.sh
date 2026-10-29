#!/bin/bash
# VC2 on v4 (2026-10-29; consolidation/V4_VC2.md, variants/v4vc2 = v4udm + VC2: a second worker channel and a receiver
# channel that only delivers). test_fabric_vc2_at_least_2x2_mesh.yaml (data-checked neighbour exchange: VC2 alone, VC0
# and VC2 together, uni- and bidirectional; groups without VC2 too), with ROUTER:
#   v1     main's router as shipped (kernel v1, flags 0, firmware keepalive): the control
#   v4vc2  variants/v4vc2 in the v2 slot (the study's settings)
# in both ERISC modes (two, and one: TT_METAL_DISABLE_FABRIC_TWO_ERISC=1). Every group was vetted on the mock cluster
# first; the yaml needed the streams port's VC2 fix (fabric_exp/streams/port/streams_vc2_replay.patch). Hang capture
# armed (a hang stops the job); test failures do not. Fresh cache per router and mode. Offline: the 84 VC2 configs build;
# every other config's image equals v4udm's (22 VC1 senders up to a stack-slot permutation in sender::run).
# Logs fabric_exp/results/vc2_<router>_<mode>.log.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
Y=tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_vc2_at_least_2x2_mesh.yaml
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
r=$1
modes() {  # env...
    local c=/home/aagarwal/.tt-metal-cache-vc2-$r
    rm -rf "${c:?}-two" "${c:?}-one"; mkdir -p $c-two $c-one
    run_cap vc2_${r}_two $Y $c-two 600 "$@"; [ $? = 2 ] && return 2
    run_cap vc2_${r}_one $Y $c-one 600 "$@" $E; [ $? = 2 ] && return 2
    return 0
}
case $r in
    v1)
        KA_ENV= modes TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0 || { echo STOPPED; exit 1; } ;;
    v4vc2)
        python3 $CON/swap.py apply fabric_exp/variants/v4vc2 $BK || { echo STOPPED; exit 1; }
        modes; rc=$?
        python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
        [ $rc = 2 ] && { echo STOPPED; exit 1; } ;;
    *)
        echo "usage: vc2_runs.sh v1|v4vc2"; exit 1 ;;
esac
rm -f $M
echo ALL_DONE
