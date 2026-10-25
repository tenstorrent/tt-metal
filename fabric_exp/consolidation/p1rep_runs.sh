#!/bin/bash
# Phase 1: VC1 repeat runs at the default level (-Os) on the sanity group where both VC1 hangs happened
# (fabric_exp/vc1/vc1_sanity.yaml; RESULTS §90, §92), up to 10 runs each, stopping at the first failure or hang.
#   base   v3a (every design's ancestor; E1 has the VC1 admission bug of §87)      labels p1base_r<i>_
#   l2be   v3l2be1b (v3l2b's E0 with v3a's exact idle tests + v3e1b's E1)        labels p1l2be_r<i>_
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
V=fabric_exp/variants
case "$1" in
base) vc1_repeat v3a $V/v3a p1base auto 10 ;;
l2be) vc1_repeat v3l2be1b $V/v3l2be1b p1l2be auto 10 ;;
*)    echo "usage: p1rep_runs.sh base|l2be"; exit 1 ;;
esac
rm -f $M
echo ALL_DONE
