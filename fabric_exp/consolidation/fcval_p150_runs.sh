#!/bin/bash
# Possible v4rxg3 P150 defect (TODO): FrgValFullRingMcast (Ring, 1 link, flow control, 4096 B, full-ring multicast)
# was killed on two LoudBoxes with v4rxg3. Main first (user, 2026-11-03), short timeout; v4rxg3 only if main passes.
# run_cap: past the timeout it is a hang (killed, read-only hang capture, rc 2). A hang or failure stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
y=$CON/frgap/p150_fcval_fullring.yaml
TO=${FCVAL_TIMEOUT:-150}
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
c=/home/aagarwal/.tt-metal-cache-frg-main_fw; mkdir -p $c
KA_ENV= run_cap fcval_main_fw $y $c $TO TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0; rc=$?
echo "fcval_main_fw: rc $rc"
[ $rc = 0 ] || { echo "!!! main did not pass: v4rxg3 not run"; exit 1; }
c=/home/aagarwal/.tt-metal-cache-frg-v4rxg3; mkdir -p $c
python3 $CON/swap.py apply fabric_exp/variants/v4rxg3 $BK || stop
run_cap fcval_v4rxg3 $y $c $TO; rc=$?
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
echo "fcval_v4rxg3: rc $rc"
echo ALL_DONE
