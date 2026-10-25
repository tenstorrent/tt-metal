#!/bin/bash
# VC1 hang-state capture (user-approved, 2026-10-25): read-only tt-exalens capture of every router on a hang
# (hang_capture.py, decoded by hang_decode.py; vc1_capture_lib.sh).
#   selftest  one VC1 sanity run of v3l3 at -O3, then a capture of the idle device (checks the tool)   label v3l3_o3_capselftest
#   rep       v3l3 at -O3, up to 10 sanity runs, stop at the first failure; on a hang, capture         labels p1capl3_r<i>, p1capl3_hangcap
#   lead      the lead (v3l3be1c) robustness repeats: 10 sanity runs at each of -Os (auto), -O2, -O3, stopping at the
#             first failure; on a hang, capture                                                  labels p1capm_<lvl>_r<i>, p1capm_<lvl>_hangcap
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
V=fabric_exp/variants
case "$1" in
selftest) vc1_capture_selftest v3l3 $V/v3l3 O3 ;;
rep)      vc1_repeat_capture v3l3 $V/v3l3 p1capl3 O3 10 ;;
lead)     for o in auto O2 O3; do
              vc1_repeat_capture v3l3be1c $V/v3l3be1c p1capm_$(echo $o | tr A-Z a-z) $o 10 || { echo STOPPED; break; }
          done ;;
*)        echo "usage: p1cap_runs.sh selftest|rep|lead"; exit 1 ;;
esac
rm -f $M
echo ALL_DONE
