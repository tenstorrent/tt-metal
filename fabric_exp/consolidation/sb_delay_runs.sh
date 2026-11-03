#!/bin/bash
# Send-batch study (V4_SEND_BATCH.md, 2026-11-01): delay-perturbation sweep (feedback-robust-over-peak) of routers that
# carry the study's spin hooks (variants/v4bdd = the lead + hooks; v4sbk2d = the P1 probe + the same hooks; at 0 every
# image equals its router's). ROUTERS='v4bdd v4sbk2d' [ARMS='base tx8 ...'] [EXTRA_ENV=...] sb_delay_runs.sh TAG ROW...
# Arms: base, tx<N> (spin after each send), rx<N> (after each packet received), sp<N> / rp<N> (per sender / receiver
# pass). ROW as in sb_rows_runs.sh. Fresh cache per router and tag. Labels sbd<TAG>_<router>_<row>_<arm> (row '/' -> '-').
# Report: sb_delay_report.py TAG. Hang capture armed; a hang or a failed test stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2
T=$1; shift
[ -n "$T" ] && [ $# -gt 0 ] && [ -n "$ROUTERS" ] || { echo "usage: ROUTERS='a b' sb_delay_runs.sh TAG ROW..."; exit 1; }
ARMS=${ARMS:-base tx8 tx16 tx32 rx8 rx16 rx32}
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
yaml_of() {
    case $1 in
        bnd/*|sb/*) echo $CON/$1.yaml ;;
        *) echo fabric_exp/pcs3/$1.yaml ;;
    esac
}
arm_env() {
    case $1 in
        base) echo "" ;; tx*) echo "TT_FABRIC_EXP_TX_DELAY=${1#tx}" ;; rx*) echo "TT_FABRIC_EXP_RX_DELAY=${1#rx}" ;;
        sp*) echo "TT_FABRIC_EXP_DRISC_DELAY=${1#sp}" ;; rp*) echo "TT_FABRIC_EXP_DRISC_DELAY=$(( ${1#rp} << 16 ))" ;;
    esac
}
for row in "$@"; do [ -f "$(yaml_of $row)" ] || { echo "no yaml for row $row"; exit 1; }; done
for r in $ROUTERS; do [ -d fabric_exp/variants/$r ] || { echo "no variant $r"; exit 1; }; done
for r in $ROUTERS; do
    c=/home/aagarwal/.tt-metal-cache-sbd-$T-$r
    rm -rf "${c:?}"; mkdir -p $c
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    for row in "$@"; do
        for a in $ARMS; do
            lab=sbd${T}_${r}_${row//\//-}_$a
            run_cap $lab $(yaml_of $row) $c 300 $(arm_env $a)
            [ $? = 0 ] || stop  # a hang (2) or a failed test (1) stops the job
        done
    done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
