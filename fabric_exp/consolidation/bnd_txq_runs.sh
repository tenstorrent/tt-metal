#!/bin/bash
# TX-queue probe (2026-11-01; V4_BOUNDS.md §C): variants/v4bdq (v4rxv2i3 + router_txq_probe.hpp, attribution only) with
# TT_FABRIC_TXQ_COUNTERS=1 (the builder allocates the phase-profile block the probe writes; the host diffs it per test
# into results/<label>_phase.csv). Per row: how often the post-send status read still saw the TX queue busy and the busy
# polls then, and on every 61st send the queue's busy time after the data command and after the register command.
#   bnd_txq_runs.sh TAG ROW...   rows: fabric_exp/pcs3/<row>.yaml or fabric_exp/consolidation/bnd/<row>.yaml
# Labels bndq_<tag>_<row> (.csv rate, _phase.csv probe). One cache. A hang stops the job. bnd_txq_report.py TAG decodes.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
T=$1; shift
[ -n "$T" ] && [ $# -gt 0 ] || { echo "usage: bnd_txq_runs.sh TAG ROW..."; exit 1; }
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2
C=/home/aagarwal/.tt-metal-cache-bndq-$T
rm -rf "${C:?}"; mkdir -p $C
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
python3 $CON/swap.py apply fabric_exp/variants/v4bdq $BK || stop
for r in "$@"; do
    y=fabric_exp/pcs3/$r.yaml
    [ -f $CON/bnd/$r.yaml ] && y=$CON/bnd/$r.yaml
    lab=bndq_${T}_$r
    rm -f $out/${lab}_phase.csv
    run_cap $lab $y $C 200 TT_FABRIC_TXQ_COUNTERS=1 TT_FABRIC_PHASE_PROFILE_CSV=$PWD/$out/${lab}_phase.csv $EXTRA_ROW_ENV
    rc=$?
    [ $rc = 2 ] && stop
    [ $rc = 0 ] || stop
    [ -s $out/${lab}_phase.csv ] || { echo "!!! no phase CSV for $lab"; stop; }
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
