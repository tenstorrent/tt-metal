#!/bin/bash
# Phase 1, round 1: v3a (faithful consolidation of s79mc as the new v3 kernel file, run in v2's place).
#   -O3: validation (RELAXED test kernel), validation + flushed-signal suites with the STRICT test kernel, benchmarks
#                                                                          labels p1v3a_, p1v3as_ (strict)
#   -O2, -Os: benchmarks, each in its own cache                            labels p1v3ao2_, p1v3aos_
#   -O3 + hook probe: benchmarks                                           label p1v3ahook_
# Prepared by prep_round.sh (v3a_strict, v3a_hook variant dirs).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
case "${1:-a}" in
a)  build v3a $V/v3a $C3 O3 p1v3a $VAL $BENCH
    build v3a_strict $V/v3a_strict $C3 O3 p1v3as $VAL flushmix flushmix4
    build v3a_O2 $V/v3a $C2 O2 p1v3ao2 $BENCH
    build v3a_Os $V/v3a $CS Os p1v3aos $BENCH
    build v3a_hook $V/v3a_hook $C3 O3 p1v3ahook $BENCH ;;
esac
rm -f $M
echo ALL_DONE
