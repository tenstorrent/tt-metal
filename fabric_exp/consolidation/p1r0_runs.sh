#!/bin/bash
# Phase 1, round 0: references on the baseline (s79mc) before any v3 candidate.
#   s79mc_hook    s79mc + the `hook` probe (perturb.py: a noinline debug call inside the main loop on a value
#                 that is never 0x5AFE), -O3, benchmarks                                        label p1hook_
#   s79mcu        s79mc without the five branch hints the study added (two E0 TX-queue status checks, three E1
#                 pending-atomic branches), -O3, benchmarks                                     label p1nohint_
#   s79mc_strict  s79mc with the test kernel sending flushed packets as FLUSH_STRICT (the default) instead of
#                 FLUSH_RELAXED: validation, then the flushed-signal suites                     label p1strict_
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
case "${1:-a}" in
a)  build hook $V/s79mc_hook $C3 O3 p1hook $BENCH
    build nohint $V/s79mcu $C3 O3 p1nohint $BENCH
    build strict $V/s79mc_strict $C3 O3 p1strict $VAL flushmix flushmix4 ;;
esac
rm -f $M
echo ALL_DONE
