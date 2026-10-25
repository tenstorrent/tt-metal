#!/bin/bash
# Phase 1, round 3: v3l (lean specialized E0; E1 = v3a) against v3a (§85). Variants from prep_round.sh.
#   a: v3l -O3: validation (RELAXED), validation + flushed-signal suites (STRICT), benchmarks; hook probe
#                                                          labels p3l_, p3ls_ (strict), p3lhook_
#   b: v3l at -O2 and -Os (benchmarks), and VC1 (two 2x2 meshes) at the default level (-Os), -O2 and -O3
#                                                          labels p3lo2_, p3los_, p3vc1l_{auto,o2,o3}_
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
case "${1:-a}" in
a)  build v3l $V/v3l $C3 O3 p3l $VAL $BENCH
    build v3l_strict $V/v3l_strict $C3 O3 p3ls $VAL flushmix flushmix4
    build v3l_hook $V/v3l_hook $C3 O3 p3lhook $BENCH ;;
b)  build v3l_O2 $V/v3l $C2 O2 p3lo2 $BENCH
    build v3l_Os $V/v3l $CS Os p3los $BENCH
    source $CON/vc1_lib.sh
    vc1_variant v3l $V/v3l p3vc1l auto O2 O3 ;;
esac
rm -f $M
echo ALL_DONE
