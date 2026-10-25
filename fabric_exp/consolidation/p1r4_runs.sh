#!/bin/bash
# Phase 1, round 4: v3e1 (E1 program rewrite; E0 = v3a) against v3a (§85). Variants from prep_round.sh.
#   a: v3e1 -O3: validation (RELAXED), validation + flushed-signal suites (STRICT), benchmarks; hook probe
#                                                          labels p4e_, p4es_ (strict), p4ehook_
#   b: v3e1 at -O2 and -Os (benchmarks), and VC1 (two 2x2 meshes) at the default level (-Os), -O2 and -O3
#                                                          labels p4eo2_, p4eos_, p4vc1e_{auto,o2,o3}_
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
case "${1:-a}" in
a)  build v3e1 $V/v3e1 $C3 O3 p4e $VAL $BENCH
    build v3e1_strict $V/v3e1_strict $C3 O3 p4es $VAL flushmix flushmix4
    build v3e1_hook $V/v3e1_hook $C3 O3 p4ehook $BENCH ;;
b)  build v3e1_O2 $V/v3e1 $C2 O2 p4eo2 $BENCH
    build v3e1_Os $V/v3e1 $CS Os p4eos $BENCH
    source $CON/vc1_lib.sh
    vc1_variant v3e1 $V/v3e1 p4vc1e auto O2 O3 ;;
esac
rm -f $M
echo ALL_DONE
