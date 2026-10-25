#!/bin/bash
# Phase 1, round 2: v3u (E0 uniform runtime channels; E1 = v3a) against v3a. Variants from prep_round.sh.
#   a: v3u -O3: validation (RELAXED), validation + flushed-signal suites (STRICT test kernel), benchmarks, hook probe
#                                                          labels p2u_, p2us_ (strict), p2uhook_
#   b: stability at -O2/-Os for v3u and v3a, and v3a's hook probe (v3a's -O3 numbers = s79m without hints, §83)
#                                                          labels p2uo2_, p2uos_, p2ao2_, p2aos_, p2ahook_
#   a2: a's STRICT and hook builds (job 218 stopped after a's first build on a false "no ELFs built" check)
#   c: VC1 (two 2x2 meshes) for v3u at the default level (-Os), -O2 and -O3 (v3u fits at -O3)
#                                                          labels p2vc1u_auto_, p2vc1u_o2_, p2vc1u_o3_
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
case "${1:-a}" in
a)  build v3u $V/v3u $C3 O3 p2u $VAL $BENCH
    build v3u_strict $V/v3u_strict $C3 O3 p2us $VAL flushmix flushmix4
    build v3u_hook $V/v3u_hook $C3 O3 p2uhook $BENCH ;;
a2) build v3u_strict $V/v3u_strict $C3 O3 p2us $VAL flushmix flushmix4
    build v3u_hook $V/v3u_hook $C3 O3 p2uhook $BENCH ;;
b)  build v3u_O2 $V/v3u $C2 O2 p2uo2 $BENCH
    build v3u_Os $V/v3u $CS Os p2uos $BENCH
    build v3a_O2 $V/v3a $C2 O2 p2ao2 $BENCH
    build v3a_Os $V/v3a $CS Os p2aos $BENCH
    build v3a_hook $V/v3a_hook $C3 O3 p2ahook $BENCH ;;
c)  source $CON/vc1_lib.sh
    vc1_variant v3u $V/v3u p2vc1u auto O2 O3 ;;
esac
rm -f $M
echo ALL_DONE
