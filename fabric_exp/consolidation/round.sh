#!/bin/bash
# Phase 1 hardware round for one candidate (variants from prep_round.sh), same protocol as rounds 2-5:
#   round.sh NAME PREFIX a   -O3: validation (RELAXED), validation + flushed-signal suites (STRICT test kernel),
#                            benchmarks; hook probe                      labels PREFIX_, PREFIXs_, PREFIXhook_
#   round.sh NAME PREFIX b [VC1 levels]   -O2 and -Os benchmarks; VC1 (two 2x2 meshes) at the given levels
#                            (default "auto O2 O3"; auto = the builder's -Os)   labels PREFIXo2_, PREFIXos_, PREFIXvc1_*
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
N=$1; P=$2; VC1L=${4:-auto O2 O3}
case "$3" in
a)  build $N $V/$N $C3 O3 $P $VAL $BENCH
    build ${N}_strict $V/${N}_strict $C3 O3 ${P}s $VAL flushmix flushmix4
    build ${N}_hook $V/${N}_hook $C3 O3 ${P}hook $BENCH ;;
b)  build ${N}_O2 $V/$N $C2 O2 ${P}o2 $BENCH
    build ${N}_Os $V/$N $CS Os ${P}os $BENCH
    source $CON/vc1_lib.sh
    source $CON/vc1_capture_lib.sh   # every VC1 run has the read-only hang capture armed (RESULTS §94)
    VC1_RUN=run_vc1_cap vc1_variant $N $V/$N ${P}vc1 $VC1L ;;
*)  echo "usage: round.sh NAME PREFIX a|b"; exit 1 ;;
esac
rm -f $M
echo ALL_DONE
