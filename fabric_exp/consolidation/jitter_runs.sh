#!/bin/bash
# Jitter band (RESULTS §96): jitter_runs.sh O3|Os [BASE [LABEL]]   (default BASE v3tk, LABEL jit)
# The benchmark suites for BASE and for each of its ordinary edits (jitter_edits.py -> fabric_exp/variants/BASE_jit/)
# at one opt level. BASE runs first and last, which gives the run-to-run noise and any drift over the job.
# Labels: LABEL3_<edit> / LABELs_<edit>, and LABEL3_base, LABEL3_base2 (LABELs_ for -Os).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
B=${2:-v3tk}; L=${3:-jit}
case "$1" in
O3) C=$C3; P=${L}3 ;;
Os) C=$CS; P=${L}s ;;
*)  echo "usage: jitter_runs.sh O3|Os [BASE [LABEL]]"; exit 1 ;;
esac
build ${B}_jit_base_$1 $V/$B $C $1 ${P}_base $BENCH
for e in reorder statefield hb coldmul rxreorder cursorfield init templates; do
    build ${B}_jit_${e}_$1 $V/${B}_jit/$e $C $1 ${P}_$e $BENCH
done
build ${B}_jit_base2_$1 $V/$B $C $1 ${P}_base2 $BENCH
rm -f $M
echo ALL_DONE
