#!/bin/bash
# Jitter band (RESULTS §96): jitter_runs.sh O3|Os
# The benchmark suites for v3tk and for each of its ordinary edits (jitter_edits.py -> fabric_exp/variants/v3tk_jit/)
# at one opt level. v3tk runs first and last, which gives the run-to-run noise and any drift over the job.
# Labels: jit3_<edit> / jits_<edit>, and jit3_base, jit3_base2 (jits_ for -Os).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
case "$1" in
O3) C=$C3; P=jit3 ;;
Os) C=$CS; P=jits ;;
*)  echo "usage: jitter_runs.sh O3|Os"; exit 1 ;;
esac
build v3tk_jit_base_$1 $V/v3tk $C $1 ${P}_base $BENCH
for e in reorder statefield hb coldmul rxreorder cursorfield init templates; do
    build v3tk_jit_${e}_$1 $V/v3tk_jit/$e $C $1 ${P}_$e $BENCH
done
build v3tk_jit_base2_$1 $V/v3tk $C $1 ${P}_base2 $BENCH
rm -f $M
echo ALL_DONE
