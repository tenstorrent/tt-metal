#!/bin/bash
# Timing-perturbation sweep (RESULTS §98; V3E1_NOTES §12.18): pert_runs.sh A|B. DIAGNOSTIC builds <base>_pA<d> (spin d
# core cycles right before the completion check) and <base>_pB<d> (right after a packet's local write is issued),
# d = 8..32, with the base itself as d = 0. Suites 4352, mix, 1d at -O3. A: v3tk, v3v6. B: v3w, v3w4.
# Labels p20_<base>_0, p20_<base>_pA<d>, p20_<base>_pB<d>.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
case "$1" in A) BASES="v3tk v3v6" ;; B) BASES="v3w v3w4" ;; *) echo "usage: pert_runs.sh A|B"; exit 1 ;; esac
for b in $BASES; do
    build ${b}_pert0 $V/$b $C3 O3 p20_${b}_0 4352 mix 1d
    for p in A B; do for d in 8 16 24 32; do
        build ${b}_p$p$d $V/${b}_p$p$d $C3 O3 p20_${b}_p$p$d 4352 mix 1d
    done; done
done
rm -f $M
echo ALL_DONE
