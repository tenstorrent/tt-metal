#!/bin/bash
# Image sizes of the single-ERISC A/B's offline builds (offline.sh mock A / B; local, never committed): every router ELF
# the JIT built in each mock case, linked into size_report.py's layout (COV/builds/<arm>_O3/<case>__<hash>/<proc>/) under
# /home/aagarwal/.tt-metal-sab-size, then consolidation/size_report.py (code bytes = PT_LOAD sizes, static stack depth,
# free stack) with A as the reference.   sab_sizes.sh [CASE,...]
set -uo pipefail
S=/home/aagarwal/.tt-metal-sab-size
declare -A COVD=([A]=/home/aagarwal/.tt-metal-sab-A-cov [B]=/home/aagarwal/.tt-metal-sab-B-cov)
rm -rf $S && mkdir -p $S/builds
for arm in A B; do
    for e in ${COVD[$arm]}/cases/*/cache/*/kernels/fabric_erisc_router/*/*/*.elf; do
        case $e in *.xip.elf) continue ;; esac
        proc=$(basename $(dirname $e)); h=$(basename $(dirname $(dirname $e)))
        case=${e#${COVD[$arm]}/cases/}; case=${case%%/*}
        d=$S/builds/${arm}_O3/${case}__$h/$proc
        mkdir -p $d && ln -s $e $d/$proc.elf
    done
    echo "$arm: $(find $S/builds/${arm}_O3 -name '*.elf' | wc -l) router ELFs"
done
TT_COV_DIR=$S python3 /home/aagarwal/tt-metal/fabric_exp/consolidation/size_report.py --opt O3 ${1:+--cases $1} --detail A B
