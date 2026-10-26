#!/bin/bash
# fullcheck.sh VARIANT HOTPREFIX: all-config builds at -O3/-O2/-Os (+VC1 -Os from -O2 commands), E0/E1 PT_LOAD vs v3tk,
# VC1 size, and the four ordinary edits on the hot configs (run_main_loop same/moved/DIFF).
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
V=$1; P=$2; F=$S/full/${V}K
cd /home/aagarwal/tt-metal
for l in O3 O2 Os; do python3 fabric_exp/consolidation/offline_build.py build --variant fabric_exp/variants/$V --out ${F}_$l --opt $l --jobs 12 > ${F}_$l.log 2>&1; echo "$V $l rc=$? $(tail -1 ${F}_$l.log | sed 's/ into.*//')"; done
python3 $S/ob.py build --variant fabric_exp/variants/$V --out ${F}_Os_vc1 --opt O2 --set-opt Os --hashes $(cat $S/vc1.txt) --jobs 12 > ${F}_Os_vc1.log 2>&1; echo "$V Os-vc1 $(tail -1 ${F}_Os_vc1.log | sed 's/ into.*//')"
for l in O3 O2 Os; do echo "$V $l vs v3tk: $(python3 $S/loadcmp.py $S/full/v3tk_$l ${F}_$l active_erisc subordinate_active_erisc | tr '\n' ' ')"; done
echo "$V Os-vc1 vs v3tk: $(python3 $S/loadcmp.py $S/full/v3tk_Os_vc1 ${F}_Os_vc1 active_erisc subordinate_active_erisc | tr '\n' ' ')"
python3 $S/sizes.py ${F}_O3 ${F}_O2 ${F}_Os_vc1 | sed "s#$S/full/##"
rm -rf $S/jit_$V && python3 fabric_exp/consolidation/jitter_edits.py fabric_exp/variants/$V $S/jit_$V coldmul init templates os > /dev/null 2>&1
# per-variant edit build dirs: reuse stabcheck with a unique prefix
sed "s#\$S/st_\${e}_#\$S/st_${V}_\${e}_#g" $S/stabcheck.sh > $S/stabcheck_$V.sh; chmod +x $S/stabcheck_$V.sh
$S/stabcheck_$V.sh $P $S/jit_$V coldmul init templates os
