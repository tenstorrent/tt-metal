#!/bin/bash
# diagcheck.sh VARIANT REF: all-config builds at -O3/-O2/-Os (+VC1 -Os from -O2 commands), E0/E1 PT_LOAD vs REF's full builds.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
V=$1; R=$2; F=$S/full/${V}K
cd /home/aagarwal/tt-metal
for l in O3 O2 Os; do python3 fabric_exp/consolidation/offline_build.py build --variant fabric_exp/variants/$V --out ${F}_$l --opt $l --jobs 12 > ${F}_$l.log 2>&1; echo "$V $l rc=$? $(tail -1 ${F}_$l.log | sed 's/ into.*//')"; done
python3 $S/ob.py build --variant fabric_exp/variants/$V --out ${F}_Os_vc1 --opt O2 --set-opt Os --hashes $(cat $S/vc1.txt) --jobs 12 > ${F}_Os_vc1.log 2>&1; echo "$V Os-vc1 $(tail -1 ${F}_Os_vc1.log | sed 's/ into.*//')"
for l in O3 O2 Os Os_vc1; do echo "$V $l vs $R: $(python3 $S/loadcmp.py $S/full/${R}_$l ${F}_$l active_erisc subordinate_active_erisc | tr '\n' ' ')"; done
python3 $S/sizes.py ${F}_O3 ${F}_O2 ${F}_Os_vc1 | sed "s#$S/full/##"
