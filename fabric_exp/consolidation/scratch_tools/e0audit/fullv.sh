#!/bin/bash
# fullv.sh VARIANT_PATH NAME [JOBS]: like full.sh, for a variant given by path.
E=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/e0audit
V=$1; N=$2; J=${3:-24}; F=$E/full/$N
cd /home/aagarwal/tt-metal
for l in O3 O2 Os; do rm -rf ${F}_$l; python3 $E/tools/ob.py build --variant $V --out ${F}_$l --opt $l --jobs $J > ${F}_$l.log 2>&1; echo "$N $l rc=$? $(tail -1 ${F}_$l.log | sed 's/ into.*//')"; done
rm -rf ${F}_Os_vc1; python3 $E/tools/ob.py build --variant $V --out ${F}_Os_vc1 --opt O2 --set-opt Os --hashes $(cat $E/tools/vc1.txt) --jobs $J > ${F}_Os_vc1.log 2>&1; echo "$N Os-vc1 $(tail -1 ${F}_Os_vc1.log | sed 's/ into.*//')"
