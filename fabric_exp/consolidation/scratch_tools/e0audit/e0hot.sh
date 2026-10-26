#!/bin/bash
# e0hot.sh VARIANT_DIR OUT_PREFIX: build the five hot E0 configs (2D E/W, 2D N/S, 1D line, 1D ring, largest VC1) at -O3, -O2, -Os.
E=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/e0audit
V=$1; P=$2
cd /home/aagarwal/tt-metal
H="10733501891889981293,8414232161770541884,10744374302904868627,16671252331734229980,1526198486225049198"
rm -rf ${P}_O3 ${P}_O2 ${P}_Os
for l in O3 O2; do python3 $E/tools/ob.py build --variant $V --out ${P}_$l --opt $l --hashes $H --jobs 10 > ${P}_$l.log 2>&1; done
python3 $E/tools/ob.py build --variant $V --out ${P}_Os --opt Os --hashes 10733501891889981293,8414232161770541884,10744374302904868627,16671252331734229980 --jobs 10 > ${P}_Os.log 2>&1
python3 $E/tools/ob.py build --variant $V --out ${P}_Os --opt O2 --set-opt Os --hashes 1526198486225049198 --jobs 10 >> ${P}_Os.log 2>&1
echo "$(basename $P): $(grep -h 'built' ${P}_O3.log ${P}_O2.log ${P}_Os.log | sed 's/ into.*//' | tr '\n' ' ')"
