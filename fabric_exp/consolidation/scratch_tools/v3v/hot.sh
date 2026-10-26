#!/bin/bash
# hot.sh VARIANT_DIR OUT_PREFIX: build the four hot E1 configs (mid-W, N/S, 1D, largest VC1) at -O3, -O2 and -Os.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v
V=$1; P=$2
cd /home/aagarwal/tt-metal
H="10856749301170935304,1539970155337268356,10388088869406661095,9530214107180203457"
rm -rf ${P}_O3 ${P}_O2 ${P}_Os
for l in O3 O2; do python3 $S/ob.py build --variant $V --out ${P}_$l --opt $l --hashes $H --jobs 16 2>&1 | tail -1; done
python3 $S/ob.py build --variant $V --out ${P}_Os --opt Os --hashes 10856749301170935304,1539970155337268356,10388088869406661095 --jobs 16 2>&1 | tail -1
python3 $S/ob.py build --variant $V --out ${P}_Os --opt O2 --set-opt Os --hashes 9530214107180203457 --jobs 16 2>&1 | tail -1
OD=/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
n=$(basename $P)
mkdir -p $S/dis
for l in O3 O2 Os; do
  for c in W:10856749301170935304 NS:1539970155337268356 1D:10388088869406661095 VC1:9530214107180203457; do
    k=${c%%:*}; h=${c##*:}
    e=${P}_$l/$h/subordinate_active_erisc/subordinate_active_erisc.elf
    ln -sf $e $S/dis/${n}_${l}_$k.elf
    $OD -dl $e > $S/dis/${n}_${l}_$k.dis &
  done
done
wait
