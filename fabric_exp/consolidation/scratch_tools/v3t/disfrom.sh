#!/bin/bash
# disfrom.sh BUILD_PREFIX NAME: hot-config ELF links and disassembly from full builds (<prefix>_O3, _O2, _Os, _Os_vc1).
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3t
P=$1; n=$2
OD=/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
for l in O3 O2 Os; do
  for c in W:10856749301170935304 NS:1539970155337268356 1D:10388088869406661095 VC1:9530214107180203457; do
    k=${c%%:*}; h=${c##*:}; d=${P}_$l
    [ $l = Os ] && [ $k = VC1 ] && d=${P}_Os_vc1
    e=$d/$h/subordinate_active_erisc/subordinate_active_erisc.elf
    ln -sf $e $S/dis/${n}_${l}_$k.elf
    $OD -dl $e > $S/dis/${n}_${l}_$k.dis &
  done
done
wait
